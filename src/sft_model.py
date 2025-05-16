import ast
import json
import os
import re
import numpy as np
import pandas as pd
import wandb
import yaml
import sys
from typing import List, Dict, Tuple, Optional, Any
import random
from datasets import Dataset

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    BitsAndBytesConfig,
)

from trl import SFTConfig
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer


def calculate_grid_dimension_score(possible: list[list], solution: list[list]) -> bool:
    """
    Calculate the grid dimension score between the model's output and the solution.
    """
    if is_jagged(possible):
        return False
    possible = np.array(possible)
    solution = np.array(solution)

    if possible.shape != solution.shape:
        return False

    return True


def is_jagged(grid: list[list]) -> bool:
    """
    Check if the grid is jagged (not all rows have the same length).
    """
    row_lengths = [len(row) for row in grid]
    return len(set(row_lengths)) > 1


def calculate_cell_value_match_normalised(
    possible: list[list], solution: list[list]
) -> float:
    """
    Calculate the cell value match score between the model's output and the solution.
    """
    if is_jagged(possible):
        return 0.0
    possible = np.array(possible)
    solution = np.array(solution)

    if possible.shape != solution.shape:
        return 0.0

    matching_cells = np.sum(possible == solution)

    total_cells = possible.size

    score = matching_cells / total_cells

    return score


def calculate_cell_value_match_non_zero(possible: list[list], solution: list[list]):
    """
    Calculate the cell value match score between the model's output and the solution,
    ignoring zero values.
    """
    if is_jagged(possible):
        return 0.0

    possible = np.array(possible)
    solution = np.array(solution)

    if possible.shape != solution.shape:
        return 0.0

    solution_non_zero = np.where(solution != 0, 1.0, 0.0)

    possible_correct = np.where(possible == solution, 1.0, 0.0)

    all_correct_non_zeros = solution_non_zero * possible_correct

    return np.sum(all_correct_non_zeros) / np.sum(solution_non_zero)


def calculate_exact_match_score(possible: list[list], solution: list[list]) -> bool:
    """
    Calculate the exact match score between the model's output and the solution.
    """
    if is_jagged(possible):
        return False

    if not calculate_grid_dimension_score(possible, solution):
        return False

    return all(
        [
            sublist_possible == sublist_solution
            for sublist_possible, sublist_solution in zip(possible, solution)
        ]
    )







def find_last_list_of_lists_with_indices(
    text: str,
) -> Optional[Tuple[List[List[Any]], int, int]]:
    """
    Finds the last substring representing a JSON list of lists in the text.
    Scans backward to ensure it finds the *last* valid structure.
    Returns: tuple: (parsed_list, start_index, end_index) if found, otherwise None.
    """
    last_checked_end = len(text)
    while True:
        end_bracket_index = text.rfind("]", 0, last_checked_end)
        if end_bracket_index == -1:
            return None

        balance = 0
        start_bracket_index = -1
        for i in range(end_bracket_index, -1, -1):
            char = text[i]
            if char == "]":
                balance += 1
            elif char == "[":
                balance -= 1
                if balance == 0:
                    start_bracket_index = i
                    break

        if start_bracket_index != -1:
            potential_json_str = text[start_bracket_index : end_bracket_index + 1]
            try:
                
                if not potential_json_str.startswith(
                    "[["
                ) or not potential_json_str.endswith("]]"):
                    if not (
                        potential_json_str.startswith("[]")
                        and len(potential_json_str) == 2
                    ):  
                        if not (
                            potential_json_str.startswith("[[]")
                            and potential_json_str.endswith("]]")
                        ):  
                            raise json.JSONDecodeError(
                                "Does not start/end with '[[' ']]'",
                                potential_json_str,
                                0,
                            )

                parsed_data = json.loads(potential_json_str)

                
                if isinstance(parsed_data, list) and all(
                    isinstance(item, list) for item in parsed_data
                ):
                    
                    
                    return parsed_data, start_bracket_index, end_bracket_index + 1

            except json.JSONDecodeError:
                pass  

        
        last_checked_end = end_bracket_index







class LogGenerationCallback(TrainerCallback):
    """
    A TrainerCallback that generates text from evaluation examples,
    parses results, calculates scores, and logs details and aggregates to W&B.
    """

    def __init__(
        self,
        eval_examples: List[
            Dict[str, Any]
        ],  
        tokenizer: AutoTokenizer,
        num_examples: int = 5,
        wandb_table_max_str_len: int = 20000,
        solution_map: Dict[str, str] = None,
        config: Dict[str, Any] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.wandb_table_max_str_len = wandb_table_max_str_len
        self.solution_map = solution_map
        self.config = config

        
        self.num_examples = min(num_examples, len(eval_examples))
        if self.num_examples > 0:
            self.eval_examples = random.sample(eval_examples, self.num_examples)
            print(
                f"LogGenerationCallback: Initialized with {self.num_examples} evaluation examples."
            )
        else:
            self.eval_examples = []
            print(
                "Warning: LogGenerationCallback initialized with zero evaluation examples."
            )

        
        self.eos_token_id = tokenizer.eos_token_id
        if (
            self.eos_token_id is None
            and hasattr(tokenizer, "pad_token_id")
            and tokenizer.pad_token_id is not None
        ):
            print(
                "LogGenerationCallback Warning: EOS token ID not found, using PAD token ID as EOS for generation check."
            )
            self.eos_token_id = tokenizer.pad_token_id
        elif self.eos_token_id is None:
            print(
                "LogGenerationCallback Warning: No EOS or PAD token ID found in tokenizer. Cannot reliably check for EOS termination."
            )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: AutoModelForCausalLM,
        **kwargs,
    ):
        if not wandb.run:
            print("LogGenerationCallback: W&B run not initialized. Skipping logging.")
            return
        if not self.eval_examples:
            print("LogGenerationCallback: No evaluation examples to process.")
            return

        generations = []  
        model.eval()
        device = model.device

        print(
            f"\n--- LogGenerationCallback: Generating and Scoring {self.num_examples} examples (Step: {state.global_step}) ---"
        )

        for i, eval_example in enumerate(self.eval_examples):
            task_id = eval_example.get("task_id", "UNKNOWN_TASK_ID")

            
            generation_log = {
                "step": state.global_step,
                "task_id": task_id,
                "prompt": "[PROMPT GENERATION FAILED]",
                "solution": "[SOLUTION EXTRACTION FAILED]",
                "generated_text": "[GENERATION PENDING]",
                "parsed_prediction": None,
                "grid_dimension_match": False,
                "pixel_match": 0.0,
                "exact_match": False,
                "foreground_pixel_match": 0.0,
                "parse_error_pred": None,
                "num_generated_tokens": 0,
                "hit_max_length": False,
                "error": None,
            }
            messages = eval_example.get("messages")
            solution = self.solution_map.get(task_id, None)
            solution = ast.literal_eval(solution) if solution else None

            if solution is None:
                print(
                    f"  Warning: No solution found for Task ID {task_id}. Skipping this example."
                )
                sys.exit(1)

            generation_log["solution"] = json.dumps(solution)

            if not messages or not isinstance(messages, list):
                print(
                    f"  Skipping example {i + 1} (Task ID: {task_id}) due to missing/invalid messages."
                )
                generation_log["error"] = "Missing or invalid messages in eval_example"
                generations.append(generation_log)
                continue

            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=False)
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                prompt_token_length = input_ids.shape[1]
                generation_log["prompt"] = self.tokenizer.decode(
                    input_ids[0], skip_special_tokens=False
                )
            except Exception as e:
                print(f"  Error preparing prompt for Task ID {task_id}: {e}")
                generation_log["error"] = f"Prompt/Tokenization error: {str(e)}"
                generations.append(generation_log)
                continue

            try:
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.config["baseline_models"]["max_tokens"],
                        eos_token_id=self.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.eos_token_id,
                        do_sample=False,
                    )
                generated_sequence = outputs[0]
                generated_token_ids = generated_sequence[prompt_token_length:]
                num_generated_tokens = len(generated_token_ids)

                generated_text = self.tokenizer.decode(
                    generated_token_ids, skip_special_tokens=True
                )
                generation_log["generated_text"] = generated_text
                generation_log["num_generated_tokens"] = num_generated_tokens

                
                hit_max_length = (
                    num_generated_tokens >= self.config["baseline_models"]["max_tokens"]
                )
                last_gen_token_id = (
                    generated_token_ids[-1].item() if num_generated_tokens > 0 else None
                )
                if (
                    self.eos_token_id is not None
                    and last_gen_token_id == self.eos_token_id
                ):
                    termination_reason = "EOS token"
                elif hit_max_length:
                    termination_reason = "Max length"
                else:
                    termination_reason = f"Stopped early (token: {last_gen_token_id})"
                generation_log["termination_reason"] = termination_reason
                generation_log["hit_max_length"] = hit_max_length

                
                predicted_grid = None

                
                try:
                    parse_result = find_last_list_of_lists_with_indices(generated_text)
                    if parse_result:
                        predicted_grid, _, _ = parse_result
                        generation_log["parsed_prediction"] = json.dumps(predicted_grid)
                    else:
                        generation_log["parse_error_pred"] = (
                            "No grid found in generated text"
                        )
                except Exception as e:
                    generation_log["parse_error_pred"] = (
                        f"Pred Parsing Exception: {str(e)}"
                    )

                
                if predicted_grid is not None and solution is not None:
                    if is_jagged(predicted_grid):
                        generation_log["parse_error_pred"] = (
                            generation_log.get("parse_error_pred") or ""
                        ) + "; Is Jagged"
                        

                    generation_log["grid_dimension_match"] = (
                        calculate_grid_dimension_score(predicted_grid, solution)
                    )
                    generation_log["pixel_match"] = (
                        calculate_cell_value_match_normalised(predicted_grid, solution)
                    )
                    generation_log["exact_match"] = calculate_exact_match_score(
                        predicted_grid, solution
                    )
                    generation_log["foreground_pixel_match"] = (
                        calculate_cell_value_match_non_zero(predicted_grid, solution)
                    )
                    print(
                        f"  Scored Task {task_id}: Exact Match={generation_log['exact_match']}, Grid Dimension Match={generation_log['grid_dimension_match']}, Pixel Match={generation_log['pixel_match']:.3f}, Foreground Pixel Match={generation_log['foreground_pixel_match']:.3f}"
                    )

                    print("")
                elif predicted_grid is None:
                    print(
                        f"  Skipped scoring Task {task_id}: Prediction parsing failed."
                    )
                elif solution is None:
                    print(
                        f"  Skipped scoring Task {task_id}: Gold answer parsing failed."
                    )

                

            except Exception as e:
                error_msg = f"Error during generation/scoring: {str(e)}"
                print(f"  {error_msg} for Task ID {task_id}, Example 
                generation_log["error"] = (
                    error_msg  
                )

            generations.append(generation_log)  

        
        num_total_examples_processed = len(
            generations
        )  
        successful_scores_list = []

        for log_entry in generations:
            
            
            pred_ok = (
                log_entry.get("parse_error_pred") is None
                and log_entry.get("parsed_prediction") is not None
            )
            gold_ok = log_entry.get("parse_error_gold") is None
            overall_ok = log_entry.get("error") is None

            if pred_ok and gold_ok and overall_ok:
                successful_scores_list.append(
                    {
                        "grid_dimension_match": log_entry["grid_dimension_match"],
                        "pixel_match": log_entry["pixel_match"],
                        "exact_match": log_entry["exact_match"],
                        "foreground_pixel_match": log_entry["foreground_pixel_match"],
                    }
                )

        num_successful = len(successful_scores_list)
        num_failed = num_total_examples_processed - num_successful

        
        if num_successful > 0:
            avg_scores = {
                f"eval/avg_grid_dimension_match_{self.num_examples}samples": np.mean(
                    [s["grid_dimension_match"] for s in successful_scores_list]
                ),
                f"eval/avg_pixel_match_{self.num_examples}samples": np.mean(
                    [s["pixel_match"] for s in successful_scores_list]
                ),
                f"eval/avg_exact_match_{self.num_examples}samples": np.mean(
                    [s["exact_match"] for s in successful_scores_list]
                ),
                f"eval/avg_foreground_pixel_match_{self.num_examples}samples": np.mean(
                    [s["foreground_pixel_match"] for s in successful_scores_list]
                ),
            }
        else:  
            avg_scores = {
                f"eval/avg_grid_dimension_match_{self.num_examples}samples": 0.0,
                f"eval/avg_pixel_match_{self.num_examples}samples": 0.0,
                f"eval/avg_exact_match_{self.num_examples}samples": 0.0,
                f"eval/avg_foreground_pixel_match_{self.num_examples}samples": 0.0,
            }

        avg_scores.update(
            {
                f"eval/num_callback_examples_processed_{self.num_examples}samples": num_total_examples_processed,
                f"eval/num_successful_scored_{self.num_examples}samples": num_successful,
                f"eval/num_failed_scored_{self.num_examples}samples": num_failed,
                f"eval/score_success_rate_{self.num_examples}samples": (
                    num_successful / num_total_examples_processed * 100
                )
                if num_total_examples_processed > 0
                else 0.0,
            }
        )

        try:
            wandb.log(avg_scores)
        except Exception as e:
            print(
                f"LogGenerationCallback Error: Failed logging aggregate scores to W&B: {e}"
            )

        try:
            if self.wandb_table_max_str_len > 0:
                wandb.Table._MAX_STR_LEN = self.wandb_table_max_str_len

            df_generations = pd.DataFrame(generations)

            
            
            
            

            
            cols_order = [
                "step",
                "task_id",
                "prompt",
                "solution",
                "parsed_prediction",
                "generated_text",
                "grid_dimension_match",
                "pixel_match",
                "foreground_pixel_match",
                "error",
                "parse_error_pred",
                "num_generated_tokens",
                "termination_reason",
                "hit_max_length",  
            ]
            
            existing_cols = [col for col in cols_order if col in df_generations.columns]
            df_generations = df_generations[existing_cols]  

            wandb.log(
                {"evaluation_generations": wandb.Table(dataframe=df_generations)},
                step=state.global_step,
            )
        except ImportError:  
            print(
                "LogGenerationCallback Warning: pandas not found. Logging table as list of dicts."
            )
            wandb.log({"evaluation_generations": generations}, step=state.global_step)
        except Exception as e:
            print(
                f"LogGenerationCallback Error: Failed logging generations table to W&B: {e}"
            )
        

        print(
            f"--- LogGenerationCallback: Finished logging ({num_successful}/{num_total_examples_processed} scored successfully) ---"
        )


def train_model(train_dataset, eval_dataset, solutions, config):
    print(
        f"Starting fine-tuning process for base model: {config['baseline_models']['default_model']}"
    )

    if torch.cuda.is_available():
        print(f"CUDA available. Using {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Training on CPU (will be very slow).")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["baseline_models"]["default_model"]
    )

    pad_token_added = False
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer does not have a pad token.")
        if hasattr(tokenizer, "unk_token") and tokenizer.unk_token is not None:
            print("Using unk_token as pad_token")
            tokenizer.pad_token = tokenizer.unk_token
        else:
            print(
                "Adding [PAD] as pad_token - will resize model embeddings after model loading"
            )
            special_tokens_dict = {"pad_token": "[PAD]"}
            tokenizer.add_special_tokens(special_tokens_dict)
            pad_token_added = True
        tokenizer.padding_side = "right"
        print(
            f"Tokenizer loaded. Pad token: {tokenizer.pad_token}, EOS token: {tokenizer.eos_token}"
        )

    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False

    use_peft = True
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1

    lora_target_modules = [
        "q_proj",
        "k_proj",
    ]

    num_train_epochs = config["baseline_models"]["train_epochs"]
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1
    gradient_accumulation_steps = 8
    max_grad_norm = 0.3
    learning_rate = 2e-4
    weight_decay = 0.001
    optimizer_type = "paged_adamw_32bit"
    lr_scheduler_type = "cosine"
    max_steps = -1
    warmup_ratio = 0.03
    logging_steps = 1
    save_steps = 10

    print("Loading base model...")
    quantization_config = None
    if use_4bit:
        print("Using 4-bit quantization (BNB).")
        try:
            compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_nested_quant,
            )
        except ImportError:
            print(
                "Error: bitsandbytes library not found. Please install it to use 4-bit quantization."
            )
            print("pip install bitsandbytes")
            return
        except AttributeError:
            print(
                f"Error: Compute dtype '{bnb_4bit_compute_dtype}' not found in torch. Using float16."
            )
            compute_dtype = torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_nested_quant,
            )

    model = AutoModelForCausalLM.from_pretrained(
        config["baseline_models"]["default_model"],
        quantization_config=quantization_config,
    )

    if pad_token_added:
        print("Resizing model embeddings to account for new pad token")
    model.resize_token_embeddings(len(tokenizer))

    print("Base model loaded.")

    peft_config = None

    if use_peft:
        print("Attempting to configure PEFT (LoRA).")

        if use_4bit:
            if quantization_config:
                print("Preparing model for k-bit training (PEFT + Quantization).")
                model = prepare_model_for_kbit_training(model)
            else:
                print(
                    "Warning: Quantization config failed or not used, cannot prepare for k-bit training."
                )

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )
        print("PEFT config created successfully.")

    print("Defining training arguments...")

    training_arguments = SFTConfig(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optimizer_type if use_4bit else "adamw_torch",
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=False if bnb_4bit_compute_dtype == "bfloat16" else True,
        bf16=True if bnb_4bit_compute_dtype == "bfloat16" else False,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        eval_strategy="steps",
        eval_steps=config["baseline_models"]["eval_steps"],
        save_strategy="steps",
        save_steps=config["baseline_models"]["save_steps"],
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        max_length=config["baseline_models"]["max_tokens"],
    )

    print("Training arguments defined.")

    wandb.init(
        project="arc-fine-tuning",
        name=f"{config['baseline_models']['default_model']}-arc-sft",
        config={
            "model": config["baseline_models"]["default_model"],
            "batch_size": per_device_train_batch_size,
            "learning_rate": learning_rate,
            "epochs": num_train_epochs,
            "use_peft": use_peft,
            "use_4bit": use_4bit,
            "eval_steps": config["baseline_models"]["eval_steps"],
        },
    )

    eval_examples_list = []
    if (
        eval_dataset
        and "messages" in eval_dataset.column_names
        and "task_id" in eval_dataset.column_names
    ):
        if isinstance(eval_dataset, Dataset):
            num_samples = min(10, len(eval_dataset))
            print(f"Selecting {num_samples} examples for evaluation callback pool.")
            for i in range(num_samples):
                eval_examples_list.append(eval_dataset[i])
        else:
            print(
                "Warning: eval_dataset is not indexable or not a Dataset. Cannot extract examples for callback."
            )
    elif not eval_dataset:
        print("Warning: eval_dataset is None. Cannot extract examples for callback.")
    else:
        print(
            f"Warning: eval_dataset columns missing 'messages' or 'task_id'. Columns: {eval_dataset.column_names}. Cannot extract examples for callback."
        )

    if not eval_examples_list:
        print(
            "Warning: No evaluation examples collected for LogGenerationCallback. Callback might not function correctly."
        )

    
    log_generation_callback = LogGenerationCallback(
        eval_examples=eval_examples_list,
        solution_map=solutions,
        tokenizer=tokenizer,
        num_examples=5,
        config=config,
    )

    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  
        peft_config=peft_config if use_peft else None,
        args=training_arguments,
        callbacks=[log_generation_callback],
    )

    print("*** Starting Training ***")
    train_result = trainer.train()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print("Training metrics saved.")

    wandb.finish()


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: Config file not found at {path}. Using default empty config.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error loading or parsing config file {path}: {e}", file=sys.stderr)
        return {}
    except Exception as e:
        print(
            f"An unexpected error occurred loading config {path}: {e}", file=sys.stderr
        )
        return {}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Loads data from a JSON Lines file."""
    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Skipping line {i + 1} in {path} due to JSON decode error: {e}"
                    )
        return data
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}", file=sys.stderr)
        return []
    except Exception as e:
        print(
            f"An unexpected error occurred loading JSONL {path}: {e}", file=sys.stderr
        )
        return []


def load_json(path: str) -> Dict[str, Any]:
    """Loads data from a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}", file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {path}: {e}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"An unexpected error occurred loading JSON {path}: {e}", file=sys.stderr)
        return {}


def grid_to_str(grid: list[list[int]]):
    """Converts a grid to a string representation."""
    grid_strs = []
    for row in grid:
        row_str = ", ".join([str(x) for x in row])
        grid_strs.append("[" + row_str + "]")
    return "[" + ", ".join(grid_strs) + "]"


def find_last_list_of_lists_with_indices(
    text: str,
) -> Optional[Tuple[List[List[Any]], int, int]]:
    """
    Finds the last substring representing a JSON list of lists in the text.

    Scans backward to ensure it finds the *last* valid structure.

    Returns:
        tuple: (parsed_list, start_index, end_index) if found, otherwise None.
               end_index is exclusive, suitable for slicing text[:end_index].
    """
    last_checked_end = len(text)
    while True:
        end_bracket_index = text.rfind("]", 0, last_checked_end)
        if end_bracket_index == -1:
            return None

        balance = 0
        start_bracket_index = -1

        for i in range(end_bracket_index, -1, -1):
            char = text[i]
            if char == "]":
                balance += 1
            elif char == "[":
                balance -= 1
                if balance == 0:
                    start_bracket_index = i
                    break

        if start_bracket_index != -1:
            potential_json_str = text[start_bracket_index : end_bracket_index + 1]
            try:
                parsed_data = json.loads(potential_json_str)

                if (
                    isinstance(parsed_data, list)
                    and parsed_data
                    and all(isinstance(item, list) for item in parsed_data)
                ):
                    return parsed_data, start_bracket_index, end_bracket_index + 1
            except json.JSONDecodeError:
                pass
        last_checked_end = end_bracket_index


def reformat_raw_response(
    input_cot_data: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """
    Reformats the 'raw_response' in CoT data by replacing the last JSON grid
    with its compact string representation.

    Args:
        input_cot_data: List of dictionaries, each potentially containing 'raw_response'.

    Returns:
        A tuple containing:
        - A new list of dictionaries with modified 'raw_response' where applicable.
        - The number of entries modified.
        - The total character count before modification (for modified entries).
        - The total character count after modification (for modified entries).
    """
    output_cot_data = []
    modified_count = 0
    char_count_previous = 0
    char_count_after = 0

    for data in input_cot_data:
        new_data = data.copy()
        raw_response = new_data.get("raw_response")

        if isinstance(raw_response, str):
            find_result = find_last_list_of_lists_with_indices(raw_response)

            if find_result:
                parsed_grid, start_index, end_index = find_result
                new_grid_string = grid_to_str(parsed_grid)

                new_response = (
                    raw_response[:start_index]
                    + new_grid_string
                    + raw_response[end_index:]
                )

                new_data["raw_response"] = new_response
                modified_count += 1
                char_count_previous += len(raw_response)
                char_count_after += len(new_response)

        output_cot_data.append(new_data)

    return output_cot_data, modified_count, char_count_previous, char_count_after


def create_task_prompt_section(task_data: Dict[str, Any]) -> str:
    """Formats the examples and test input for a single ARC task."""
    task_id = task_data.get("task_id", "UNKNOWN")
    prompt_section = ""

    if task_data.get("train"):
        for i, pair in enumerate(task_data["train"]):
            if isinstance(pair, dict) and "input" in pair and "output" in pair:
                prompt_section += (
                    f"Example {i + 1} Input:\n{grid_to_str(pair['input'])}\n"
                )
                prompt_section += (
                    f"Example {i + 1} Output:\n{grid_to_str(pair['output'])}\n\n"
                )
            else:
                print(
                    f"Warning: Malformed train pair {i + 1} in task {task_id}. Data: {pair}"
                )
                prompt_section += f"Example {i + 1}: [Malformed train pair data]\n\n"
    else:
        print(f"Warning: No 'train' data found in task {task_id}")

    test_cases = task_data.get("test", [])
    if test_cases and isinstance(test_cases, list):
        first_test_case = test_cases[0]
        if isinstance(first_test_case, dict) and "input" in first_test_case:
            test_input_grid = first_test_case["input"]
            prompt_section += f"Test Input:\n{grid_to_str(test_input_grid)}\n"
        else:
            print(
                f"Warning: Test case 0 in task {task_id} is malformed or missing 'input'. Data: {first_test_case}"
            )
            prompt_section += f"Test Input: [Test case 0 exists but missing 'input' grid in task {task_id}]\n"
    else:
        print(
            f"Warning: No 'test' data or invalid format for task {task_id}. Data: {test_cases}"
        )
        prompt_section += (
            f"Test Input: [No valid test input data provided for task {task_id}]\n"
        )

    return prompt_section.strip()


def generate_dataset(config):
    cot_data = load_jsonl(config["filtered_sft_output_file"])
    challenges_raw = load_json(
        os.path.join(config["arc_data_dir"], config["training_challenges_file"])
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.get("baseline_models")["default_model"]
    )

    if tokenizer.chat_template is None:
        print("Error: Could not load Chat T. Exiting.", file=sys.stderr)
        sys.exit(1)

    if not cot_data or not challenges_raw:
        print("Error: Could not load critical data files. Exiting.", file=sys.stderr)
        sys.exit(1)

    task_id_to_data = {
        task_id: {**data, "task_id": task_id}
        for task_id, data in challenges_raw.items()
    }
    task_id_to_data = {task_id: data for task_id, data in challenges_raw.items()}

    process_stats = {
        "final_sft_data": [],
        "total_final_chars": 0,
        "processed_ids": set(),
        "token_counts": [],
        "skipped_long_count": 0,
        "token_lengths": [],
    }

    final_sft_data = []
    solutions = {}

    for data in cot_data:
        task_id = data.get("task_id")

        if not task_id:
            print(
                f"Warning: Skipping entry due to missing 'task_id'. Data snippet: {str(data)[:100]}..."
            )
            continue

        if data.get("task_id") not in task_id_to_data:
            print(
                f"Warning: Skipping Task ID {task_id} as it's not found in challenges data."
            )
            continue
        challenge_data = task_id_to_data[task_id]

        try:
            task_prompt_section = create_task_prompt_section(challenge_data)
            prompt = config["CONCISE_BASE_TEMPLATE"].format(
                task_prompt_section=task_prompt_section
            )
        except Exception as e:
            print(f"Error creating prompt for Task ID '{task_id}': {e}. Skipping.")
            continue

        if config["reasoning_sft"]:
            reasoning_raw = data.get("raw_response")
            if reasoning_raw is None:
                reasoning = "[MISSING REASONING]"
            elif isinstance(reasoning_raw, list):
                reasoning = (
                    str(reasoning_raw[0]) if reasoning_raw else "[EMPTY REASONING]"
                )
            elif not isinstance(reasoning_raw, str):
                reasoning = str(reasoning_raw)
            else:
                reasoning = reasoning_raw

        model_answer_grid = data.get("model_answer")
        answer_str = (
            grid_to_str(model_answer_grid)
            if model_answer_grid is not None
            else "[MISSING ANSWER]"
        )

        solutions[task_id] = answer_str

        if config["reasoning_sft"]:
            reasoning_grid = find_last_list_of_lists_with_indices(reasoning)
            if reasoning_grid:
                reasoning = (
                    reasoning[: reasoning_grid[1]] + reasoning[reasoning_grid[2] :]
                )

            reasoning = reasoning.replace("<thinking>", "").replace("</thinking>", "")
            reasoning = reasoning.replace("<answer>", "").replace("</answer>", "")
            reasoning = reasoning.replace("answer", "")
            reasoning = reasoning.replace("thinking", "")

            reasoning = re.sub(r"\n{2,}", "\n", reasoning)
            reasoning = re.sub(r" {2,}", " ", reasoning)
            reasoning = re.sub(r"[ \t]+\n", "\n", reasoning)
            reasoning = re.sub(r"\n[ \t]+", "\n", reasoning)
            reasoning = reasoning.strip()

            model_response_content = (
                f"<thinking>{reasoning}</thinking><answer>{answer_str}</answer>"
            )
        else:
            model_response_content = answer_str

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": model_response_content},
        ]

        try:
            formatted_string = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            print(
                f"Error applying chat template for Task ID {data.get('task_id')}: {e}"
            )
            print(f"Messages structure: {messages}")
            formatted_string = f"USER: {prompt}\nASSISTANT: {model_response_content}"

        num_tokens = len(tokenizer.encode(formatted_string))
        max_token_length = config.get("max_token_length", 8000)
        process_stats["token_lengths"].append(num_tokens)

        if num_tokens > max_token_length:
            process_stats["skipped_long_count"] += 1
            continue
        process_stats["token_counts"].append(num_tokens)

        final_entry = {
            "task_id": task_id,
            "text": formatted_string,
            "messages": messages,
        }

        final_sft_data.append(final_entry)

        process_stats["final_sft_data"].append(final_entry)
        process_stats["total_final_chars"] += len(formatted_string)
        process_stats["processed_ids"].add(task_id)

    print(
        f"\nSkipped {process_stats['skipped_long_count']} entries due to exceeding token limit ({max_token_length})."
    )
    print(
        f"Processed {len(process_stats['final_sft_data'])} entries for {len(process_stats['processed_ids'])} unique Task IDs."
    )

    if not final_sft_data:
        print(
            "Error: No data available to create datasets after filtering. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    random.shuffle(final_sft_data)

    n_samples = len(final_sft_data)
    n_eval = max(1, int(n_samples * 0.2))  
    n_train = n_samples - n_eval

    if n_train <= 0:
        print(
            "Error: Not enough data to create a training set after filtering. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    training_list = final_sft_data[:n_train]
    evaluation_list = final_sft_data[n_train:]  

    print(
        f"Training set size: {len(training_list)}, Evaluation set size: {len(evaluation_list)}"
    )

    training_dataset = Dataset.from_list(training_list)
    evaluation_dataset = Dataset.from_list(evaluation_list)

    print(f"Training dataset columns: {training_dataset.column_names}")
    print(f"Evaluation dataset columns: {evaluation_dataset.column_names}")

    return training_dataset, evaluation_dataset, solutions


def main():
    config = load_yaml_config("./config/config.yaml")
    training_dataset, evaluation_dataset, solutions = generate_dataset(config)
    train_model(training_dataset, evaluation_dataset, solutions, config)


if __name__ == "__main__":
    main()
