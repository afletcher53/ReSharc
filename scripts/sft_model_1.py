import json
import wandb
import yaml
import sys
from typing import List, Dict, Tuple, Optional, Any
import random
from datasets import Dataset

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer

# --- Constants ---
CONFIG_PATH = "config/config.yaml"
COT_DATA_FILE = "./data/filtered_sft/arc_correct_cots.jsonl"
CHALLENGES_FILE = (
    "./data/arc/arc-agi_training_challenges.json"  # Corrected path assumption
)


# --- Main Training Function ---
def train_model(train_dataset, eval_dataset, config):
    print(
        f"Starting fine-tuning process for base model: {config['baseline_models']['default_model']}"
    )

    if torch.cuda.is_available():
        print(f"CUDA available. Using {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Training on CPU (will be very slow).")
        device_map = None  # Use CPU

    # --- 2. Load Tokenizer ---
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
    eval_steps = 10

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
    model.config.use_cache = False  # Disable cache for training efficiency
    model.config.pretraining_tp = 1  # May be needed for some models

    if pad_token_added:
        print("Resizing model embeddings to account for new pad token")
    model.resize_token_embeddings(len(tokenizer))

    print("Base model loaded.")

    peft_config = None

    if use_peft:  # Check the global configuration flag FIRST
        print("Attempting to configure PEFT (LoRA).")
        # Try importing PEFT components *inside* the check

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

    training_arguments = TrainingArguments(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optimizer_type if use_4bit else "adamw_torch",
        save_steps=save_steps,
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
        eval_steps=eval_steps,
        eval_strategy="steps",
        report_to="wandb",
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
            "eval_steps": eval_steps,
        },
    )

    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config if use_peft else None,
        args=training_arguments,
    )

    # --- 7. Start Training ---
    print("*** Starting Training ***")
    train_result = trainer.train()

    # --- 8. Save Final Model & Metrics ---

    # Log metrics
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
            return yaml.safe_load(f) or {}  # Return empty dict if file is empty
    except FileNotFoundError:
        print(f"Warning: Config file not found at {path}. Using default empty config.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error loading or parsing config file {path}: {e}", file=sys.stderr)
        # Depending on severity, you might want to exit or return a default
        return {}  # Returning empty dict for now
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
        return []  # Return empty list on critical error
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
        return {}  # Return empty dict on critical error
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {path}: {e}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"An unexpected error occurred loading JSON {path}: {e}", file=sys.stderr)
        return {}


def grid_to_str(grid: List[List[int]]) -> str:
    """Converts a grid (list of lists) to a compact string representation."""
    if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
        # Handle potential malformed grid data gracefully
        print(f"Warning: Invalid grid format encountered: {grid}")
        return "|INVALID_GRID|"
    row_strs = [" ".join(map(str, row)) for row in grid]
    return "|" + "|".join(row_strs) + "|"


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
        # Find the last potential closing bracket before the last checked position
        end_bracket_index = text.rfind("]", 0, last_checked_end)
        if end_bracket_index == -1:
            return None  # No more closing brackets found

        balance = 0
        start_bracket_index = -1

        # Scan backward from the potential end bracket to find the matching start bracket
        for i in range(end_bracket_index, -1, -1):
            char = text[i]
            if char == "]":
                balance += 1  # Increase balance for closing bracket
            elif char == "[":
                balance -= 1  # Decrease balance for opening bracket
                if balance == 0:
                    # Found the matching opening bracket for the initial closing bracket
                    start_bracket_index = i
                    break  # Stop scanning backward for this pair

        # If a potential pair was found (start_bracket_index != -1)
        if start_bracket_index != -1:
            potential_json_str = text[start_bracket_index : end_bracket_index + 1]
            try:
                # Attempt to parse the substring as JSON
                parsed_data = json.loads(potential_json_str)

                # Validate if it's a non-empty list where all items are lists
                if (
                    isinstance(parsed_data, list)
                    and parsed_data  # Check if list is not empty
                    and all(isinstance(item, list) for item in parsed_data)
                ):
                    # Found a valid list of lists
                    return parsed_data, start_bracket_index, end_bracket_index + 1
            except json.JSONDecodeError:
                # If JSON parsing fails, it's not the list we're looking for, continue search
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
        new_data = data.copy()  # Work on a copy
        raw_response = new_data.get("raw_response")

        if isinstance(raw_response, str):
            find_result = find_last_list_of_lists_with_indices(raw_response)

            if find_result:
                parsed_grid, start_index, end_index = find_result
                new_grid_string = grid_to_str(parsed_grid)

                # Replace the JSON list with the compact string version
                new_response = (
                    raw_response[:start_index]
                    + new_grid_string
                    + raw_response[end_index:]
                )

                new_data["raw_response"] = new_response  # Update the copied dict
                modified_count += 1
                char_count_previous += len(raw_response)
                char_count_after += len(new_response)
            # Else: no list-of-lists found or raw_response wasn't string, keep original
        output_cot_data.append(new_data)  # Add original or modified data to output list

    return output_cot_data, modified_count, char_count_previous, char_count_after


def create_task_prompt_section(task_data: Dict[str, Any]) -> str:
    """Formats the examples and test input for a single ARC task."""
    task_id = task_data.get("task_id", "UNKNOWN")  # Get task_id for better warnings
    prompt_section = ""

    # Format Training Examples
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

    # Format Test Input
    test_cases = task_data.get("test", [])
    if test_cases and isinstance(test_cases, list):
        # Assuming we only use the first test case's input for the prompt
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


def main():
    config = load_yaml_config(CONFIG_PATH)  # Config might be used later
    cot_data = load_jsonl(COT_DATA_FILE)
    challenges_raw = load_json(CHALLENGES_FILE)

    if not cot_data or not challenges_raw:
        print("Error: Could not load critical data files. Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- Preprocessing ---

    print("Reformatting raw responses...")
    reformatted_cot_data, modified_count, chars_before, chars_after = (
        reformat_raw_response(cot_data)
    )
    print(f"{'-' * 20} Summary of Grid Shortening {'-' * 20}")
    print(f"Modified {modified_count} entries.")
    if modified_count > 0:
        # Calculate average reduction only on modified entries for a clearer picture
        avg_reduction = (chars_before - chars_after) / modified_count
        print(f"Average reduction per modified entry: {avg_reduction:.2f} characters")
    else:
        print("No entries required modification.")
    print("-" * (40 + len(" Summary of Grid Shortening ")))

    task_id_to_data = {task_id: data for task_id, data in challenges_raw.items()}
    for task_id, data in task_id_to_data.items():
        data["task_id"] = task_id

    # --- Final Assembly ---
    print("\nAssembling final data strings...")
    final_sft_data = []
    total_final_chars = 0
    processed_ids = set()

    tokenizer = AutoTokenizer.from_pretrained(
        config.get("baseline_models")["default_model"]
    )

    if tokenizer.chat_template is None:
        print(
            f"Warning: Tokenizer for {config.get('baseline_models')['default_model']} does not have a default chat template. Applying a basic default."
        )

        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    token_counts = []
    for data in reformatted_cot_data:
        task_id = data.get("task_id")
        if not task_id:
            print(f"Warning: Skipping entry due to missing 'task_id'. Data: {data}")
            continue

        if task_id not in task_id_to_data:
            print(
                f"Warning: Skipping Task ID {task_id} as it's not found in challenges data."
            )
            continue

        # Get the corresponding challenge data
        challenge_data = task_id_to_data[task_id]

        # Create the specific prompt section for this task
        task_prompt_section = create_task_prompt_section(challenge_data)

        # Format the base prompt
        prompt = config["CONCISE_BASE_TEMPLATE"].format(
            task_prompt_section=task_prompt_section
        )

        reasoning = data.get("raw_response")
        # Handle cases where raw_response might still be a list (e.g., from older data)
        if isinstance(reasoning, list):
            if reasoning:  # Check if list is not empty
                reasoning = str(reasoning[0])  # Take the first element as string
            else:
                reasoning = "[EMPTY REASONING]"  # Handle empty list case
        elif not isinstance(reasoning, str):
            reasoning = str(reasoning)  # Convert other types to string if necessary

        # Extract and format the ground truth answer grid
        model_answer_grid = data.get("model_answer")
        if model_answer_grid is None:
            print(
                f"Warning: Missing 'model_answer' for Task ID {task_id}. Cannot create answer string."
            )
            answer_str = "[MISSING ANSWER]"
        else:
            answer_str = grid_to_str(model_answer_grid)  # Use the same compact format

        user_prompt_content = prompt

        model_response_content = f"<thinking>\n{reasoning}\n</thinking>\n<answer>\n{answer_str}\n</answer>"  # Add the tags here

        messages = [
            {"role": "user", "content": user_prompt_content},
            {"role": "assistant", "content": model_response_content},
        ]

        # Ensure template application works
        try:
            formatted_string = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            print(f"Error applying chat template for Task ID {task_id}: {e}")
            print(f"Messages structure: {messages}")
            # Optionally skip this entry or use a fallback format
            formatted_string = f"USER: {user_prompt_content}\nASSISTANT: {model_response_content}"  # Basic fallback

        token_ids = tokenizer.encode(formatted_string)
        num_tokens = len(token_ids)

        if num_tokens > 7000:
            print(
                f"Warning: Token count for Task ID {task_id} exceeds 7000 tokens ({num_tokens} tokens). Removing"
            )
            continue
        token_counts.append(num_tokens)
        final_entry = {
            "task_id": task_id,
            "text": formatted_string,  # Preferred format for SFT using Hugging Face trainers often
        }
        final_sft_data.append(final_entry)
        total_final_chars += len(formatted_string)  # Use formatted string length
        processed_ids.add(task_id)

    print(
        f"\nProcessed {len(final_sft_data)} entries for {len(processed_ids)} unique Task IDs."
    )

    if token_counts:  # Avoid division by zero if no entries were processed
        print(
            f"Averaged tokenized token count per entry: {sum(token_counts) / len(token_counts):.2f}"
        )
    else:
        print("No entries processed, cannot calculate average token count.")

    # --- Dataset Creation ---
    if not final_sft_data:
        print("Error: No data available to create datasets. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Shuffle the data before splitting for better randomness
    random.shuffle(final_sft_data)

    # Split the final_sft_data into training and evaluation sets (80-20 split)
    split_index = int(len(final_sft_data) * 0.8)
    training_list = final_sft_data[:split_index]
    evaluation_list = final_sft_data[split_index:]

    print(
        f"Training set size: {len(training_list)}, Evaluation set size: {len(evaluation_list)}"
    )

    # Create Hugging Face Datasets
    # Using from_list as it directly takes the list of dictionaries
    training_dataset = Dataset.from_list(training_list)
    evaluation_dataset = Dataset.from_list(evaluation_list)

    print("\nCreated Hugging Face Datasets:")
    print("Training Dataset:")
    print(training_dataset)
    print("\nEvaluation Dataset:")
    print(evaluation_dataset)

    train_model(training_dataset, evaluation_dataset, config)


if __name__ == "__main__":
    main()
