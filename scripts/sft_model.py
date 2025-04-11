import json
import sys

import yaml
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset

from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import wandb


try:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config/config.yaml not found.")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing config file: {e}")
    sys.exit(1)

BASE_PROMPT_TEMPLATE = config["BASE_PROMPT_TEMPLATE"]

run = wandb.init(
    project="no_hope_no_vram",
)


def grid_to_str(grid: list[list[int]]):
    """Converts a grid to a string representation."""
    grid_strs = []
    for row in grid:
        row_str = " ".join([str(x) for x in row])
        grid_strs.append(row_str)
    return "|" + "|".join(grid_strs) + "|"


def data_instance_to_chat_input(challenge_data_instance, tokenizer):
    """
    Formats a data instance into a chat input for the model.
    No longer prompts the LLM to guess rows and cols,
    just shows input-output examples and ends with the test input -> ?
    """
    instance_string_list = []
    for i, io_pair in enumerate(challenge_data_instance["train"]):
        io_pair_as_string = f"Example {i + 1} input: {grid_to_str(io_pair['input'])}\nExample {i + 1} output: {grid_to_str(io_pair['output'])}"
        instance_string_list.append(io_pair_as_string)

    instance_string = "\n".join(instance_string_list)

    test_input_as_string = grid_to_str(challenge_data_instance["test"][0]["input"])
    instance_string += f"\nTest input: {test_input_as_string}"

    full_prompt = BASE_PROMPT_TEMPLATE.format(task_prompt_section=instance_string)

    messages = [
        {
            "role": "user",
            "content": full_prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return text


def create_task_prompt_section(task_data):
    """Formats the examples and test input for a single ARC task."""
    prompt_section = ""
    if task_data.get("train"):
        for i, pair in enumerate(task_data["train"]):
            if "input" in pair and "output" in pair:
                prompt_section += (
                    f"Example {i + 1} Input:\n{grid_to_str(pair['input'])}\n"
                )
                prompt_section += (
                    f"Example {i + 1} Output:\n{grid_to_str(pair['output'])}\n\n"
                )
            else:
                prompt_section += f"Example {i + 1}: [Malformed train pair data]\n\n"
    if task_data.get("test") and task_data["test"]:
        if "input" in task_data["test"][0]:
            test_input_grid = task_data["test"][0]["input"]
            prompt_section += f"Test Input:\n{grid_to_str(test_input_grid)}\n"
        else:
            prompt_section += (
                "Test Input: [Test case exists but missing 'input' grid]\n"
            )
    else:
        prompt_section += "Test Input: [No test input data provided for this task]\n"

    return prompt_section.strip()


def load_cot_responses(tokenizer, cofig):
    """Load the ARC challenges and their solutions."""
    cot_data_file = "./data/filtered_sft/combined.jsonl"
    with open(cot_data_file, "r", encoding="utf-8") as f:
        cot_data = [json.loads(line) for line in f.readlines()]

    cot_data_lengths = sorted([len(d["raw_response"]) for d in cot_data])
    max_len_threshold = max(
        cot_data_lengths[: int(len(cot_data_lengths) * config["max_threshold"])]
    )

    included_task_ids = set([d["task_id"] for d in cot_data])

    from collections import defaultdict

    def find_last_list_of_lists(text: str):
        last_checked_end = len(text)
        while True:
            end_index = text.rfind("]", 0, last_checked_end)
            if end_index == -1:
                return None
            balance = 0
            start_index = -1
            for i in range(end_index, -1, -1):
                char = text[i]
                if char == "]":
                    balance += 1
                elif char == "[":
                    balance -= 1
                    if balance == 0:
                        start_index = i
                        break
            if start_index != -1:
                potential_json_str = text[start_index : end_index + 1]
                try:
                    parsed_data = json.loads(potential_json_str)
                    if (
                        isinstance(parsed_data, list)
                        and parsed_data
                        and all(isinstance(item, list) for item in parsed_data)
                    ):
                        return parsed_data
                except json.JSONDecodeError:
                    pass
            last_checked_end = end_index

    all_data = defaultdict(list)
    for task_id in included_task_ids:
        for d in cot_data:
            if d["task_id"] == task_id:
                if isinstance(d["raw_response"], list):
                    if len(d["raw_response"][0]) < max_len_threshold:
                        grid = find_last_list_of_lists(d["raw_response"][0])
                        if grid is not None:
                            all_data[task_id].append(grid_to_str(grid))

                else:
                    if len(d["raw_response"]) < max_len_threshold:
                        grid = find_last_list_of_lists(d["raw_response"])
                        if grid is not None:
                            all_data[task_id].append(grid_to_str(grid))

    train_ids, val_ids = train_test_split(
        list(all_data.keys()), test_size=0.2, random_state=42
    )

    challenges_file_path = config.get("training_challenges_file")
    challenges_dir = "./data/arc/arc-agi_training_challenges.json"
    if not challenges_file_path:
        print("Error: 'arc_challenges_file' not found in config.")
        sys.exit(1)

    with open(challenges_dir, "r", encoding="utf-8") as f:
        challenges = json.load(f)

    _mappings = []
    for set_of_ids in (train_ids, val_ids):
        ids_to_input_prompts_mapping = {}
        for single_id in set_of_ids:
            task_prompt_section = create_task_prompt_section(challenges[single_id])
            full_prompt = config["BASE_PROMPT_TEMPLATE"].format(
                task_prompt_section=task_prompt_section
            )

            ids_to_input_prompts_mapping[single_id] = full_prompt
        _mappings.append(ids_to_input_prompts_mapping)

    train_ids_to_input_prompts, val_ids_to_input_prompts = _mappings

    training_dataset = []
    for i in train_ids:
        for num_in_var, j in enumerate(all_data[i]):
            training_dataset.append(
                (f"{i}_{num_in_var}", train_ids_to_input_prompts[i], j)
            )

    validation_dataset = []
    for i in val_ids:
        for num_in_var, j in enumerate(all_data[i]):
            validation_dataset.append(
                (f"{i}_{num_in_var}", val_ids_to_input_prompts[i], j)
            )

    train_ds = Dataset.from_dict(
        {
            "cot_task_id": [x[0] for x in training_dataset],
            "input": [x[1] for x in training_dataset],
            "output": [x[2] for x in training_dataset],
            "chat_str_input": [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": x[1]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for x in training_dataset
            ],
        }
    )
    val_ds = Dataset.from_dict(
        {
            "cot_task_id": [x[0] for x in validation_dataset],
            "input": [x[1] for x in validation_dataset],
            "output": [x[2] for x in validation_dataset],
            "chat_str_input": [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": x[1]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for x in validation_dataset
            ],
        }
    )

    def tokenize_and_mask(example):
        full_text = example["chat_str_input"] + " " + example["output"]
        tokenized = tokenizer(
            full_text,
        )

        seq_len = len(tokenized["input_ids"])
        # Log lengths, especially long ones
        if seq_len > 4096:  # Adjust this threshold based on expectations
            print(f"INFO: Found long tokenized sequence length: {seq_len}")

        labels = tokenized["input_ids"].copy()

        prompt_len = len(tokenizer(example["chat_str_input"])["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len

        tokenized["labels"] = labels

        return tokenized

    tokenized_map_train_ds = train_ds.map(tokenize_and_mask)
    tokenized_map_val_ds = val_ds.map(tokenize_and_mask)
    return tokenized_map_train_ds, tokenized_map_val_ds


import torch
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("baseline_models")["default_model"]
    )

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    (train_ds, val_ds) = load_cot_responses(tokenizer, config)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Recommended type
        bnb_4bit_compute_dtype=torch.bfloat16,  # Or torch.float16 if BF16 not supported
        bnb_4bit_use_double_quant=True,  # Optional further memory saving
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.get("baseline_models")["default_model"],
        attn_implementation="flash_attention_2",
        device_map="auto",
        quantization_config=quantization_config,
    )

    # Prepare the model for k-bit training *before* applying LoRA
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "v_proj",
            "down_proj",
            "gate_proj",
            "up_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    peft_model.config.gradient_checkpointing = True

    print(peft_model.get_memory_footprint() / 1e6)

    batch_size = 1

    args = TrainingArguments(
        gradient_checkpointing=True,
        remove_unused_columns=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=32,
        per_device_eval_batch_size=batch_size,
        bf16=True,
        num_train_epochs=5,
        logging_steps=5,
        load_best_model_at_end=True,
        label_names=["labels"],
        output_dir="./data/outputs",
        optim="adafactor",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        peft_model,
        args,
        train_dataset=train_ds.remove_columns(
            ["input", "cot_task_id", "output", "chat_str_input"]
        ),
        eval_dataset=val_ds.remove_columns(
            ["input", "cot_task_id", "output", "chat_str_input"]
        ),
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
