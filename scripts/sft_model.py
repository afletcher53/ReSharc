import json
import sys
from pathlib import Path


import yaml
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments

try:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config/config.yaml not found.")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing config file: {e}")
    sys.exit(1)

BASE_PROMPT_TEMPLATE = """You are an expert in solving Abstraction and Reasoning Corpus (ARC) problems. Analyze the provided input/output examples and determine the transformation rule. Apply this rule to the final test input grid.

**Task Description:**
The user will provide several pairs of example input grids and their corresponding output grids. They represent a hidden transformation rule. Finally, a single test input grid is provided. Your goal is to deduce the rule from the examples and apply it to the test input grid to produce the correct test output grid.

**Output Format:**
Provide your step-by-step reasoning within `<thinking>` tags. Explain how you identified the pattern and how you are applying it to the test input.
Provide the final predicted output grid for the test input within `<answer>` tags. The grid should be formatted as a JSON list of lists, with integers representing colors. Example: [[1, 0], [0, 1]]

Ensure that you check the consistency of your answer.

---
**Current Task:**

{task_prompt_section}
---
Now, please solve the current task using the specified format. Remember to output the reasoning in <thinking> tags and the final grid as a JSON list of lists in <answer> tags.
"""


def grid_to_str(grid: list[list[int]]):
    """Converts a grid to a string representation."""
    grid_strs = []
    for row in grid:
        row_str = ", ".join([str(x) for x in row])
        grid_strs.append("[" + row_str + "]")
    return "[" + ", ".join(grid_strs) + "]"


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


def load_arc_challenges_soltuions():
    """Load the ARC challenges and their solutions."""
    challenges_file_path = config.get("training_challenges_file")
    challenges_dir = "./data/arc/arc-agi_training_challenges.json"
    if not challenges_file_path:
        print("Error: 'arc_challenges_file' not found in config.")
        sys.exit(1)

    with open(challenges_dir, "r", encoding="utf-8") as f:
        challenges = json.load(f)

    ids = challenges.keys()

    # Split the IDs into training and validation sets
    train_ids, val_ids = train_test_split(list(ids), test_size=0.2, random_state=42)

    print(f"Training IDs: {len(train_ids)}")
    print(f"Validation IDs: {len(val_ids)}")
    print(f"Total IDs: {len(ids)}")

    training_challenges = {k: challenges[k] for k in train_ids}
    validation_challenges = {k: challenges[k] for k in val_ids}

    solutions_dir = "./data/arc/arc-agi_training_solutions.json"
    with open(solutions_dir, "r", encoding="utf-8") as f:
        solutions = json.load(f)

    training_solutions = {k: solutions[k] for k in train_ids}
    validation_solutions = {k: solutions[k] for k in val_ids}

    testing_dir = "./data/arc/arc-agi_test_challenges.json"
    with open(testing_dir, "r", encoding="utf-8") as f:
        testing_challenges = json.load(f)
    testing_challenges = {k: testing_challenges[k] for k in testing_challenges.keys()}

    testing_solution_dir = "./data/arc/arc-agi_evaluation_solutions.json"
    with open(testing_solution_dir, "r", encoding="utf-8") as f:
        testing_solutions = json.load(f)
    testing_solutions = {k: testing_solutions[k] for k in testing_challenges.keys()}
    return (
        training_challenges,
        training_solutions,
        validation_challenges,
        validation_solutions,
        testing_challenges,
        testing_solutions,
    )


from datasets import Dataset


def merge_jsons(tgt_dr="./data/generated_sft/"):
    # find all json files in the directory

    merged_outputs = []
    final_list = {}

    for file in tgt_dr.rglob("*.jsonl"):
        # load the jsonl file as a list
        with open(file, "r") as f:
            _data = [json.loads(line) for line in f.readlines()]
            merged_outputs.extend(_data)

    return merge_jsons


def main():
    # load datasets
    (
        training_challenges,
        training_solutions,
        validation_challenges,
        validation_solutions,
        testing_challenges,
        testing_solutions,
    ) = load_arc_challenges_soltuions()

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"Training challenges: {len(training_challenges)}")
    print(f"Training solutions: {len(training_solutions)}")
    print(f"Validation challenges: {len(validation_challenges)}")
    print(f"Validation solutions: {len(validation_solutions)}")
    print(f"Testing challenges: {len(testing_challenges)}")
    print(f"Testing solutions: {len(testing_solutions)}")

    # Convert to datasets

    # TODO: verify this works tomorrow
    training_challenges = merge_jsons(tgt_dr="./data/generated_sft/")

    dataset_dict = {"inputs": [], "labels": []}

    training_keys = list(training_challenges.keys())
    print(f"Training keys: {len(training_keys)}")
    tokenizer = AutoTokenizer.from_pretrained(config.get("baseline_model"))

    for i, key in enumerate(training_keys):
        train = data_instance_to_chat_input(training_challenges[key], tokenizer)
        # GENERATE multiple labels corresponding to the same input, because each sample is passed into 10 different models as labels
        label = training_solutions[key][0]

        dataset_dict["inputs"].append(train)
        dataset_dict["labels"].append(label)

    ds = Dataset.from_dict(dataset_dict)

    # convert to string format

    print(ds)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(config.get("baseline_model"))

    from peft import LoraConfig, get_peft_model

    print(model)

    config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "up_proj",
            "o_proj",
            "k_proj",
            "down_proj",
            "gate_proj",
            "v_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(model)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # account = "stevhliu"
    # peft_model_id = f"{account}/google/vit-base-patch16-224-in21k-lora"
    batch_size = 128

    # args = TrainingArguments(
    #     # peft_model_id,
    #     remove_unused_columns=False,
    #     eval_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=5e-3,
    #     per_device_train_batch_size=batch_size,
    #     gradient_accumulation_steps=4,
    #     per_device_eval_batch_size=batch_size,
    #     fp16=True,
    #     num_train_epochs=1,
    #     logging_steps=1,
    #     load_best_model_at_end=True,
    #     label_names=["labels"],
    # )

    # trainer = Trainer(
    #     model,
    #     args,
    #     train_dataset=train_ds,
    #     eval_dataset=val_ds,
    #     tokenizer=tokenizer,
    #     data_collator=collate_fn,
    # )
    # trainer.train()


main()
