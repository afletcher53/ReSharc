import argparse
import json
import os
import re
import sys
from datetime import datetime

import wandb

from utils import run_scoring
import tqdm
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ["TRANSFORMERS_CACHE"] = "/mnt/parscratch/users/aaron/huggingface"

run = wandb.init(
    project="arc-baseline-models",
)

try:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config/config.yaml not found.")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing config file: {e}")
    sys.exit(1)


def generate_datestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_model(config):
    model_name = config.get("baseline_model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    return model, tokenizer


def prompt_model(model, tokenizer, prompts, config):
    """Prompts the model with the given input and returns the output."""

    texts = tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        add_generation_prompt=True,
    )

    output_dir = config.get("arc_outputs_dir")
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(
        f"./data/outputs/{config['run_datetimestamp']}_baseline_replies_chat_string.json"
    )
    with open(output_file_path, "a", encoding="utf-8") as f:
        json.dump(texts, f, indent=4)

    model_inputs = tokenizer(
        texts, return_tensors="pt", padding=True, padding_side="left"
    ).to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=config["baseline_models"]["max_tokens"],
    )

    generated_ids = generated_ids[:, len(model_inputs.input_ids[0]) :]

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses


def load_arc_test_set(config):
    """Load the ARC challenges and their solutions. Note that we are using the evaluation dataset as the test set."""
    testing_file_challenges = os.path.join(
        config.get("arc_data_dir"), config.get("evaluation_challenges_file")
    )

    testing_file_solution = os.path.join(
        config.get("arc_data_dir"), config.get("evaluation_solutions_file")
    )
    with open(testing_file_challenges, "r", encoding="utf-8") as f:
        testing_challenges = json.load(f)

    with open(testing_file_solution, "r", encoding="utf-8") as f:
        testing_solutions = json.load(f)
    testing_challenges = {k: testing_challenges[k] for k in testing_challenges.keys()}
    testing_solutions = {k: testing_solutions[k] for k in testing_challenges.keys()}
    return (testing_challenges, testing_solutions)


def grid_to_str(grid: list[list[int]]):
    """Converts a grid to a string representation."""
    grid_strs = []
    for row in grid:
        row_str = ", ".join([str(x) for x in row])
        grid_strs.append("[" + row_str + "]")
    return "[" + ", ".join(grid_strs) + "]"


def data_instance_to_chat_input(challenge_data_instance, config):
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

    full_prompt = config["CONCISE_BASE_TEMPLATE"].format(
        task_prompt_section=instance_string
    )

    messages = [
        {
            "role": "user",
            "content": full_prompt,
        },
    ]

    return messages


def find_last_list_of_lists(text: str):
    """
    Finds the last substring in 'text' that represents a JSON list of lists.
    """
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
            except TypeError:
                pass

        last_checked_end = end_index


def parse_answer_string(text: str):
    list_of_lists = re.findall(r"\[(\s)?\[.+\](\s)?\]", text)
    return f"[[{list_of_lists[-1]}]]"


def save_config(config):
    """
    Saves the config to a YAML file.
    """

    output_dir = config.get("arc_data_dir")
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(
        f"./data/outputs/{config['run_datetimestamp']}_config.yaml"
    )
    with open(output_file_path, "w") as f:
        yaml.dump(config, f)


def generate_task_prompt_sections(testing_challenges, config):
    """
    Generates task prompt sections from the testing challenges.
    """
    task_prompt_sections = {}
    for task_id, task_prompt_section in testing_challenges.items():
        task_prompt_sections[task_id] = data_instance_to_chat_input(
            task_prompt_section, config
        )

    output_dir = config.get("arc_data_dir")
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(
        config["arc_outputs_dir"], f"{config['run_datetimestamp']}_test_prompts.json"
    )
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(task_prompt_sections, f, indent=4)

    return task_prompt_sections


def baseline(testing_challenges, config):
    test_ids = list(testing_challenges.keys())
    batch_size = config["baseline_models"]["batch_size"]
    save_config(config)

    model, tokenizer = load_model(config)

    answers = {}
    answers_parsed = {}

    task_prompt_sections = generate_task_prompt_sections(testing_challenges, config)

    for i in range(0, len(test_ids), batch_size):
        if i + batch_size > len(test_ids):
            batch_size = len(test_ids) - i
        batch_ids = test_ids[i : i + batch_size]
        batch_task_prompt_sections = [
            task_prompt_sections[task_id] for task_id in batch_ids
        ]

        responses = prompt_model(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_task_prompt_sections,
            config=config,
        )

        query_list = [find_last_list_of_lists(res) for res in responses]

        for short_i, (task_id, res) in enumerate(zip(batch_ids, responses)):
            answers[task_id] = res
            answers_parsed[task_id] = query_list[short_i]

        output_file_path = os.path.join(
            config["arc_outputs_dir"],
            f"{config['run_datetimestamp']}_baseline_replies.json",
        )
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=4)

        output_file_path = os.path.join(
            config["arc_outputs_dir"],
            f"{config['run_datetimestamp']}_baseline_replies_parsed.json",
        )
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(answers_parsed, f, indent=4)

    return answers, answers_parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Model for ARC")

    parser.add_argument(
        "--model_name",
        type=str,
        default=config["baseline_models"]["default_model"],
        help="Name of the model to use.",
    )
    args = parser.parse_args()
    config["run_datetimestamp"] = generate_datestamp()
    config["baseline_model"] = args.model_name

    testing_challenges, testing_solutions = load_arc_test_set(config)

    print(
        "Loaded ARC challenges and solutions - Evaluation Dataset. Running Inference on base models."
    )
    print(f"Model: {config['baseline_model']}")

    print(f"Testing Challenges: {len(testing_challenges)}")

    model, tokenizer = load_model(config)
    answers, answers_parsed = baseline(testing_challenges, config)
    print("Inference completed.")
    run_scoring(config)
