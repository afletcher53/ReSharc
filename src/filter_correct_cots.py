"""Filter the correct COTS from the COTS list."""

import json
import os


def load_cots_list(cots_file_path):
    """Loads COTS from a JSONL file."""
    print(f"Attempting to load COTS from: {cots_file_path}")
    if not os.path.exists(cots_file_path):
        print(f"Error: COTS file not found at {cots_file_path}")
        # Provide more context on where the script expects the file
        expected_dir = os.path.dirname(cots_file_path)
        print(f"Expected directory: {expected_dir}")
        print(f"Current working directory: {os.getcwd()}")
        return None

    cots_list = []
    try:
        with open(cots_file_path, "r", encoding="utf-8") as f:
            for line in f:
                cots_list.append(json.loads(line))
        print(
            f"Successfully loaded {len(cots_list)} COTS from {os.path.basename(cots_file_path)}."
        )
    except json.JSONDecodeError as e:
        print(
            f"Error: Could not decode JSON from {cots_file_path}. Invalid JSON format. {e}"
        )
    except IOError as e:
        print(f"Error reading file {cots_file_path}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return cots_list


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


# --- ANSI Color Codes ---
RED = "\033[91m"  # For differences/errors
GREEN = "\033[92m"  # For perfect matches
RESET = "\033[0m"  # Reset color
DIM = "\033[2m"  # Dim text (for padding/missing cells)
YELLOW = "\033[93m"  # For warnings
CYAN = "\033[96m"  # For info/headers

# --- Load Ground Truth Solutions ---
ground_truth_solutions = {}


def load_json_solutions(filepath):
    # (Function remains the same)
    solutions = {}
    try:
        with open(filepath, "r") as f:
            solutions = json.load(f)
        print(f"Loaded solutions from {filepath}")
    except FileNotFoundError:
        print(f"{YELLOW}Warning: Solutions file not found at {filepath}{RESET}")
    except json.JSONDecodeError:
        print(f"{RED}Error: Could not decode JSON from {filepath}{RESET}")
    except Exception as e:
        print(f"{RED}Error loading {filepath}: {e}{RESET}")
    return solutions


cots = load_cots_list("./data/filtered_sft/combined.jsonl")
TRAINING_SOLUTIONS_FILE = "./data/arc/arc-agi_training_solutions.json"
EVALUATION_SOLUTIONS_FILE = "./data/arc/arc-agi_evaluation_solutions.json"

ground_truth_solutions.update(load_json_solutions(TRAINING_SOLUTIONS_FILE))
ground_truth_solutions.update(load_json_solutions(EVALUATION_SOLUTIONS_FILE))


def parse_for_answer(cots):
    """
    Parses the COTS list to find the correct answers.
    """

    for i, cots_item in enumerate(cots):
        response = cots_item["raw_response"]
        if isinstance(response, list):
            response = response[0]
        answer = find_last_list_of_lists(response)
        cots_item["model_answer"] = answer
        task_id = cots_item["task_id"]
        solution_entry = ground_truth_solutions.get(task_id)[0]
        if solution_entry == answer:
            cots_item["correct"] = True

        if "prompt_tokens" in cots_item and "completion_tokens" in cots_item:
            cots_item["prompt_tokens"] = int(cots_item["prompt_tokens"])
            cots_item["completion_tokens"] = int(cots_item["completion_tokens"])
            cots_item["total_tokens"] = (
                cots_item["prompt_tokens"] + cots_item["completion_tokens"]
            )


parse_for_answer(cots)


# retrieve only the correct cots
correct_cots = [cots_item for cots_item in cots if cots_item.get("correct")]

# average all total_tokens and display the average
total_tokens = 0
counter = 0
for cots_item in correct_cots:
    if "total_tokens" in cots_item:
        cots_item["total_tokens"] = int(cots_item["total_tokens"])
        counter += 1
        total_tokens += cots_item["total_tokens"]
average_total_tokens = total_tokens / counter

# save the correct cots to a jsonl file
with open("./data/filtered_sft/arc_correct_cots.jsonl", "w", encoding="utf-8") as f:
    for cots_item in correct_cots:
        f.write(json.dumps(cots_item) + "\n")

print(f"{CYAN}Average total tokens for correct COTS: {average_total_tokens}{RESET}")

print(f"{CYAN}Total number of correct COTS: {len(correct_cots)}{RESET}")
