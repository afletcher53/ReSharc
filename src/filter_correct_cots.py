"""Filter the correct COTS from the COTS list and create a dataset with 0/1 correctness."""

import json
import os

# --- ANSI Color Codes --- (Keep these as they are)
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
DIM = "\033[2m"
YELLOW = "\033[93m"
CYAN = "\033[96m"


def load_cots_list(cots_file_path):
    """Loads COTS from a JSONL file."""
    print(f"Attempting to load COTS from: {cots_file_path}")
    if not os.path.exists(cots_file_path):
        print(f"{RED}Error: COTS file not found at {cots_file_path}{RESET}")
        expected_dir = os.path.dirname(cots_file_path)
        print(f"Expected directory: {expected_dir}")
        print(f"Current working directory: {os.getcwd()}")
        return None  # Return None if file not found

    cots_list = []
    try:
        with open(cots_file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    cots_list.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(
                        f"{YELLOW}Warning: Skipping malformed JSON line {line_number} in {cots_file_path}: {line.strip()} - Error: {e}{RESET}"
                    )
        print(
            f"{GREEN}Successfully loaded {len(cots_list)} COTS from {os.path.basename(cots_file_path)}.{RESET}"
        )
    except IOError as e:
        print(f"{RED}Error reading file {cots_file_path}: {e}{RESET}")
        return None  # Return None on IO error
    except Exception as e:
        print(f"{RED}Unexpected error during COTS loading: {e}{RESET}")
        return None  # Return None on other unexpected errors
    return cots_list


def find_last_list_of_lists(text: str):
    # (Your function remains the same)
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
        last_checked_end = end_index  # Ensure progress in the loop


def load_json_solutions(filepath):
    # (Your function remains the same)
    solutions = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:  # Added encoding
            solutions = json.load(f)
        print(f"{GREEN}Loaded solutions from {filepath}{RESET}")
    except FileNotFoundError:
        print(f"{YELLOW}Warning: Solutions file not found at {filepath}{RESET}")
    except json.JSONDecodeError:
        print(f"{RED}Error: Could not decode JSON from {filepath}{RESET}")
    except Exception as e:
        print(f"{RED}Error loading {filepath}: {e}{RESET}")
    return solutions


# --- Load Ground Truth Solutions ---
ground_truth_solutions = {}
TRAINING_SOLUTIONS_FILE = "./data/arc/arc-agi_training_solutions.json"
EVALUATION_SOLUTIONS_FILE = "./data/arc/arc-agi_evaluation_solutions.json"

ground_truth_solutions.update(load_json_solutions(TRAINING_SOLUTIONS_FILE))
ground_truth_solutions.update(load_json_solutions(EVALUATION_SOLUTIONS_FILE))

# --- Load COTS Data ---
cots_input_file = "./data/filtered_sft/combined_output.jsonl"
cots = load_cots_list(cots_input_file)


def parse_and_evaluate_cots(cots_data):
    """
    Parses the COTS list to find model answers and marks each item with
    'correct' (1 for correct, 0 for incorrect).
    """
    if not cots_data:
        print(f"{YELLOW}Warning: COTS data is empty or None. Nothing to parse.{RESET}")
        return []  # Return empty list if input is None or empty

    processed_cots = []
    for i, cots_item_orig in enumerate(cots_data):
        cots_item = cots_item_orig.copy()  # Work on a copy to avoid modifying original dict during iteration if re-assigning cots_data

        # Initialize correctness and model_answer
        cots_item["model_answer"] = None
        cots_item["correct"] = 0  # Default to incorrect

        response_data = cots_item.get("raw_response")
        response_str = None

        if isinstance(response_data, list) and response_data:
            response_str = response_data[0]  # Take the first element if it's a list
            if not isinstance(response_str, str):
                print(
                    f"{YELLOW}Warning: Item {i}, task_id '{cots_item.get('task_id', 'N/A')}': raw_response list element is not a string. Response: {response_str}{RESET}"
                )
                response_str = None  # Cannot parse non-string
        elif isinstance(response_data, str):
            response_str = response_data
        else:
            print(
                f"{YELLOW}Warning: Item {i}, task_id '{cots_item.get('task_id', 'N/A')}': raw_response is not a list or string or is an empty list. Response: {response_data}{RESET}"
            )

        parsed_model_answer = None
        if response_str:
            parsed_model_answer = find_last_list_of_lists(response_str)
            cots_item["model_answer"] = parsed_model_answer
        # else: model_answer remains None, correct remains 0

        task_id = cots_item.get("task_id")
        if not task_id:
            print(
                f"{YELLOW}Warning: Item {i} is missing 'task_id'. Cannot determine correctness. Marking as incorrect.{RESET}"
            )
            # 'correct' is already 0
            processed_cots.append(cots_item)
            continue

        solution_list = ground_truth_solutions.get(task_id)
        if not solution_list:
            print(
                f"{YELLOW}Warning: No ground truth solution found for task_id '{task_id}'. Item {i} marked as incorrect.{RESET}"
            )
            # 'correct' is already 0
            processed_cots.append(cots_item)
            continue

        # Assuming the first solution in the list is the target
        ground_truth_answer = solution_list[0]

        if (
            parsed_model_answer is not None
            and parsed_model_answer == ground_truth_answer
        ):
            cots_item["correct"] = 1
        # else: 'correct' remains 0 (if no model_answer or it doesn't match)

        # Process token counts
        if "prompt_tokens" in cots_item and "completion_tokens" in cots_item:
            try:
                # Ensure they are numbers, convert if possible
                p_tokens = cots_item["prompt_tokens"]
                c_tokens = cots_item["completion_tokens"]

                # Handle cases where tokens might already be numbers or strings
                cots_item["prompt_tokens"] = (
                    int(p_tokens) if p_tokens is not None else 0
                )
                cots_item["completion_tokens"] = (
                    int(c_tokens) if c_tokens is not None else 0
                )

                cots_item["total_tokens"] = (
                    cots_item["prompt_tokens"] + cots_item["completion_tokens"]
                )
            except (ValueError, TypeError) as e:
                print(
                    f"{YELLOW}Warning: Could not convert token counts to integers for task_id '{task_id}'. Error: {e}. Setting tokens to 0.{RESET}"
                )
                cots_item["prompt_tokens"] = 0
                cots_item["completion_tokens"] = 0
                cots_item["total_tokens"] = 0

        processed_cots.append(cots_item)
    return processed_cots


# --- Main Processing ---
if cots is not None:
    print(
        f"\n{CYAN}Starting parsing and evaluation of {len(cots)} COTS items...{RESET}"
    )
    # This function will now add 'correct': 0 or 1 to each item.
    # It returns a new list with the processed items.
    processed_cots_dataset = parse_and_evaluate_cots(cots)

    # Save the entire dataset with the 0/1 'correct' field
    output_full_dataset_path = "./data/filtered_sft/arc_cots_full_eval.jsonl"
    try:
        with open(output_full_dataset_path, "w", encoding="utf-8") as f:
            for cots_item in processed_cots_dataset:
                f.write(json.dumps(cots_item) + "\n")
        print(
            f"\n{GREEN}Successfully saved the full dataset with 0/1 correctness to: {output_full_dataset_path}{RESET}"
        )
    except IOError as e:
        print(f"{RED}Error saving the full dataset: {e}{RESET}")
    except Exception as e:
        print(f"{RED}Unexpected error while saving full dataset: {e}{RESET}")

    # --- Retrieve only the COTS items marked as correct (1) ---
    # This is similar to your original 'correct_cots' but uses the 0/1 value
    actually_correct_cots = [
        item for item in processed_cots_dataset if item.get("correct") == 1
    ]

    # --- Calculate average total_tokens for the 'actually_correct_cots' ---
    total_tokens_sum = 0
    correct_items_with_tokens_count = 0

    for cots_item in actually_correct_cots:
        if "total_tokens" in cots_item and isinstance(
            cots_item["total_tokens"], (int, float)
        ):
            total_tokens_sum += cots_item["total_tokens"]
            correct_items_with_tokens_count += 1

    if correct_items_with_tokens_count > 0:
        average_total_tokens = total_tokens_sum / correct_items_with_tokens_count
        print(
            f"{CYAN}Average total tokens for COTS marked as correct (1): {average_total_tokens:.2f}{RESET}"
        )
    else:
        print(
            f"{YELLOW}No COTS marked as correct (1) had valid 'total_tokens' to average.{RESET}"
        )

    # --- Save only the COTS marked as correct (1) to a separate jsonl file ---
    # This is equivalent to your original 'arc_correct_cots.jsonl'
    output_correct_only_path = "./data/filtered_sft/arc_cots_correct_only.jsonl"
    try:
        with open(output_correct_only_path, "w", encoding="utf-8") as f:
            for cots_item in actually_correct_cots:
                f.write(json.dumps(cots_item) + "\n")
        print(
            f"{GREEN}Successfully saved COTS marked as correct (1) to: {output_correct_only_path}{RESET}"
        )
    except IOError as e:
        print(f"{RED}Error saving the correct-only COTS dataset: {e}{RESET}")
    except Exception as e:
        print(
            f"{RED}Unexpected error while saving correct-only COTS dataset: {e}{RESET}"
        )

    print(
        f"\n{CYAN}Total number of COTS items processed: {len(processed_cots_dataset)}{RESET}"
    )
    print(
        f"{CYAN}Total number of COTS items marked as correct (1): {len(actually_correct_cots)}{RESET}"
    )

else:
    print(f"{RED}COTS data could not be loaded. Aborting further processing.{RESET}")
