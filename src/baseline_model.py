import argparse
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import tqdm
import yaml
import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def generate_datestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_model(model_name=None):
    model_name = config.get("baseline_model")

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def prompt_model(model, tokenizer, prompt):
    """ """

    text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=4096)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


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

    testing_dir = "./data/arc/arc-agi_evaluation_challenges.json"
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


def grid_to_str(grid: list[list[int]]):
    """Converts a grid to a string representation."""
    grid_strs = []
    for row in grid:
        row_str = ", ".join([str(x) for x in row])
        grid_strs.append("[" + row_str + "]")
    return "[" + ", ".join(grid_strs) + "]"


def data_instance_to_chat_input(challenge_data_instance, systemPrompt):
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


def baseline():
    date_stamp = generate_datestamp()
    test_ids = list(testing_challenges.keys())

    # save a copy of the config.yaml with the datestamp
    config_copy = config.copy()

    os.makedirs(config_copy["arc_data_dir"], exist_ok=True)
    with open(f"./data/outputs/{date_stamp}_config.yaml", "w") as f:
        yaml.dump(config_copy, f)

    model, tokenizer = load_model()

    answers = {}
    answers_parsed = {}

    for task_id in tqdm.tqdm(test_ids):
        task_prompt_section = testing_challenges[task_id]
        task_prompt_section = data_instance_to_chat_input(
            task_prompt_section, systemPrompt=None
        )

        res = prompt_model(
            model=model,
            tokenizer=tokenizer,
            prompt=task_prompt_section,
        )

        query_list = find_last_list_of_lists(res)

        print(f"Query List: {query_list}")

        answers[task_id] = res

        # Save the answers to a file
        output_dir = config.get("arc_data_dir")
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(
            f"./data/outputs/{date_stamp}_baseline_answers.json"
        )
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=4)

        answers_parsed[task_id] = query_list
        output_file_path = f"./data/outputs/{date_stamp}_baseline_answers_parsed.json"
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(answers_parsed, f, indent=4)

    return answers, answers_parsed


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

    # Calculate the number of matching cells
    matching_cells = np.sum(possible == solution)

    # Calculate the total number of cells
    total_cells = possible.size

    # Calculate the normalized score
    score = matching_cells / total_cells

    return score


def calculate_cell_value_match_non_zero(possible: list[list], solution: list[list]):
    if is_jagged(possible):
        return 0.0

    # Count the amount of zeros in possible and solution
    possible = np.array(possible)
    solution = np.array(solution)

    if possible.shape != solution.shape:
        return 0.0

    # from the solution create a true false array

    solution_non_zero = np.where(solution != 0, 1.0, 0.0)

    possible_correct = np.where(possible == solution, 1.0, 0.0)

    all_correct_non_zeros = solution_non_zero * possible_correct

    return np.sum(all_correct_non_zeros) / np.sum(solution_non_zero)


def calculate_exact_match_score(possible: list[list], solution: list[list]) -> bool:
    """
    Calculate the exact match score between the model's output and the solution.
    """
    # Check if the lengths of the lists are equal
    if is_jagged(possible):
        return False

    if not calculate_grid_dimension_score(possible, solution):
        return False
    # Check if each sublist is identical

    return all(
        [
            sublist_possible == sublist_solution
            for sublist_possible, sublist_solution in zip(possible, solution)
        ]
    )


# Add this function
def generate_scorecard(results: dict, total_tasks: int):
    """
    Calculates and formats a scorecard from the comparison results.

    Args:
        results: A dictionary where keys are task IDs and values are
                 dictionaries containing scores ('grid_dim', 'cell_match', 'exact_match')
                 or None if the answer was invalid/unparseable.
        total_tasks: The total number of tasks attempted.

    Returns:
        A dictionary representing the scorecard.
    """
    parsed_tasks = 0
    unparseable_tasks = 0
    jagged_answers = 0
    grid_dim_matches = 0
    exact_matches = 0
    total_cell_match_score = 0.0
    valid_cell_match_tasks = 0  # Tasks that were parseable and non-jagged

    for task_id, scores in results.items():
        if scores is None:
            unparseable_tasks += 1
        else:
            parsed_tasks += 1
            if scores["is_jagged"]:
                jagged_answers += 1
            else:
                # Only count scores for non-jagged, parseable answers
                valid_cell_match_tasks += 1
                if scores["grid_dim"]:
                    grid_dim_matches += 1
                if scores["exact_match"]:
                    exact_matches += 1
                total_cell_match_score += scores["cell_match"]

    # Calculate averages and percentages, handling potential division by zero
    avg_cell_match_score = (
        (total_cell_match_score / valid_cell_match_tasks)
        if valid_cell_match_tasks > 0
        else 0.0
    )
    percent_parsed = (parsed_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0
    percent_grid_dim_match_overall = (
        (grid_dim_matches / total_tasks) * 100 if total_tasks > 0 else 0.0
    )
    percent_grid_dim_match_parsed = (
        (grid_dim_matches / parsed_tasks) * 100 if parsed_tasks > 0 else 0.0
    )
    percent_exact_match_overall = (
        (exact_matches / total_tasks) * 100 if total_tasks > 0 else 0.0
    )
    percent_exact_match_parsed = (
        (exact_matches / parsed_tasks) * 100 if parsed_tasks > 0 else 0.0
    )
    percent_jagged_parsed = (
        (jagged_answers / parsed_tasks) * 100 if parsed_tasks > 0 else 0.0
    )

    scorecard = {
        "Total Tasks Attempted": total_tasks,
        "Tasks Successfully Parsed": parsed_tasks,
        "Tasks Failed to Parse": unparseable_tasks,
        "Percentage Parsed": f"{percent_parsed:.2f}%",
        "--- Scores (based on successfully parsed tasks) ---": "",
        "Parsed Answers Found Jagged": f"{jagged_answers} ({percent_jagged_parsed:.2f}%)",
        "Grid Dimension Matches": f"{grid_dim_matches} ({percent_grid_dim_match_parsed:.2f}% of parsed)",
        "Exact Matches": f"{exact_matches} ({percent_exact_match_parsed:.2f}% of parsed)",
        "Average Cell Value Match (Normalised, non-jagged only)": f"{avg_cell_match_score:.4f}",
        "--- Scores (based on total tasks) ---": "",
        "Grid Dimension Match Rate (Overall)": f"{percent_grid_dim_match_overall:.2f}%",
        "Exact Match Rate (Overall)": f"{percent_exact_match_overall:.2f}%",
    }

    return scorecard


def compare_answers_to_solutions():
    try:
        with open("./data/arc/baseline_answers_parsed.json", "r") as f:
            answers = json.load(f)
    except FileNotFoundError:
        print("Error: ./data/arc/baseline_answers_parsed.json not found.")
        print("Run the baseline() function first to generate predictions.")
        return None  # Return None if predictions file doesn't exist
    except json.JSONDecodeError:
        print(
            "Error: Could not decode JSON from ./data/arc/baseline_answers_parsed.json."
        )
        return None

    try:
        with open("./data/arc/arc-agi_evaluation_solutions.json", "r") as f:
            solutions = json.load(f)
    except FileNotFoundError:
        print("Error: ./data/arc/arc-agi_evaluation_solutions.json not found.")
        return None
    except json.JSONDecodeError:
        print(
            "Error: Could not decode JSON from ./data/arc/arc-agi_evaluation_solutions.json."
        )
        return None

    answers_ids = list(answers.keys())
    task_results = {}

    print(f"Comparing {len(answers_ids)} answers to solutions...")

    for id_ in answers_ids:
        # Ensure the solution exists for this ID
        if id_ not in solutions:
            print(
                f"Warning: Solution for ID {id_} not found in evaluation solutions file. Skipping."
            )
            task_results[id_] = None  # Mark as uncomparable
            continue

        solution = solutions[id_][0]  # ARC solutions have test/output structure
        answer = answers[id_]

        # Initialize scores for this task
        scores = {
            "is_jagged": False,
            "grid_dim": False,
            "cell_match": 0.0,
            "exact_match": False,
        }

        # Check if the answer is a valid list of lists
        # Handle potential None values from parsing
        if (
            answer is None
            or not isinstance(answer, list)
            or not all(isinstance(i, list) for i in answer)
        ):
            print(f"Answer for ID {id_} is invalid or not a list of lists.")
            task_results[id_] = None  # Mark as unparseable/invalid
            continue  # Skip scoring for this task

        # Check if the solution is valid (should always be, but good practice)
        if not isinstance(solution, list) or not all(
            isinstance(i, list) for i in solution
        ):
            print(f"Warning: Solution format for ID {id_} is unexpected. Skipping.")
            task_results[id_] = None  # Mark as uncomparable
            continue

        # Check if the answer is jagged *before* calculating scores

        # Most scores will be 0 or False if jagged, as per score function logic
        # We still calculate them to be consistent, but note it was jagged.

        # Calculate scores using the existing functions
        # Note: calculation functions handle jagged/shape mismatch internally
        scores["grid_dim"] = calculate_grid_dimension_score(answer, solution)
        scores["cell_match"] = calculate_cell_value_match_normalised(answer, solution)
        scores["exact_match"] = calculate_exact_match_score(answer, solution)
        scores["matching_non_zero"] = calculate_cell_value_match_non_zero(
            answer, solution
        )

        task_results[id_] = scores  # Store the calculated scores

        # Optional: Print individual task scores during comparison
        print(
            f"ID: {id_} -> Grid Dim: {scores['grid_dim']}, Cell Match: {scores['cell_match']:.4f}, Exact Match: {scores['exact_match']}, Matching Non Zero: {scores['matching_non_zero']}, Jagged: {scores['is_jagged']}"
        )

    # check for None values in rows
    for id_, scores in task_results.items():
        if scores is None:
            task_results[id_] = {
                "is_jagged": True,
                "grid_dim": False,
                "cell_match": 0.0,
                "exact_match": False,
                "matching_non_zero": 0.0,
            }

    save_scorecard_pandas(task_results)

    return task_results


def save_scorecard_pandas(scorecard):
    """
    Saves the scorecard to a CSV file using pandas.
    """
    df = pd.DataFrame.from_dict(scorecard, orient="index")
    output_dir = "./data/outputs"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{generate_datestamp()}_scorecard.csv"
    output_file_path = os.path.join(output_dir, file_name)
    df.to_csv(output_file_path, index=True)
    print(f"Scorecard saved to {output_file_path}")


if __name__ == "__main__":
    (
        training_challenges,
        training_solutions,
        validation_challenges,
        calidation_solutions,
        testing_challenges,
        testing_solutions,
    ) = load_arc_challenges_soltuions()

    parser = argparse.ArgumentParser(description="Baseline Model for ARC")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="Name of the model to use.",
    )
    args = parser.parse_args()

    config["baseline_model"] = args.model_name

    training_ids = list(training_challenges.keys())
    validation_ids = list(validation_challenges.keys())
    test_ids = list(testing_challenges.keys())

    model, tokenizer = load_model(model_name=args.model_name)
    answers, answers_parsed = baseline()

    compare_answers_to_solutions()
