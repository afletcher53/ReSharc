import json
import os
import numpy as np
import pandas as pd

# ========== Scoring Metrics ========== #


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


# ========== Scoring Functions ========== #
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


def run_scoring(config):
    try:
        answers_file = os.path.join(
            config["arc_outputs_dir"],
            f"{config['run_datetimestamp']}_baseline_replies_parsed.json",
        )
        with open(answers_file, "r") as f:
            answers = json.load(f)
    except FileNotFoundError:
        print(f"Error: {answers_file} not found.")
        print("Run the baseline() function first to generate predictions.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {answers_file}.")
        return None

    try:
        solutions_file = os.path.join(
            config["arc_data_dir"], config["evaluation_solutions_file"]
        )
        with open(solutions_file, "r") as f:
            solutions = json.load(f)
    except FileNotFoundError:
        print("Error: s")
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

    save_scorecard_pandas(task_results, config)

    return task_results


def save_scorecard_pandas(scorecard, config):
    """
    Saves the scorecard to a CSV file using pandas.
    """
    df = pd.DataFrame.from_dict(scorecard, orient="index")
    output_dir = "./data/outputs"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{config['run_datetimestamp']}_scorecard.csv"
    output_file_path = os.path.join(output_dir, file_name)
    df.to_csv(output_file_path, index=True)
    print(f"Scorecard saved to {output_file_path}")
