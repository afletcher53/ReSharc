from ast import Tuple
from typing import Any, List, Optional
import pandas as pd
import json

from src.sft_model import (
    calculate_cell_value_match_non_zero,
    calculate_cell_value_match_normalised,
    calculate_exact_match_score,
    calculate_grid_dimension_score,
)

evaluation_files = [
    "data/generated_sft/cot_gemin_2-5_flash_thinking_raw_generations.jsonl",
    "data/generated_sft/cot_evaluation_gpt-3.5-turbo-0125_generations.jsonl",
    "data/generated_sft/cot_evaluation_o4-mini-2025-04-16_generations.jsonl",
    "data/generated_sft/cot_reflection_evaluation_o4-mini-2025-04-16_generations.jsonl",
    "data/generated_sft/cot_repeat_reflection_evaluation_o4-mini-2025-04-16_generations.jsonl",
    "data/generated_sft/raw_generations.jsonl",
]


def find_last_list_of_lists_with_indices(
    text: str,
):
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
                return None, -1, -1

        last_checked_end = end_bracket_index


with open("data/arc/arc-agi_evaluation_solutions.json", "r") as f:
    solutions = json.load(f)


def get_solution_from_task_id(task_id: str):
    return solutions[task_id]


for file in evaluation_files:
    df = pd.read_json(file, lines=True)

    grid_dimension_score = 0
    cell_value_match_score = 0
    exact_match_score = 0
    cell_match_non_zero_score = 0
    prompt_tokens = 0
    completion_tokens = 0
    for index, row in df.iterrows():
        task_id = row["task_id"]
        raw_response = row["raw_response"]
        solution = get_solution_from_task_id(task_id)

        if solution is None:
            continue
        else:
            solution = solution[0]

        try:
            grid_response, _, _ = find_last_list_of_lists_with_indices(raw_response)
        except Exception as e:
            continue

        if grid_response is None:
            continue

        cell_value_match_score += calculate_cell_value_match_normalised(
            grid_response, solution
        )
        exact_match_score += calculate_exact_match_score(grid_response, solution)
        cell_match_non_zero_score += calculate_cell_value_match_non_zero(
            grid_response, solution
        )
        grid_dimension_score += calculate_grid_dimension_score(grid_response, solution)
        prompt_tokens += row["prompt_tokens"]
        completion_tokens += row["completion_tokens"]

    print("--------------------------------")
    print(f"File: {file}")
    print(f"Grid Dimension Score: {grid_dimension_score / 400}")
    print(f"Cell Value Match Score: {cell_value_match_score / 400}")
    print(f"Exact Match Score: {exact_match_score / 400}")
    print(f"Cell Match Non Zero Score: {cell_match_non_zero_score / 400}")
    print(f"Prompt Tokens: {prompt_tokens}")
    print(f"Completion Tokens: {completion_tokens}")
