import json
import os
import sys


def grid_to_str(grid: list[list[int]]):
    """
    Converts a grid (list of lists of ints) to a string representation
    with rows separated by newlines and elements joined with no separator.
    Example: [[0, 7, 7], [7, 7, 7], [0, 7, 7]] -> "077\\n777\\n077" (11 chars)
    """

    if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
        print(
            f"Warning: Invalid grid format received in grid_to_str: {grid}",
            file=sys.stderr,
        )
        return "[Invalid Grid Data]"
    try:
        return "\n".join(["".join(map(str, row)) for row in grid])
    except Exception as e:
        print(f"Error in grid_to_str: {e}\nGrid: {grid}", file=sys.stderr)
        return "[Error Formatting Grid]"


def format_grid_for_prompt(grid):
    """Formats a grid (list of lists) into a string for the LLM prompt."""
    if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
        print(f"Warning: Invalid grid format received: {grid}")
        return "[Invalid Grid Data]"
    try:
        return "\n".join(["".join(map(str, row)) for row in grid])
    except Exception as e:
        print(f"Error formatting grid: {e}\nGrid: {grid}")
        return "[Error Formatting Grid]"


def create_task_prompt_section(task_data):
    """Formats the examples and test input for a single ARC task."""
    prompt_section = ""

    if task_data.get("train"):
        for i, pair in enumerate(task_data["train"]):
            if "input" in pair and "output" in pair:
                prompt_section += f"E.g. {i + 1} Input:\n{pair['input']}\n"
                prompt_section += f"Output:\n{pair['output']}\n\n"
            else:
                prompt_section += f"E.g. {i + 1}: [Malformed train pair data]\n\n"

    if task_data.get("test") and task_data["test"]:
        if "input" in task_data["test"][0]:
            test_input_grid = task_data["test"][0]["input"]
            prompt_section += f"Test Input:\n{test_input_grid}\n"
        else:
            prompt_section += (
                "Test Input: [Test case exists but missing 'input' grid]\n"
            )
    else:
        prompt_section += "Test Input: [No test input data provided for this task]\n"

    return prompt_section.strip()


def load_arc_tasks(challenges_file_path):
    """Loads ARC challenge tasks from a JSON file."""
    print(f"Attempting to load ARC challenges from: {challenges_file_path}")
    if not os.path.exists(challenges_file_path):
        print(f"Error: Challenges file not found at {challenges_file_path}")

        expected_dir = os.path.dirname(challenges_file_path)
        print(f"Expected directory: {expected_dir}")
        print(f"Current working directory: {os.getcwd()}")
        return None

    try:
        with open(challenges_file_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        print(
            f"Successfully loaded {len(tasks)} task IDs from {os.path.basename(challenges_file_path)}."
        )
        return tasks
    except json.JSONDecodeError as e:
        print(
            f"Error: Could not decode JSON from {challenges_file_path}. Invalid JSON format. {e}"
        )
        return None
    except IOError as e:
        print(f"Error reading file {challenges_file_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading tasks: {e}")
        return None
