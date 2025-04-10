import os
import json
from collections import defaultdict

# --- Configuration ---
GENERATED_SFT_DIR = "./data/generated_sft"  # Path to your generated files
EVALUATION_SOLUTIONS_FILE = "./data/arc/arc-agi_evaluation_solutions.json"
TRAINING_SOLUTIONS_FILE = "./data/arc/arc-agi_training_solutions.json"

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


ground_truth_solutions.update(load_json_solutions(EVALUATION_SOLUTIONS_FILE))
ground_truth_solutions.update(load_json_solutions(TRAINING_SOLUTIONS_FILE))

if not ground_truth_solutions:
    print(
        f"{RED}Error: No ground truth solutions loaded. Cannot perform comparisons. Exiting.{RESET}"
    )
    exit()


# --- Helper Functions (format_cell, print_comparison, find_last_list_of_lists) ---
# (These functions remain the same as in the previous version)
def format_cell(value, is_different, is_placeholder=False, width=3):
    val_str = str(value)
    if is_placeholder:
        return f"{DIM}{'-':^{width}}{RESET}"
    elif is_different:
        return f"{RED}{val_str:^{width}}{RESET}"
    else:
        return f"{val_str:^{width}}"


def print_comparison(candidate, solution, task_id):
    if candidate is None and solution is None:
        print(f"\n--- Comparison for Task ID: {task_id} ---")
        print(f"{RED}Both candidate and solution are missing/invalid.{RESET}")
        return False
    if candidate is None:
        print(f"\n--- Comparison for Task ID: {task_id} ---")
        print(f"{RED}Candidate grid is missing or invalid.{RESET}")
        if solution:
            print("Solution exists.")
        return False
    if solution is None:
        print(f"\n--- Comparison for Task ID: {task_id} ---")
        print(f"{RED}Solution grid is missing for task {task_id}.{RESET}")
        if candidate:
            print("Candidate exists.")
        return False

    match = candidate == solution
    print(f"\n--- Comparison for Task ID: {task_id} ---")
    if match:
        print(f"{GREEN}Candidate perfectly matches solution.{RESET}")
        return True
    else:
        print(f"{RED}Candidate does NOT match solution.{RESET}")

    cand_rows = len(candidate)
    sol_rows = len(solution)
    max_rows = max(cand_rows, sol_rows)
    cand_cols = max(len(r) for r in candidate if r) if any(candidate) else 0
    sol_cols = max(len(r) for r in solution if r) if any(solution) else 0
    max_cols = max(cand_cols, sol_cols)
    cell_width = 3
    cand_header_width = max(len("Candidate"), max_cols * (cell_width + 1) - 1)
    sol_header_width = max(len("Solution"), max_cols * (cell_width + 1) - 1)
    print(f"\n{'Candidate':^{cand_header_width}} | {'Solution':^{sol_header_width}}")
    print("-" * cand_header_width + "-+-" + "-" * sol_header_width)
    for r in range(max_rows):
        cand_row_str = []
        sol_row_str = []
        for c in range(max_cols):
            is_placeholder_cand = (
                r >= cand_rows or c >= len(candidate[r]) if r < cand_rows else True
            )
            cand_val = candidate[r][c] if not is_placeholder_cand else "-"
            is_placeholder_sol = (
                r >= sol_rows or c >= len(solution[r]) if r < sol_rows else True
            )
            sol_val = solution[r][c] if not is_placeholder_sol else "-"
            is_different = (cand_val != sol_val) or (
                is_placeholder_cand != is_placeholder_sol
            )
            cand_row_str.append(
                format_cell(cand_val, is_different, is_placeholder_cand, cell_width)
            )
            sol_row_str.append(
                format_cell(sol_val, is_different, is_placeholder_sol, cell_width)
            )
        print(
            f"{' '.join(cand_row_str):<{cand_header_width}} | {' '.join(sol_row_str):<{sol_header_width}}"
        )
    print("-" * cand_header_width + "-+-" + "-" * sol_header_width)
    print(f"Legend: {RED}Difference{RESET}, {DIM}Missing/Padding (-){RESET}")
    print("-" * (cand_header_width + 3 + sol_header_width))
    return False


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


# --- Main Script Logic ---

print(f"\nStarting validation in directory: {GENERATED_SFT_DIR}")

if not os.path.isdir(GENERATED_SFT_DIR):
    print(f"{RED}Error: Directory not found: {GENERATED_SFT_DIR}{RESET}")
    exit()

# --- Initialize Per-File Statistics Storage ---
per_file_stats = {}
overall_file_processing_errors = 0  # Count files that couldn't be opened/read at all

# Process each file
processed_files_count = 0
for filename in os.listdir(GENERATED_SFT_DIR):
    if not filename.endswith(".jsonl"):
        continue  # Process only .jsonl files
    processed_files_count += 1

    jsonl_file_path = os.path.join(GENERATED_SFT_DIR, filename)
    all_extracted_data = []  # Store extracted data for this file
    file_has_line_errors = False  # Track errors within lines of this file

    # --- Initialize Stats for Current File ---
    current_file_stats = {
        "total_tasks": 0,
        "comparisons_attempted": 0,
        "matches": 0,
        "mismatches": 0,
        "candidate_errors": 0,  # Unparseable answers
        "solution_errors": 0,
        "unknown_ids": 0,
        "line_json_errors": 0,  # Count lines that failed JSON parsing
        "line_processing_errors": 0,  # Count other errors processing a line
    }

    print(f"\n{CYAN}{'=' * 15} Processing file: {filename} {'=' * 15}{RESET}\n")
    try:
        with open(jsonl_file_path, "r", encoding="utf-8") as infile:
            for i, line in enumerate(infile):
                line_number = i + 1
                task_id = f"UNKNOWN_{filename}_{line_number}"  # Default
                extracted_list = None  # Initialize

                # --- Increment Task Count for this file ---
                current_file_stats["total_tasks"] += 1

                try:
                    data = json.loads(line.strip())
                    task_id = data.get("task_id", task_id)  # Overwrite if found

                    # Handle potential list in raw_response
                    raw_response_value = data.get("raw_response", "")
                    text_to_search = ""
                    if isinstance(raw_response_value, str):
                        text_to_search = raw_response_value
                    elif isinstance(raw_response_value, list):
                        if len(raw_response_value) > 0 and isinstance(
                            raw_response_value[0], str
                        ):
                            text_to_search = raw_response_value[0]
                        else:
                            print(
                                f"{YELLOW}Warning: 'raw_response' is list but not usable. Task {task_id} line {line_number}.{RESET}"
                            )
                            file_has_line_errors = True
                            current_file_stats["line_processing_errors"] += 1
                    else:
                        print(
                            f"{YELLOW}Warning: 'raw_response' not string or list. Task {task_id} line {line_number}. Type: {type(raw_response_value)}.{RESET}"
                        )
                        file_has_line_errors = True
                        current_file_stats["line_processing_errors"] += 1

                    # Attempt extraction if we have text
                    if text_to_search:
                        extracted_list = find_last_list_of_lists(text_to_search)
                    # else: extracted_list remains None

                    # --- Store Result ---
                    result = {
                        "line": line_number,
                        "task_id": task_id,
                        "extracted_list": extracted_list,
                    }
                    all_extracted_data.append(result)

                except json.JSONDecodeError:
                    print(
                        f"{YELLOW}Warning: Invalid JSON on line {line_number} in {filename}. Skipping line.{RESET}"
                    )
                    file_has_line_errors = True
                    current_file_stats["line_json_errors"] += 1
                    # Append a placeholder to keep counts consistent if needed, or just skip
                    all_extracted_data.append(
                        {
                            "line": line_number,
                            "task_id": task_id,
                            "extracted_list": None,
                            "error": "json_decode",
                        }
                    )
                except Exception as e:
                    print(
                        f"{RED}Error processing line {line_number} in {filename} (Task: {task_id}): {e}{RESET}"
                    )
                    file_has_line_errors = True
                    current_file_stats["line_processing_errors"] += 1
                    all_extracted_data.append(
                        {
                            "line": line_number,
                            "task_id": task_id,
                            "extracted_list": None,
                            "error": str(e),
                        }
                    )

    except Exception as e:
        print(f"{RED}Error opening or reading file {jsonl_file_path}: {e}{RESET}")
        overall_file_processing_errors += 1
        per_file_stats[filename] = {
            "error": f"Failed to open/read: {e}"
        }  # Mark file as failed
        continue  # Skip to next file

    print(f"\n--- File {filename}: Comparison Phase ---")

    # --- Perform Comparisons and Update File Stats ---
    processed_task_ids_in_file = set()
    for task_data in all_extracted_data:
        # Skip if this line had a fundamental error recorded during parsing
        if task_data.get("error"):
            continue

        task_id = task_data.get("task_id")

        if task_id in processed_task_ids_in_file:
            continue  # Avoid duplicate task processing
        processed_task_ids_in_file.add(task_id)

        # Check Task ID Validity
        if task_id.startswith("UNKNOWN"):
            print(f"\n--- Comparison for Task ID: {task_id} ---")
            print(f"{YELLOW}Skipping comparison: Task ID is unknown.{RESET}")
            current_file_stats["unknown_ids"] += 1
            continue

        # Check Candidate Validity (Parseability)
        candidate = task_data.get("extracted_list")
        if candidate is None:
            # No comparison possible, count as candidate error
            current_file_stats["candidate_errors"] += 1
            # Optionally print context (already handled by print_comparison if called)
            # print_comparison(candidate, None, task_id) # Call to show context if needed
            print(f"\n--- Comparison for Task ID: {task_id} ---")
            print(
                f"{YELLOW}Skipping comparison: Candidate grid could not be extracted or is invalid.{RESET}"
            )
            if task_id in ground_truth_solutions:
                print(f"{DIM}(Ground truth solution exists){RESET}")
            continue  # Cannot compare

        # Check Solution Validity
        solution_entry = ground_truth_solutions.get(task_id)
        solution = None
        if (
            solution_entry
            and isinstance(solution_entry, list)
            and len(solution_entry) > 0
        ):
            solution = solution_entry[0]
        else:
            # Have valid candidate, but no valid solution
            current_file_stats["solution_errors"] += 1
            print(f"\n--- Comparison for Task ID: {task_id} ---")
            print(
                f"{YELLOW}Skipping comparison: Ground truth solution not found or invalid.{RESET}"
            )
            # print_comparison(candidate, solution, task_id) # Call to show context
            continue  # Cannot compare

        # --- Perform Comparison (Both Candidate and Solution are Valid) ---
        current_file_stats["comparisons_attempted"] += 1
        is_match = print_comparison(candidate, solution, task_id)

        if is_match:
            current_file_stats["matches"] += 1
        else:
            current_file_stats["mismatches"] += 1

    # --- Store Final Stats for this File ---
    per_file_stats[filename] = current_file_stats
    print(
        f"\n{CYAN}{'=' * 15} Finished processing file: {filename} {'=' * 15}{RESET}\n"
    )


# --- Final Summary Reports ---

# --- Per-Model Summary ---
print("\n" + "=" * 70)
print(f"{CYAN}{' ' * 25}PER-MODEL SUMMARY{RESET}")
print("=" * 70)

# Sort files alphabetically for consistent reporting
sorted_filenames = sorted(per_file_stats.keys())

for filename in sorted_filenames:
    stats = per_file_stats[filename]
    print(f"{CYAN}--- Model File: {filename} ---{RESET}")

    if "error" in stats:
        print(f"  {RED}Processing Error: {stats['error']}{RESET}")
        continue

    total_tasks = stats["total_tasks"]
    comparisons = stats["comparisons_attempted"]
    matches = stats["matches"]
    candidate_errors = stats["candidate_errors"]  # Unparseable
    solution_errors = stats["solution_errors"]
    unknown_ids = stats["unknown_ids"]
    line_errors = stats["line_json_errors"] + stats["line_processing_errors"]

    # Calculate Parseable count (tasks where candidate was successfully extracted)
    # Note: This count includes tasks where the solution might have been missing.
    parseable_count = total_tasks - candidate_errors - unknown_ids - line_errors

    # Calculate Percentages (handle division by zero)
    pct_parseable = (parseable_count / total_tasks) * 100 if total_tasks > 0 else 0
    pct_unparseable = (candidate_errors / total_tasks) * 100 if total_tasks > 0 else 0
    pct_correct_of_comparable = (matches / comparisons) * 100 if comparisons > 0 else 0
    # Optional: Correctness relative to parseable tasks (might be misleading if many solution errors)
    # pct_correct_of_parseable = (matches / parseable_count) * 100 if parseable_count > 0 else 0
    pct_solution_errors = (
        (solution_errors / total_tasks) * 100 if total_tasks > 0 else 0
    )
    pct_unknown_ids = (unknown_ids / total_tasks) * 100 if total_tasks > 0 else 0
    pct_line_errors = (line_errors / total_tasks) * 100 if total_tasks > 0 else 0

    print(f"  Total Task Entries: {total_tasks}")
    print(f"  - Parseable Answers: {parseable_count} ({pct_parseable:.2f}%)")
    print(
        f"  - Unparseable Answers (Candidate Errors): {candidate_errors} ({pct_unparseable:.2f}%)"
    )
    print(f"  - Line Processing/JSON Errors: {line_errors} ({pct_line_errors:.2f}%)")
    print(f"  - Unknown Task IDs: {unknown_ids} ({pct_unknown_ids:.2f}%)")
    print(
        f"  - Ground Truth Solution Missing/Invalid: {solution_errors} ({pct_solution_errors:.2f}%)"
    )  # (where candidate was parseable)
    print("-" * 40)
    print(
        f"  Comparisons Attempted (Parseable Candidate & Valid Solution): {comparisons}"
    )
    if comparisons > 0:
        print(
            f"    - Correct Matches: {matches} ({pct_correct_of_comparable:.2f}% of comparable)"
        )
        print(
            f"    - Mismatches: {stats['mismatches']} ({(stats['mismatches'] / comparisons) * 100:.2f}% of comparable)"
        )
    else:
        print("    - Correct Matches: 0")
        print("    - Mismatches: 0")
    print("-" * 70)


# --- Overall Summary ---
print("\n" + "=" * 70)
print(f"{CYAN}{' ' * 25}OVERALL SUMMARY{RESET}")
print("=" * 70)

# Calculate overall totals by summing per-file stats
overall_stats = defaultdict(int)
valid_files_processed = 0
for filename in sorted_filenames:
    stats = per_file_stats[filename]
    if "error" not in stats:
        valid_files_processed += 1
        for key, value in stats.items():
            overall_stats[key] += value

print(f"Total Model Files Found: {processed_files_count}")
print(f"Files Successfully Opened & Read: {valid_files_processed}")
print(f"Files with Opening/Reading Errors: {overall_file_processing_errors}")
print("-" * 70)

total_tasks = overall_stats["total_tasks"]
comparisons = overall_stats["comparisons_attempted"]
matches = overall_stats["matches"]
mismatches = overall_stats["mismatches"]
candidate_errors = overall_stats["candidate_errors"]
solution_errors = overall_stats["solution_errors"]
unknown_ids = overall_stats["unknown_ids"]
line_errors = (
    overall_stats["line_json_errors"] + overall_stats["line_processing_errors"]
)

print(f"Overall Task Entries Processed: {total_tasks}")
if total_tasks > 0:
    overall_parseable_count = total_tasks - candidate_errors - unknown_ids - line_errors
    overall_pct_parseable = (overall_parseable_count / total_tasks) * 100
    overall_pct_unparseable = (candidate_errors / total_tasks) * 100
    overall_pct_line_errors = (line_errors / total_tasks) * 100
    overall_pct_unknown_ids = (unknown_ids / total_tasks) * 100
    overall_pct_solution_errors = (
        solution_errors / total_tasks
    ) * 100  # Relative to total tasks

    print(
        f"  - Parseable Answers: {overall_parseable_count} ({overall_pct_parseable:.2f}%)"
    )
    print(
        f"  - Unparseable Answers (Candidate Errors): {candidate_errors} ({overall_pct_unparseable:.2f}%)"
    )
    print(
        f"  - Line Processing/JSON Errors: {line_errors} ({overall_pct_line_errors:.2f}%)"
    )
    print(f"  - Unknown Task IDs: {unknown_ids} ({overall_pct_unknown_ids:.2f}%)")
    print(
        f"  - Ground Truth Solution Missing/Invalid: {solution_errors} ({overall_pct_solution_errors:.2f}%)"
    )
else:
    print("  (No tasks processed)")
print("-" * 70)
print(
    f"Overall Comparisons Attempted (Parseable Candidate & Valid Solution): {comparisons}"
)
if comparisons > 0:
    overall_match_percentage = (matches / comparisons) * 100
    overall_mismatch_percentage = (mismatches / comparisons) * 100
    print(
        f"  - Perfect Matches: {matches} ({overall_match_percentage:.2f}% of comparable)"
    )
    print(
        f"  - Mismatches: {mismatches} ({overall_mismatch_percentage:.2f}% of comparable)"
    )
else:
    print("  - Perfect Matches: 0")
    print("  - Mismatches: 0")
print("=" * 70)
