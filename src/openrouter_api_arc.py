# scripts/phase1_generate.py

import os
import sys
import json
import yaml
import argparse

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Project Root (added to sys.path): {project_root}")
print(f"Current Working Directory: {os.getcwd()}")
print(f"Python sys.path: {sys.path}")
# --- End Path Setup ---

from dotenv import load_dotenv

# Import all necessary functions from llm_utils
from src.llm_utils import (
    setup_openrouter_client,
    call_openrouter_llm,
)

# Import arc_utils
try:
    from src.arc_utils import (
        load_arc_tasks,
        create_task_prompt_section,
    )
except ImportError:
    print("Error: Could not import functions from src.arc_utils.")
    print("Please ensure src/arc_utils.py exists and contains the required functions.")
    sys.exit(1)


# --- Main Execution Logic ---
if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate LLM responses for ARC tasks, optionally starting from a specific task ID."
    )
    parser.add_argument(
        "--start_task_id",
        type=str,
        default=None,
        help="The Task ID to start processing from. If not provided, starts from the beginning.",
    )
    args = parser.parse_args()
    start_task_id_arg = args.start_task_id

    # 1. Load Configuration
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config/config.yaml not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)

    # 2. Determine API Provider and Setup Client/Configuration
    # (Keep this section as it is)
    api_provider = config.get("api_provider", "google").lower()
    print(f"Selected API Provider: {api_provider}")
    llm_client = None
    models_to_use = []
    api_settings = {}

    if api_provider == "openrouter":
        api_settings = config.get("openrouter_settings", {})
        if not api_settings:
            print("Error: 'openrouter_settings' not found in config.yaml")
            sys.exit(1)
        llm_client = setup_openrouter_client(api_settings)
        if llm_client is None:
            print("Exiting due to OpenRouter client setup failure.")
            sys.exit(1)
        models_to_use = api_settings.get("teacher_models", [])
    else:
        print(
            f"Error: Invalid api_provider '{api_provider}'. Use 'google' or 'openrouter'."
        )
        sys.exit(1)

    if not isinstance(models_to_use, list) or not models_to_use:
        print(
            f"Error: 'teacher_models' not found or empty in '{api_provider}_settings'."
        )
        sys.exit(1)
    print(f"Target '{api_provider}' teacher models: {models_to_use}")

    # 3. Load ARC Tasks
    arc_data_dir = config.get("arc_data_dir", "data/arc")
    challenges_file = config.get(
        "training_challenges_file", "arc-agi_training_challenges.json"
    )

    evaluation_challenges_file = config.get(
        "evaluation_challenges_file", "arc-agi_evaluation_challenges.json"
    )

    evaluation_challenges_path = os.path.join(arc_data_dir, evaluation_challenges_file)

    # arc_challenges_path = os.path.join(arc_data_dir, challenges_file)

    arc_tasks = load_arc_tasks(evaluation_challenges_path)
    if arc_tasks is None:
        sys.exit(1)

    model_name = models_to_use[0]
    # 4. Prepare Output File
    output_file = config.get(
        "raw_generations_output_file",
        f"data/generated_sft/evaluation_{model_name}_generations.jsonl",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # --- 5. Process Tasks --- <<< MODIFIED: Section adjusted for start_task_id >>>
    all_task_ids = list(arc_tasks.keys())  # Get all task IDs first

    # Load up raw generations file and get the existing task IDs
    existing_task_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    result_record = json.loads(line)
                    existing_task_ids.add(result_record["task_id"])
                except json.JSONDecodeError:
                    print(f"Error decoding JSON line: {line}")
                    continue
    print(f"Found {len(existing_task_ids)} existing task IDs in {output_file}")

    # Filter out existing task IDs from all_task_ids
    all_task_ids = [
        task_id for task_id in all_task_ids if task_id not in existing_task_ids
    ]
    print(
        f"Filtered {len(existing_task_ids)} existing task IDs, {len(all_task_ids)} total tasks left to process"
    )

    # Determine the actual list of task IDs to process based on start_task_id_arg
    tasks_to_consider = all_task_ids
    if start_task_id_arg:
        try:
            start_index = all_task_ids.index(start_task_id_arg)
            tasks_to_consider = all_task_ids[start_index:]
            print(
                f"Attempting to start processing from Task ID: {start_task_id_arg} (index {start_index})"
            )
        except ValueError:
            print(
                f"Warning: Start Task ID '{start_task_id_arg}' not found in the task list. Processing all tasks from the beginning."
            )

    else:
        print("No start_task_id provided. Starting processing from the beginning.")

    max_tasks = config.get("max_tasks_to_process")
    if max_tasks is not None and max_tasks > 0:
        print(
            f"Processing a maximum of {max_tasks} tasks from the determined starting point."
        )
        task_ids_to_process = tasks_to_consider[:max_tasks]
    else:
        print(
            f"Processing all {len(tasks_to_consider)} tasks from the determined starting point."
        )
        task_ids_to_process = tasks_to_consider
    results_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    try:
        with open(output_file, "a", encoding="utf-8") as f_out:
            for task_id in task_ids_to_process:
                task_data = arc_tasks[task_id]

                if not task_data.get("test") or not task_data["test"]:
                    print(f"Skipping Task ID {task_id}: No test cases found.")
                    continue

                test_case_index = 0
                task_prompt_section = create_task_prompt_section(task_data)
                full_prompt = config.get("BASE_PROMPT_TEMPLATE").format(
                    task_prompt_section=task_prompt_section
                )

                for model_name in models_to_use:
                    print(
                        f"--- Processing Task ID: {task_id} (Test Case {test_case_index}) with Model: {model_name} ({api_provider}) ---"
                    )

                    raw_response = None
                    prompt_tokens = 0
                    completion_tokens = 0

                    # Call the appropriate LLM function
                    if api_provider == "openrouter":
                        raw_response, prompt_tokens, completion_tokens = (
                            call_openrouter_llm(
                                llm_client, model_name, full_prompt, api_settings
                            )
                        )

                    # Accumulate token counts
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens

                    # Save result if successful
                    if raw_response is not None:
                        result_record = {
                            "task_id": task_id,
                            "test_case_index": test_case_index,
                            "api_provider": api_provider,
                            "teacher_model": model_name,
                            # "prompt": full_prompt, # Optional
                            "raw_response": raw_response,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        }
                        try:
                            f_out.write(json.dumps(result_record) + "\n")
                            f_out.flush()
                            results_count += 1
                        except Exception as write_e:
                            print(
                                f"Error writing record to JSONL for task {task_id}, model {model_name}: {write_e}"
                            )
                    else:
                        print(
                            f"Failed to get response for Task {task_id} from {model_name} ({api_provider}) after retries."
                        )

                    # Optional: Add delay between API calls
                    # time.sleep(1)

    except IOError as e:
        print(f"Error opening or writing to output file {output_file}: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback

        print(f"An unexpected error occurred during task processing: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n--- Generation Complete ---")
    print(f"API Provider Used: {api_provider}")
    print(f"Attempted processing for {len(task_ids_to_process)} task IDs.")
    if start_task_id_arg:
        print(
            f"(Processing started from task ID: {start_task_id_arg if start_task_id_arg in all_task_ids else 'beginning (start ID not found)'})"
        )
    print(f"Successfully saved {results_count} raw generations to {output_file}")
    print(
        f"Estimated Token Usage: Prompts≈{total_prompt_tokens}, Completions≈{total_completion_tokens}"
    )
