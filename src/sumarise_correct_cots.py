import os
import sys
import json
import yaml
import argparse
import re
import time
import traceback


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Project Root (added to sys.path): {project_root}")
print(f"Current Working Directory: {os.getcwd()}")


from dotenv import load_dotenv


from src.llm_utils import (
    setup_openrouter_client,
    call_openrouter_llm,
)


try:
    from src.arc_utils import (
        format_grid_for_prompt,
        load_arc_tasks,
        create_task_prompt_section,
    )
except ImportError:
    print("Error: Could not import functions from src.arc_utils.")
    print("Please ensure src/arc_utils.py exists and contains the required functions.")
    sys.exit(1)


SUMMARIZATION_TEMPLATE = """Given the ARC task examples below and the step-by-step reasoning provided in the `<thinking>` block, provide a concise summary of the algorithm or transformation rule used to solve the task.
Focus on explaining *how* the input is transformed into the output based on the reasoning. Output only the summary description, without any preamble or explanation about the summary itself.

**ARC Task Examples:**
{task_prompt_section}

**Reasoning from Teacher Model:**
<thinking>
{thinking_content}
</thinking>

**Concise Summary of Transformation Rule:**
"""


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Summarize Chain-of-Thought reasoning from raw LLM generations for ARC tasks using OpenRouter."
    )
    args = parser.parse_args()

    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config/config.yaml not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)

    input_long_cots_file = config.get("filtered_sft_output_file")
    summarized_output_file = config.get("summarized_output_file")

    openrouter_settings = config.get("openrouter_settings", {})
    summarization_model_name = openrouter_settings.get("summarization_model")

    if not input_long_cots_file:
        print("Error: 'filtered_sft_output_file' (input) not found in config.yaml.")
        sys.exit(1)
    if not summarized_output_file:
        print("Error: 'summarized_output_file' (output) not found in config.yaml.")
        sys.exit(1)
    if not summarization_model_name:
        print(
            f"Error: 'summarization_model' key not found within 'openrouter_settings' in config.yaml."
        )
        print(
            "Please add 'summarization_model: your_model_name' under 'openrouter_settings'."
        )
        sys.exit(1)

    print(f"Input Long COTs File: {input_long_cots_file}")
    print(f"Output Summarized File: {summarized_output_file}")
    print(f"Using OpenRouter for Summarization.")
    print(f"Summarization Model: {summarization_model_name}")

    summarization_llm_client = setup_openrouter_client(openrouter_settings)
    if summarization_llm_client is None:
        print("Exiting due to OpenRouter client setup failure for summarization.")
        sys.exit(1)
    print(f"Using settings from 'openrouter_settings' for summarization.")

    arc_data_dir = config.get("arc_data_dir", "data/arc")

    challenges_file = config.get(
        "training_challenges_file", "arc-agi_training_challenges.json"
    )
    print(f"Loading ARC task definitions from: {challenges_file}")
    arc_challenges_path = os.path.join(arc_data_dir, challenges_file)
    arc_tasks = load_arc_tasks(arc_challenges_path)
    if arc_tasks is None:
        print(f"Failed to load ARC tasks from {arc_challenges_path}. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(arc_tasks)} ARC task definitions.")

    os.makedirs(os.path.dirname(summarized_output_file), exist_ok=True)

    processed_count = 0
    skipped_count = 0
    error_count = 0
    total_summary_prompt_tokens = 0
    total_summary_completion_tokens = 0

    long_cots_data = {}
    line_num = 0
    try:
        print(f"Reading long COTs from: {input_long_cots_file}")
        with open(input_long_cots_file, "r", encoding="utf-8") as f:
            for line in f:
                line_num += 1
                try:
                    task_data = json.loads(line.strip())
                    task_id = task_data.get("task_id")
                    if task_id:
                        long_cots_data[task_id] = task_data
                except json.JSONDecodeError as e:
                    print(
                        f"Error decoding JSON on line {line_num}: {e} - Line: '{line.strip()}'"
                    )
                    error_count += 1
        print(f"Read {len(long_cots_data)} unique task entries from input file.")
    except FileNotFoundError:
        print(f"Error: Input file {input_long_cots_file} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file {input_long_cots_file}: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        with open(summarized_output_file, "a", encoding="utf-8") as f_out:
            task_ids_to_process = list(long_cots_data.keys())
            total_tasks = len(task_ids_to_process)

            for idx, task_id in enumerate(task_ids_to_process):
                task_data = long_cots_data[task_id]
                print(
                    f"\n--- Processing Task {idx + 1}/{total_tasks}: ID {task_id} ---"
                )

                long_cot = task_data.get("raw_response")
                if not long_cot or not isinstance(long_cot, str):
                    print(
                        f"Warning: Invalid or missing 'long_cot' for task_id {task_id}. Skipping."
                    )
                    skipped_count += 1
                    continue

                if task_id not in arc_tasks:
                    print(
                        f"Warning: Task ID {task_id} from input file not found in loaded ARC tasks ({challenges_file}). Skipping."
                    )
                    skipped_count += 1
                    continue
                current_arc_task = arc_tasks[task_id]

                try:
                    task_prompt_section = create_task_prompt_section(current_arc_task)
                except Exception as e:
                    print(f"Error creating task prompt section for {task_id}: {e}")
                    error_count += 1
                    continue

                summarization_prompt = SUMMARIZATION_TEMPLATE.format(
                    task_prompt_section=task_prompt_section, thinking_content=long_cot
                )

                summary_response, prompt_tokens, completion_tokens = None, 0, 0
                try:
                    print(
                        f"Requesting summary using Model: {summarization_model_name}..."
                    )
                    summary_response, prompt_tokens, completion_tokens = (
                        call_openrouter_llm(
                            summarization_llm_client,
                            summarization_model_name,
                            summarization_prompt,
                            openrouter_settings,
                        )
                    )
                    total_summary_prompt_tokens += prompt_tokens
                    total_summary_completion_tokens += completion_tokens
                    print(
                        f"Received summary response. Tokens: Prompt={prompt_tokens}, Completion={completion_tokens}"
                    )

                except Exception as e:
                    print(
                        f"Error calling OpenRouter API for summarization on task {task_id}: {e}"
                    )
                    traceback.print_exc()
                    error_count += 1

                    time.sleep(1)
                    continue

                if summary_response:
                    concise_summary = summary_response.strip()

                    if not concise_summary:
                        print(
                            f"Warning: Received an empty summary for task {task_id} after stripping. Skipping."
                        )
                        skipped_count += 1
                        continue

                    output_record = {
                        "task_id": task_id,
                        "summary_model": summarization_model_name,
                        "summary": concise_summary,
                        "summary_prompt_tokens": prompt_tokens,
                        "summary_completion_tokens": completion_tokens,
                    }

                    try:
                        f_out.write(json.dumps(output_record) + "\n")
                        f_out.flush()
                        processed_count += 1
                        print(
                            f"Successfully processed and saved summary for task {task_id}."
                        )
                    except Exception as write_e:
                        print(
                            f"Error writing summary record for task {task_id}: {write_e}"
                        )
                        error_count += 1
                else:
                    print(
                        f"Warning: Received empty/None response for summary of task {task_id}. Skipping."
                    )
                    skipped_count += 1
                    continue

    except IOError as e:
        print(f"Error opening or writing to output file {summarized_output_file}: {e}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during task processing loop: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n--- Summarization Complete ---")
    print(f"Processed {processed_count} tasks successfully.")
    print(
        f"Skipped {skipped_count} tasks (missing data, ARC task not found, empty summary, etc.)."
    )
    print(
        f"Encountered {error_count} errors (JSON decoding, API calls, file writing, etc.)."
    )
    print(f"Results saved to: {summarized_output_file}")
    print(
        f"Total Token Usage for Summarization: Prompts≈{total_summary_prompt_tokens}, Completions≈{total_summary_completion_tokens}"
    )
