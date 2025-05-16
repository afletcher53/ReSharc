import os
import sys
import json
import yaml
import argparse
import time
import uuid


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Project Root (added to sys.path): {project_root}")
print(f"Current Working Directory: {os.getcwd()}")


from dotenv import load_dotenv


from src.llm_utils import (
    setup_openrouter_client,
    setup_openai_client,
)


try:
    from src.arc_utils import (
        load_arc_tasks,
        create_task_prompt_section,
    )
except ImportError:
    print("Error: Could not import functions from src.arc_utils.")
    print("Please ensure src/arc_utils.py exists and contains the required functions.")
    sys.exit(1)


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
    parser.add_argument(
        "--batch_id_to_retrieve",
        type=str,
        default=None,
        help="If provided, skip generation and attempt to retrieve and process this OpenAI Batch ID.",
    )
    args = parser.parse_args()
    start_task_id_arg = args.start_task_id
    batch_id_to_retrieve_arg = args.batch_id_to_retrieve

    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config/config.yaml not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)

    api_provider = "openai_batch"
    print(f"Selected API Provider: {api_provider}")
    llm_client = None
    models_to_use = []
    openai_batch_settings = {}
    api_settings = {}

    if api_provider == "openai_batch":
        openai_batch_settings = config.get("openai_batch_settings", {})
        if not openai_batch_settings:
            print("Error: 'openai_batch_settings' not found in config.yaml")
            sys.exit(1)
        llm_client = setup_openai_client(openai_batch_settings)
        if llm_client is None:
            print("Exiting due to OpenAI client setup failure.")
            sys.exit(1)

        print(
            f"Configured for OpenAI Batch API. Model: {openai_batch_settings.get('batch_model_id')}"
        )
    else:
        print(
            f"Error: Invalid api_provider '{api_provider}'. Use 'openrouter' or 'openai_batch'."
        )
        sys.exit(1)

    arc_data_dir = config.get("arc_data_dir", "data/arc")

    evaluation_challenges_file = config.get(
        "evaluation_challenges_file", "arc-agi_evaluation_challenges.json"
    )
    evaluation_challenges_path = os.path.join(arc_data_dir, evaluation_challenges_file)
    arc_tasks = load_arc_tasks(evaluation_challenges_path)
    if arc_tasks is None:
        sys.exit(1)

    if api_provider == "openai_batch":
        output_model_name = openai_batch_settings.get(
            "batch_model_id", "openai_batch_model"
        ).replace("/", "_")

    output_file = f"data/generated_sft/evaluation_{output_model_name}_generations.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Results will be saved to: {output_file}")

    all_task_ids = list(arc_tasks.keys())
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
                f"Warning: Start Task ID '{start_task_id_arg}' not found. Processing all tasks."
            )
    else:
        print("No start_task_id provided. Starting processing from the beginning.")

    max_tasks = config.get("max_tasks_to_process")
    task_ids_to_process = tasks_to_consider
    if max_tasks is not None and max_tasks > 0:
        print(
            f"Processing a maximum of {max_tasks} tasks from the determined starting point."
        )
        task_ids_to_process = tasks_to_consider[:max_tasks]
    else:
        print(
            f"Processing all {len(tasks_to_consider)} tasks from the determined starting point."
        )

    results_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    template = "REREAD_PROMPT_TEMPLATE"

    if api_provider == "openai_batch":
        openai_model_id = openai_batch_settings.get("batch_model_id")
        batch_input_requests = []
        task_details_for_mapping = {}

        if not batch_id_to_retrieve_arg:
            print("\n--- Preparing OpenAI Batch Input File ---")
            for i, task_id in enumerate(task_ids_to_process):
                task_data = arc_tasks[task_id]
                if not task_data.get("test") or not task_data["test"]:
                    print(f"Skipping Task ID {task_id}: No test cases found.")
                    continue

                task_prompt_section = create_task_prompt_section(task_data)

                full_prompt_messages = [
                    {
                        "role": "user",
                        "content": config.get(template, "")
                        .format(task_prompt_section=task_prompt_section)
                        .strip(),
                    },
                ]

                custom_id = f"task_{task_id}_req_{uuid.uuid4()}"
                task_details_for_mapping[custom_id] = {
                    "task_id": task_id,
                    "test_case_index": 0,
                }

                batch_request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": openai_model_id,
                        "messages": full_prompt_messages,
                        "max_completion_tokens": 5024,
                    },
                }
                batch_input_requests.append(batch_request)
                if (i + 1) % 100 == 0:
                    print(f"Prepared {i + 1} requests for batch...")

            if not batch_input_requests:
                print("No requests to send to OpenAI Batch API. Exiting.")
                sys.exit(0)

            print(f"Total requests prepared for batch: {len(batch_input_requests)}")

            batch_input_file_prefix = openai_batch_settings.get(
                "batch_input_file_prefix", "data/batches/input/batch_input"
            )
            os.makedirs(os.path.dirname(batch_input_file_prefix), exist_ok=True)
            batch_input_filename = (
                f"{batch_input_file_prefix}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
            )

            try:
                with open(batch_input_filename, "w", encoding="utf-8") as f_batch_in:
                    for req in batch_input_requests:
                        f_batch_in.write(json.dumps(req) + "\n")
                print(f"Batch input file created: {batch_input_filename}")
            except IOError as e:
                print(f"Error writing batch input file {batch_input_filename}: {e}")
                sys.exit(1)

            mapping_filename = batch_input_filename.replace(".jsonl", "_mapping.json")
            try:
                with open(mapping_filename, "w", encoding="utf-8") as f_map:
                    json.dump(task_details_for_mapping, f_map)
                print(f"Task mapping file saved: {mapping_filename}")
            except IOError as e:
                print(f"Error writing mapping file {mapping_filename}: {e}")

            print("Uploading batch input file to OpenAI...")
            try:
                with open(batch_input_filename, "rb") as f_upload:
                    batch_input_file_obj = llm_client.files.create(
                        file=f_upload, purpose="batch"
                    )
                print(f"File uploaded successfully. File ID: {batch_input_file_obj.id}")
            except Exception as e:
                print(f"Error uploading file to OpenAI: {e}")
                sys.exit(1)

            print("Creating batch job on OpenAI...")
            try:
                completion_window = openai_batch_settings.get(
                    "batch_completion_window", "24h"
                )
                created_batch = llm_client.batches.create(
                    input_file_id=batch_input_file_obj.id,
                    endpoint="/v1/chat/completions",
                    completion_window=completion_window,
                    metadata={"description": f"ARC Eval Batch for {output_model_name}"},
                )
                print(f"Batch job created successfully. Batch ID: {created_batch.id}")
                print(
                    "You can now monitor this batch ID or re-run this script with --batch_id_to_retrieve <ID> later."
                )
                print(f"To retrieve, use: --batch_id_to_retrieve {created_batch.id}")
                print(f"Also, ensure the mapping file is available: {mapping_filename}")
                sys.exit(0)

            except Exception as e:
                print(f"Error creating batch job on OpenAI: {e}")
                sys.exit(1)

        elif batch_id_to_retrieve_arg:
            batch_id = batch_id_to_retrieve_arg
            print(f"\n--- Retrieving and Processing OpenAI Batch ID: {batch_id} ---")

            potential_mapping_file_prefix = openai_batch_settings.get(
                "batch_input_file_prefix", "data/batches/input/batch_input"
            )

            print(
                f"IMPORTANT: Ensure you have the correct '_mapping.json' file that was generated when batch {batch_id} was created."
            )
            print(
                f"The script will try to infer it, but you might need to specify it if it fails."
            )

            mapping_file_path = None
            batch_input_dir = os.path.dirname(
                openai_batch_settings.get(
                    "batch_input_file_prefix", "data/batches/input/batch_input_dummy"
                )
            )
            try:
                candidate_maps = [
                    os.path.join(batch_input_dir, f)
                    for f in os.listdir(batch_input_dir)
                    if f.endswith("_mapping.json")
                ]
                if candidate_maps:
                    mapping_file_path = max(candidate_maps, key=os.path.getctime)
                    print(
                        f"Attempting to use inferred mapping file: {mapping_file_path}"
                    )
                else:
                    print(
                        f"No mapping file found in {batch_input_dir}. Please specify manually or ensure it exists."
                    )
                    sys.exit(1)

                with open(mapping_file_path, "r", encoding="utf-8") as f_map:
                    task_details_for_mapping = json.load(f_map)
                print(f"Loaded task mapping from {mapping_file_path}")
            except Exception as e:
                print(
                    f"Error loading mapping file '{mapping_file_path}': {e}. Cannot process batch results without it."
                )
                sys.exit(1)

            poll_interval = openai_batch_settings.get("batch_status_poll_interval", 60)
            while True:
                try:
                    batch_status = llm_client.batches.retrieve(batch_id)
                    print(
                        f"Batch '{batch_id}' status: {batch_status.status} (Errors: {batch_status.errors}, Total: {batch_status.request_counts.total}, Failed: {batch_status.request_counts.failed}, Completed: {batch_status.request_counts.completed})"
                    )

                    if batch_status.status == "completed":
                        print("Batch completed. Retrieving results...")
                        output_file_id = batch_status.output_file_id
                        error_file_id = batch_status.error_file_id

                        if error_file_id:
                            print(
                                f"Batch has errors. Error File ID: {error_file_id}. Check OpenAI dashboard for details."
                            )

                        if not output_file_id:
                            print(
                                "Batch completed but no output file ID found. This might indicate all requests failed."
                            )
                            sys.exit(1)

                        output_content_response = llm_client.files.content(
                            output_file_id
                        )
                        batch_results_raw = output_content_response.read().decode(
                            "utf-8"
                        )

                        os.makedirs(
                            openai_batch_settings.get(
                                "batch_output_dir", "data/batches/output/"
                            ),
                            exist_ok=True,
                        )
                        raw_batch_output_filename = os.path.join(
                            openai_batch_settings.get("batch_output_dir"),
                            f"{batch_id}_results.jsonl",
                        )
                        with open(
                            raw_batch_output_filename, "w", encoding="utf-8"
                        ) as f_raw_out:
                            f_raw_out.write(batch_results_raw)
                        print(
                            f"Raw batch results saved to: {raw_batch_output_filename}"
                        )

                        with open(output_file, "a", encoding="utf-8") as f_out_final:
                            for line in batch_results_raw.strip().split("\n"):
                                try:
                                    result_item = json.loads(line)
                                    custom_id = result_item.get("custom_id")
                                    original_task_info = task_details_for_mapping.get(
                                        custom_id
                                    )

                                    if not original_task_info:
                                        print(
                                            f"Warning: No mapping found for custom_id '{custom_id}'. Skipping."
                                        )
                                        continue

                                    response_body = result_item.get("response", {}).get(
                                        "body", {}
                                    )
                                    raw_llm_response = None
                                    prompt_tokens_item = 0
                                    completion_tokens_item = 0

                                    if (
                                        result_item.get("response", {}).get(
                                            "status_code"
                                        )
                                        == 200
                                    ):
                                        choices = response_body.get("choices", [])
                                        if choices:
                                            raw_llm_response = (
                                                choices[0]
                                                .get("message", {})
                                                .get("content")
                                            )
                                        usage = response_body.get("usage", {})
                                        prompt_tokens_item = usage.get(
                                            "prompt_tokens", 0
                                        )
                                        completion_tokens_item = usage.get(
                                            "completion_tokens", 0
                                        )
                                    else:
                                        print(
                                            f"Error in batch response for {custom_id}: Status {result_item.get('response', {}).get('status_code')}, Body: {response_body.get('error', {}).get('message', 'Unknown error')}"
                                        )

                                    if raw_llm_response:
                                        result_record = {
                                            "task_id": original_task_info["task_id"],
                                            "test_case_index": original_task_info[
                                                "test_case_index"
                                            ],
                                            "api_provider": "openai_batch",
                                            "teacher_model": response_body.get(
                                                "model", openai_model_id
                                            ),
                                            "raw_response": raw_llm_response,
                                            "prompt_tokens": prompt_tokens_item,
                                            "completion_tokens": completion_tokens_item,
                                            "custom_id": custom_id,
                                            "batch_id": batch_id,
                                        }
                                        f_out_final.write(
                                            json.dumps(result_record) + "\n"
                                        )
                                        f_out_final.flush()
                                        results_count += 1
                                        total_prompt_tokens += prompt_tokens_item
                                        total_completion_tokens += (
                                            completion_tokens_item
                                        )
                                    else:
                                        print(
                                            f"No valid response content for {custom_id} in batch output."
                                        )

                                except json.JSONDecodeError as json_e:
                                    print(
                                        f"Error decoding JSON from batch output line: {json_e}. Line: '{line[:100]}...'"
                                    )
                                except Exception as e_proc:
                                    print(
                                        f"Error processing result item for custom_id {custom_id}: {e_proc}"
                                    )
                        break

                    elif batch_status.status in [
                        "failed",
                        "expired",
                        "cancelling",
                        "cancelled",
                    ]:
                        print(
                            f"Batch job {batch_status.status}. Cannot retrieve results. Error details (if any): {batch_status.errors}"
                        )
                        if batch_status.error_file_id:
                            print(
                                f"Error File ID: {batch_status.error_file_id}. Check OpenAI dashboard."
                            )
                        sys.exit(1)
                    else:
                        print(
                            f"Waiting for batch to complete. Sleeping for {poll_interval}s..."
                        )
                        time.sleep(poll_interval)

                except Exception as e:
                    print(
                        f"Error while checking batch status or processing results: {e}"
                    )
                    print(f"Sleeping for {poll_interval}s before retrying...")
                    time.sleep(poll_interval)

    else:
        print(f"API Provider '{api_provider}' not implemented for processing loop.")

    print(f"\n--- Generation/Retrieval Complete ---")
    print(f"API Provider Used: {api_provider}")
    if api_provider == "openai_batch" and batch_id_to_retrieve_arg:
        print(
            f"Processed results for Batch ID: {batch_id_to_retrieve_arg} using template: {template} using model: {openai_model_id}"
        )
    elif api_provider == "openai_batch":
        print(
            f"OpenAI Batch input file generated and job submitted. (No direct results in this run)."
        )
    else:
        print(f"Attempted processing for {len(task_ids_to_process)} task IDs.")

    if start_task_id_arg and api_provider != "openai_batch":
        print(
            f"(Processing started from task ID: {start_task_id_arg if start_task_id_arg in all_task_ids else 'beginning (start ID not found)'})"
        )
    print(f"Successfully saved {results_count} generations/results to {output_file}")
    print(
        f"Token Usage (from processed results): Prompts≈{total_prompt_tokens}, Completions≈{total_completion_tokens}"
    )
