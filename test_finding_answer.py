import json
import os


def find_last_list_of_lists(text: str):
    """
    Finds the last substring that parses into a list of lists in the text.

    Args:
        text: The string to search within.

    Returns:
        The parsed list of lists, or None if not found.
    """
    last_checked_end = len(text)
    while True:
        # Find the last potential closing bracket before the last one we checked
        end_index = text.rfind("]", 0, last_checked_end)
        if end_index == -1:
            return None  # No more closing brackets found

        balance = 0
        start_index = -1
        # Search backwards from the found ']' for its matching '['
        for i in range(end_index, -1, -1):
            char = text[i]
            if char == "]":
                balance += 1
            elif char == "[":
                balance -= 1
                if balance == 0:
                    start_index = i
                    break  # Found the matching opening bracket

        if start_index != -1:
            # Extract the potential JSON substring
            potential_json_str = text[start_index : end_index + 1]
            try:
                # Attempt to parse it
                parsed_data = json.loads(potential_json_str)

                # Check if it's a non-empty list containing only lists
                if (
                    isinstance(parsed_data, list)
                    and parsed_data  # Ensure the outer list is not empty
                    and all(isinstance(item, list) for item in parsed_data)
                ):
                    return parsed_data  # Success!

            except json.JSONDecodeError:
                # The substring wasn't valid JSON, ignore and continue searching
                pass

        # Prepare for the next iteration: search before the closing bracket we just processed
        last_checked_end = end_index


# --- Main processing ---

# Replace with the actual path to your JSONL file
jsonl_file_path = "./data/generated_sft/google_gemini_25_pro_raw_generations.jsonl"  # <--- CHANGE THIS


all_extracted_data = []

try:
    with open(jsonl_file_path, "r", encoding="utf-8") as infile:
        print(f"Processing file: {jsonl_file_path}\n" + "=" * 30)
        for i, line in enumerate(infile):
            line_number = i + 1
            try:
                # Parse the JSON object from the line
                data = json.loads(line.strip())
                task_id = data.get("task_id", "UNKNOWN")
                raw_response = data.get("raw_response", "")

                # Extract the last list of lists
                extracted_list = find_last_list_of_lists(raw_response)

                # Store or print the result
                result = {
                    "line": line_number,
                    "task_id": task_id,
                    "extracted_list": extracted_list,
                }
                all_extracted_data.append(result)

                # Optional: Print immediate feedback
                print(f"Line {line_number} (Task: {task_id}):")
                if extracted_list:
                    # print(json.dumps(extracted_list, indent=2)) # Pretty print
                    print(f"  -> Found list with {len(extracted_list)} rows.")
                else:
                    print(f"  -> No valid list-of-lists found.")

            except json.JSONDecodeError:
                print(f"Error: Invalid JSON on line {line_number}. Skipping.")
            except Exception as e:
                print(f"Error processing line {line_number}: {e}")
        print("=" * 30 + "\nProcessing complete.")

except FileNotFoundError:
    print(f"Error: File not found at {jsonl_file_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# You can now work with the 'all_extracted_data' list which contains
# the results for each line. For example:
for item in all_extracted_data:
    if item["extracted_list"]:
        print(f"Task {item['task_id']} list : {item['extracted_list']}")
