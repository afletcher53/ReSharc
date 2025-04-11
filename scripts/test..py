# load up json in data/arc/arc-agi_evaluation_challenges.json
import json


with open("./data/arc/arc-agi_evaluation_challenges.json", "r", encoding="utf-8") as f:
    testing_challenges = json.load(f)
    testing_challenges = {k: testing_challenges[k] for k in testing_challenges.keys()}

testing_keys = list(testing_challenges.keys())

with open(
    "data/outputs/20250410_194219_baseline_replies.json", "r", encoding="utf-8"
) as f:
    baseline_replies = json.load(f)
    baseline_replies = {k: baseline_replies[k] for k in baseline_replies.keys()}


baseline_keys = list(baseline_replies.keys())

print("testing keys: ", len(testing_keys))
print("baseline keys: ", len(baseline_keys))

# find the missing keys in baseline_replies
missing_keys = []

missing_keys = [k for k in testing_keys if k not in baseline_keys]

print("missing keys: ", missing_keys)
