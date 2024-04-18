import json
import os
from squad.compute_score import f1_score

input_file = "./eval/eval_output_adversarial_on_adversarialmod/eval_predictions.jsonl"

with open(input_file, 'r') as jsonl_file:
    json_list = list(jsonl_file)

bad_predictions = []

for json_str in json_list:
    result = json.loads(json_str)
    predicted = result["predicted_answer"]
    best_score = None
    for ground_truth in result["answers"]["text"]:
        score = f1_score(predicted, ground_truth)
        if best_score is None or score > best_score:
            best_score = score
    if (best_score < 0.2):
        bad_predictions.append(result)

print(len(bad_predictions))
print(bad_predictions)