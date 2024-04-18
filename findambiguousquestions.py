import json
from squad.compute_score import f1_score

# find questions that are ambiguous by searching for questions where none of the human answers are very similar

input_file = "./eval/eval_output/eval_predictions.jsonl"

with open(input_file, 'r') as jsonl_file:
    json_list = list(jsonl_file)

bad_questions = []

for json_str in json_list:
    result = json.loads(json_str)
    predicted = result["predicted_answer"]
    best_score = None
    for i in range(len(result["answers"]["text"])):
        for j in range(i+1, len(result["answers"]["text"])):
            score = f1_score(result["answers"]["text"][i], result["answers"]["text"][j])
            if best_score is None or score > best_score:
                best_score = score
    if not best_score is None:
        if best_score < 0.5 and len(result["answers"]["text"]) > 1:
            bad_questions.append(result)

print(len(bad_questions))
print(bad_questions)