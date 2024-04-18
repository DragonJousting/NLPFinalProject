# tries to identify questions where the model predicted incorrectly because there were multiple syntactic candidates for the answer
import json
import os
from squad.compute_score import f1_score
import nltk

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

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
out_predictions = []

for predic in bad_predictions:
    sentences = tokenizer.tokenize(predic["context"])
    for sentence in sentences:
        if predic["predicted_answer"] in sentence:
            flag = False
            for ground_truth in predic["answers"]["text"]:
                if ground_truth in sentence:
                    flag = True
            if "high-conf" in predic["id"] and sentences[0] == sentence:
                flag = True
                #print("adversarial")
            if flag:
                out_predictions.append(predic)
            break

print(len(out_predictions))
print(out_predictions)