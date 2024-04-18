from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_from_disk
import evaluate
import os
import json
import helpers

model_path = "/scratch/general/vast/u1297630/trained_model_fromdisk/"
dataset_path = "./data"
output_dir = "./eval_output_fromdisk"

model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
dataset = load_from_disk(dataset_path)
eval_dataset = dataset['validation']
eval_dataset_featurized = eval_dataset.map(lambda exs: helpers.prepare_validation_dataset_qa(exs, tokenizer), batched=True, num_proc=2, remove_columns=eval_dataset.column_names)
print(eval_dataset_featurized)
metric = evaluate.load("squad_v2")

task_evaluator = evaluate.evaluator("question-answering")
result = task_evaluator.compute(model_or_pipeline=model, data=eval_dataset_featurized, tokenizer=tokenizer, metric=metric, strategy="bootstrap", n_resamples=30)

os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(result, f)