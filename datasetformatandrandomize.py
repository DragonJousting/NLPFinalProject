# randomize the place in the paragraph where the adversarial sentence appears and convert json to .arrow file
import json
import pyarrow as pa
import random
import nltk

#nltk.download('punkt')

merge_with_full_dataset = True

f=open("sample1k-HCVerifyAll.json", "r")
s=json.load(f)
f.close()

s=s['data']

# want to create an arrow file with columns id, title, context, question, answers in that order

def reformat_answers(answers):
    texts = []
    answer_starts = []
    for pair in answers:
        texts.append(pair['text'])
        answer_starts.append(pair['answer_start'])
    out_dict = {"text": texts, "answer_start": answer_starts}
    return out_dict

input_file = './datasets/data/train/data-00000-of-00001.arrow'
with pa.memory_map(input_file, 'r') as source:
    reader = pa.RecordBatchStreamReader(source)
    loaded_arrays_train = reader.read_all()

input_file = './datasets/data/validation/data-00000-of-00001.arrow'
with pa.memory_map(input_file, 'r') as source:
    reader = pa.RecordBatchStreamReader(source)
    loaded_arrays_valid = reader.read_all()

id = []
title = []
context = []
question = []
answers = []
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
for subject in s:
    subj_title = subject['title']
    for context_bundle in subject['paragraphs']:
        subj_context = context_bundle['context']
        sentences = tokenizer.tokenize(subj_context)

        for qas in context_bundle['qas']:
            cont_question = qas['question']
            q_id = qas['id']
            if "high-conf" in q_id:
                # example is adversarial
                new_position = random.randint(0, len(sentences)-1)
                adver_sentence = sentences.pop()
                sentences.insert(new_position, adver_sentence)
                subj_context = " ".join(sentences)
                #print(subj_context)
                #print("context randomized")
            #else:
                #print("was not adversarial")
            #print(type(qas['answers']))
            cont_answers = reformat_answers(qas['answers'])
            #print(type(cont_answers))
            id.append(q_id)
            title.append(subj_title)
            context.append(subj_context)
            question.append(cont_question)
            answers.append(cont_answers)
        
# ints are automatically infered as int64's by pyarrow but we need it to be an int32 to match the default squad dataset
fields = [('text', pa.list_(pa.string())), ('answer_start', pa.list_(pa.int32()))]
answers = pa.array(answers, type=pa.struct(fields))
print(len(id))

if merge_with_full_dataset:
    # randomly separate data into 90% training and 10% validation
    order = [i for i in range(len(id))]
    random.shuffle(order)

    train_count = int(len(id)*.9)
    valid_count = len(id) - train_count

    id_train = id[0:train_count]
    id_valid = id[train_count:-1]

    title_train = title[0:train_count]
    title_valid = title[train_count:-1]

    context_train = context[0:train_count]
    context_valid = context[train_count:-1]

    question_train = question[0:train_count]
    question_valid = question[train_count:-1]

    answers_train = answers[0:train_count]
    answers_valid = answers[train_count:-1]


    names = ["id", "title", "context", "question", "answers"]

    


    train_table = pa.Table.from_arrays([id_train, title_train, context_train, question_train, answers_train], names=names)
    #print(train_table.schema)
    #print(loaded_arrays_train.schema)
    train_table = pa.concat_tables([loaded_arrays_train, train_table])

    valid_table = pa.Table.from_arrays([id_valid, title_valid, context_valid, question_valid, answers_valid], names=names)
    valid_table = pa.concat_tables([loaded_arrays_valid, valid_table])

    print(len(valid_table[0]))
    print(len(train_table[0]))

    for i in valid_table:
        arr = i
        print(f"{arr[0]} .. {arr[-1]}")

    with pa.RecordBatchStreamWriter('./train-data-00000-of-00001.arrow', train_table.schema) as writer:
        writer.write(train_table)
    with pa.RecordBatchStreamWriter('./valid-data-00000-of-00001.arrow', train_table.schema) as writer:
        writer.write(valid_table)
    print("done")

else:
    names = ["id", "title", "context", "question", "answers"]

    table = pa.Table.from_arrays([id, title, context, question, answers], names=names)
    
    with pa.RecordBatchStreamWriter('./data-00000-of-00001.arrow', table.schema) as writer:
        writer.write(table)
    print("done")