import pyarrow as pa
import pyarrow.parquet as pq
import ast
import random

input_file = './data/train/data-00000-of-00001.arrow'
with pa.memory_map(input_file, 'r') as source:
    reader = pa.RecordBatchStreamReader(source)
    loaded_arrays = reader.read_all()

# loaded arrays has id, title, context, question, answer in order.
# 1st try making question blank
# now try removing context
# however squad requires that the answer be a span of the context so in order to make it possible for the model to perform well
# (even though if it does it is an indicator of dataset artifacts) we need to include the answer in the context
# We cannot include just the answer as that would be creating its own artifact in a very obvious way so we must include many answers
# but no real information

#442 different titles in train
#110k titles, so thats about 250 questions per title
#19029 different contexts, 5-6 questions per context

#35 different titles in validation

#build new contexts
different_contexts = {}
for i in range(len(loaded_arrays[1])):
    context = str(loaded_arrays[2][i])
    if context in different_contexts.keys():
        answer = ast.literal_eval(str(loaded_arrays[4][i]))
        for answer_text in answer[0][1]:
            different_contexts[context] = different_contexts[context] + answer_text + " "
    else:
        answer = ast.literal_eval(str(loaded_arrays[4][i]))
        #if len(answer[0][1]) == 0:
        #    print(answer)
        for i in range(len(answer[0][1])):
            if i == 0:
                different_contexts[context] = answer[0][1][i] + " "
            else:
                different_contexts[context] = different_contexts[context] + answer[0][1][i] + " "
print(len(different_contexts))

#put new contexts in array
contexts = []
for i in range(len(loaded_arrays[1])):
    #print(len(context))
    context = str(loaded_arrays[2][i])
    # in squad_v2 sometimes there is no answer to the question in the context so the model is supposed to return nothing
    # some contexts have no questions with an answer, so for those we will choose a random context
    if context in different_contexts.keys():
        contexts.append([different_contexts[context]])
    else:
        contexts.append([different_contexts[random.choice(list(different_contexts.keys()))]])
print(len(contexts))
print(random.choice(contexts))



loaded_arrays = loaded_arrays.set_column(2, pa.field('context', pa.string()), contexts)

for i in loaded_arrays:
    arr = i
    print(f"{arr[0]} .. {arr[-1]}")

with pa.RecordBatchStreamWriter('./no_context.arrow', loaded_arrays.schema) as writer:
    writer.write(loaded_arrays)
print('done')