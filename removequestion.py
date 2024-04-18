import pyarrow as pa
import pyarrow.parquet as pq

input_file = './data/train/data-00000-of-00001.arrow'
with pa.memory_map(input_file, 'r') as source:
    reader = pa.RecordBatchStreamReader(source)
    loaded_arrays = reader.read_all()

# loaded arrays has id, title, context, question, answer in order.
# 1st try making question blank

loaded_arrays = loaded_arrays.set_column(3, pa.field('question', pa.string()) , [[""] * len(loaded_arrays[3])])

for i in loaded_arrays:
    arr = i
    print(f"{arr[0]} .. {arr[-1]}")

with pa.RecordBatchStreamWriter('./no_question.arrow', loaded_arrays.schema) as writer:
    writer.write(loaded_arrays)