import pyarrow as pa

input_file = './train-data-00000-of-00001.arrow'
with pa.memory_map(input_file, 'r') as source:
    reader = pa.RecordBatchStreamReader(source)
    loaded_arrays = reader.read_all()

for i in loaded_arrays:
    arr = i
    print(f"{arr[0]}, {arr[1]} .. {arr[-2]}, {arr[-1]}")