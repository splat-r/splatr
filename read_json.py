import json
import pickle
import gzip

with gzip.open("example_submission.json.gz", 'r') as f:
    data = json.loads(f.read().decode('utf-8'))

list1 = list(data.keys())
list2 = sorted(list1)
print(list2)
