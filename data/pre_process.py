import pandas as pd
import json
import sys
import uuid
sys.path.append('../')

# Opening JSON file
with open('du_2017_split/raw/json/test.json') as test_json_file:
    test_data = json.load(test_json_file)

para_all_compiled = []
for document in test_data:
    paragraphs = document["paragraphs"]
    for para in paragraphs:
        paragraph_id = uuid.uuid4().hex
        paragraph_text = para["context"]
        para_all_compiled.append({'paragraph_id': paragraph_id, 'paragraph_text': paragraph_text})

print("Json completed.")
print("Number of unique paragraphs: ", len(para_all_compiled))

# Save json to json file
# https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file
with open('paragraphs.json', 'w', encoding='utf-8') as file:
    json.dump(para_all_compiled, file)