import json

# Load the original JSON file.
input_file = 'data/labeled_prism.json'
output_file = 'data/labeled_prism.jsonl'

# Read the JSON data.
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Write the data to a JSONL file entry by entry.
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in data:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

