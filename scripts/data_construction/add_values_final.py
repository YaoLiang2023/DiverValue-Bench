import json
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
from chatbot_final import generate_values
from datetime import datetime


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# Load the dataset.
ds = load_dataset("json", data_files={"train": "./data/merged.jsonl"})

# Create a list to store the processed data.
processed_data = []

# Iterate through the dataset.
for i, row in tqdm(enumerate(ds['train']), desc='processing'):
    # If the row is a string (in JSON format), parse it into a dictionary.
    if isinstance(row, str):
        row = json.loads(row)
    
    # Ensure that `row` is a dictionary and contains the `'stated_prefs'` key.
    if isinstance(row, dict) and 'survey.stated_prefs' in row:
        feedback = row['survey.stated_prefs']
        print(f"feedback: {feedback}")
        try:
            values = generate_values(str(feedback))
            row['preference'] = values
            processed_data.append(row)
        except Exception as e:
            print(f"[Error] Failed to process row {i} due to: {e}")
            continue  # Skip entries with errors.
    else:
        print(f"Unexpected format for row {i}: {row}")

# Save as JSON format.
with open('data/labeled_prism.json', 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4, cls=DateTimeEncoder)
