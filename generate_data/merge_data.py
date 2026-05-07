import json
from collections import defaultdict

def load_survey(file_path):
    """
    Load the survey JSONL file and return a dictionary with `user_id` as the key and the survey content as the value.
    """
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            user_id = obj.get('user_id')
            if user_id is not None:
                data[user_id] = obj
    return data

def load_conversations(file_path):
    """
    Load the conversations JSONL file and return a dictionary with `user_id` as the key, and the value being a list of all records associated with that `user_id`.
    """
    data = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            user_id = obj.get('user_id')
            if user_id is not None:
                data[user_id].append(obj)
    return data

def merge_jsonl(survey_path, conversations_path, output_path):
    survey_data = load_survey(survey_path)
    conversations_data = load_conversations(conversations_path)

    with open(output_path, 'w', encoding='utf-8') as fout:
        for user_id, survey_item in survey_data.items():
            conv_list = conversations_data.get(user_id, [])
            if not conv_list:
                # No corresponding conversation data found — skip or handle as needed based on requirements.
                continue

            for conv_item in conv_list:
                merged = {}

                # survey 部分
                merged['survey.user_id'] = survey_item.get('user_id')
                merged['survey.stated_prefs'] = survey_item.get('stated_prefs')
                merged['survey.age'] = survey_item.get('age')
                merged['survey.gender'] = survey_item.get('gender')
                merged['survey.self_description'] = survey_item.get('self_description')
                merged['survey.system_string'] = survey_item.get('system_string')
                merged['survey.employment_status'] = survey_item.get('employment_status')
                merged['survey.education'] = survey_item.get('education')
                merged['survey.marital_status'] = survey_item.get('marital_status')
                merged['survey.english_proficiency'] = survey_item.get('english_proficiency')
                merged['survey.generated_datetime'] = survey_item.get('generated_datetime')

                location = survey_item.get('location', {})
                merged['survey.location.birth_country'] = location.get('birth_country')

                # conversations
                merged['conversations.conversation_id'] = conv_item.get('conversation_id')
                merged['conversations.user_id'] = conv_item.get('user_id')
                merged['conversations.opening_prompt'] = conv_item.get('opening_prompt')
                merged['conversations.conversation_history'] = conv_item.get('conversation_history')
                merged['conversations.open_feedback'] = conv_item.get('open_feedback')
                merged['conversations.generated_datetime'] = conv_item.get('generated_datetime')

                fout.write(json.dumps(merged, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    survey_path = 'data/survey.jsonl'
    conversations_path = 'data/conversations.jsonl'
    output_path = 'data/merged.jsonl'

    merge_jsonl(survey_path, conversations_path, output_path)
    print(f"Merge completed. Results saved to: {output_path}")
