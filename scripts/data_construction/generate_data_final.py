import json
from openai import OpenAI
import time
import os

client = OpenAI(
    base_url="",
    api_key=""
)


MODEL_NAME = "gpt-4o"
# --------------------------------------------

def call_openai_chat(messages, model=MODEL_NAME, max_tokens=512, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        return None

def generate_question_and_answers(conversation_history, preference, self_description, system_string):
    context_text = "\n".join([turn.get("content", "") for turn in conversation_history if isinstance(turn, dict)])

    prompt = f"""
You are a question-answer generation assistant. Your task is:

Based on the conversation history and the user’s multi-value preference description, generate 3 questions, each with two distinct answers:
- The first answer (answer_w) must clearly reflect the user's stated value preferences.
- The second answer (answer_l) must clearly contradict or ignore the user's value preferences.
- Answers must be contextually relevant to the conversation history, and answer_w and answer_l must be sharply contrasted.

Conversation history:
{conversation_history}

User's multi-value preference description:
- preference: {preference}
- self_description: {self_description}
- system_string: {system_string}

Please return 3 records in JSON array format. Each record should follow this structure:

[
  {{
    "question": "...",
    "answer_w": "...",
    "answer_l": "..."
  }},
  ...
]

IMPORTANT:
- Only output a valid JSON array.
- Do NOT include explanations, code blocks (e.g., ```json), Markdown, or any extra commentary.
- Ensure linguistic clarity and strong value alignment contrast between answer_w and answer_l.
"""

    messages = [
        {"role": "system", "content": "You are a specialized Q\&A pair generation assistant designed to model and reflect diverse user value preferences."},
        {"role": "user", "content": prompt}
    ]

    response_text = call_openai_chat(messages)

    if not response_text:
        return []

    try:
        qa_list = json.loads(response_text)
        if isinstance(qa_list, list):
            return qa_list
        else:
            print("Model response is not a list. Content is as follows:")
            print(response_text)
            return []
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        print("Model response:")
        print(response_text)
        return []

def process_jsonl(input_path, output_path):
    output = []
    with open(input_path, 'r', encoding='utf-8') as fin:
        for idx, line in enumerate(fin):
            try:
                item = json.loads(line)

                conversation_history = item.get("conversations.conversation_history", [])
                preference = item.get("preference", "")

                self_description = item.get("survey.self_description", "")
                system_string = item.get("survey.system_string", "")

                # Additional fields.
                user_id = item.get("survey.user_id", "")
                stated_prefs = item.get("survey.stated_prefs", "")
                age = item.get("survey.age", "")
                gender = item.get("survey.gender", "")
                employment_status = item.get("survey.employment_status", "")
                education = item.get("survey.education", "")
                marital_status = item.get("survey.marital_status", "")
                english_proficiency = item.get("survey.english_proficiency", "")
                birth_country = item.get("survey.location.birth_country", "")

                print(f"Processing data entry {idx}，user_id={user_id}...")

                try:
                    qa_pairs = generate_question_and_answers(conversation_history, preference, self_description, system_string)
                except Exception as e:
                    print(f"`generate_question_and_answers` call failed, skipping this entry. Error message: {e}")
                    qa_pairs = []

                for i, qa in enumerate(qa_pairs):
                    record = {
                        "id": f"p{idx*3 + i}",
                        "question": qa.get("question", ""),
                        "answer_w": qa.get("answer_w", ""),
                        "answer_l": qa.get("answer_l", ""),
                        # Original JSONL fields.
                        "user_id": user_id,
                        "stated_prefs": stated_prefs,
                        "age": age,
                        "gender": gender,
                        "employment_status": employment_status,
                        "education": education,
                        "marital_status": marital_status,
                        "english_proficiency": english_proficiency,
                        "birth_country/region": birth_country,
                        "self_description": self_description,
                        "system_string": system_string,
                        "preference": preference,
                    }
                    output.append(record)

            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON on line {idx}, skipping this line. Error message: {e}")
            except Exception as e:
                print(f"An unknown error occurred while processing entry {idx}, skipping. Error message: {e}")

    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(output, fout, ensure_ascii=False, indent=2)

    print(f"All processing completed. Results saved to {output_path}")

if __name__ == "__main__":
    input_file = "data/labeled_prism.jsonl"
    output_file = "data/DiverValue-Bench_dataset.json"
    process_jsonl(input_file, output_file)
