from openai import OpenAI

client = OpenAI(
    base_url="",
    api_key=""
)
region_template = {
    "role": "system",
    "content": f"""
        You are an expert assistant specialized in transforming user-stated value scores into concise English preference descriptions.
    
        Given a dictionary of value scores across seven dimensions: creativity, fluency, factuality, diversity, safety, personalisation, and helpfulness.
    
        - Scores >= 80 indicate a "High" preference.
        - Scores <= 60 indicate a "Low" preference.
        - Scores between 61-79 are ignored unless needed for context.
    
        Ignore the fields "values", "other", and "other_text".
    
        Your task: convert the value score dictionary into a short English text summary listing the user's high and low preferences.

        Example:
        
        Input:
        {{
        "values": 10,
        "creativity": 30,
        "fluency": 84,
        "factuality": 74,
        "diversity": 84,
        "safety": 39,
        "personalisation": 9,
        "helpfulness": 93,
        "other": 100,
        "other_text": "is able to differ sources from believable to less believable"
        }}

        Output:
        High helpfulness, High fluency, High diversity, low creativity, low safety, low personalisation.
    """,
}



def chat(message: str, template):
    prompt = [template, {"role": "user", "content": message}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=prompt,
        max_tokens=4096,
        temperature=0,
        top_p=0.925,
    )
    return response.choices[0].message.content

def generate_values(message: str):
    return chat(message, region_template)
