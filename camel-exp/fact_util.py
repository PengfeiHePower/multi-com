import re
import json

from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

client = genai.Client(api_key="AIzaSyAwvvlvmLBrj6vCRden7R3w3oV3RtEd41I")
model_id = "gemini-2.0-flash"

google_search_tool = Tool(
    google_search = GoogleSearch()
)

def string_to_dict(output_string):
    pattern = re.compile(
        r"```json\s*(\{[\s\S]*?\})\s*```",  # 1 backslash, DOTALL via [\s\S]
        re.MULTILINE,)
    match = pattern.search(output_string)
    if not match:
        raise ValueError("No JSON block found")
    data = json.loads(match.group(1))
    return data

def get_fact_flag(context, text):
    system_prompt = """
    Please check the factual correctness of the following text given the context with the help of google search.
    """
    user_prompt_template = "Context: <<context>>\nText: <<text>>"
    user_prompt = user_prompt_template.replace("<<context>>", context)
    user_prompt = user_prompt.replace("<<text>>", text)
    output_format = """Respond a JSON dictionary in a markdown's fenced code block as follows:
    ```json
    {
    "Factual error": True/False,
    "Evidence":
    }
    ```"""
    response = client.models.generate_content(
    model=model_id,
    contents=system_prompt+user_prompt+output_format,
    config=GenerateContentConfig(
        tools=[google_search_tool],
        response_modalities=["TEXT"],
        )
        )
    response_dict = string_to_dict(response.candidates[0].content.parts[0].text)
    # print(f"response_dict: {response_dict}")
    return response_dict["Factual error"]