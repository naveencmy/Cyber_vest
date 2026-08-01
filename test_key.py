import os
from google import genai

# Initialize client with your environment variable
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Hello!",
    config={"system_instruction": "You are a helpful coding assistant."}
)

print(response.text)