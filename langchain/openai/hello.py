#pip install openai
import os
from openai import OpenAI
from dotenv import load_dotenv

# This loads variables from .env into os.environ
load_dotenv()  

# Now available globally in your app
openai_api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=openai_api_key)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What is temperature in Bangalore today"}]
)

print(response.choices[0].message.content)
