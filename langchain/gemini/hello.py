#pip install -q -U google-genai
#pip install -U google-generativeai

import os
import google.generativeai as genai
from dotenv import load_dotenv

# This loads variables from .env into os.environ
load_dotenv()

# Now available globally in your app 
gemini_api_key = os.environ["GOOGLE_API_KEY"]

#List of models are accessible to your API key
genai.configure(api_key=gemini_api_key)

for m in genai.list_models():
    print(m.name, m.supported_generation_methods)




