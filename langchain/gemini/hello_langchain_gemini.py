from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# This loads variables from .env into os.environ
load_dotenv()

# Now available globally in your app
gemini_api_key = os.environ["GOOGLE_API_KEY"]

# Step 1: Initialize Gemini Chat Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# Step 2: Define Prompt
prompt = ChatPromptTemplate.from_template("Say hello to {name}")

# Step 3: Create Chain (Prompt → LLM → Output Parser)
chain = prompt | llm | StrOutputParser()

# Step 4: Run
response = chain.invoke({"name": "LangChain"})
print(response)
