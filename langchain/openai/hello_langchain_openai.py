# pip install -U langchain-openai
# pip install -U langchain-community
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# This loads variables from .env into os.environ
load_dotenv()  

# Now available globally in your app
openai_api_key = os.environ["OPENAI_API_KEY"]
model_name = "gpt-3.5-turbo"

llm = ChatOpenAI(
    model_name = model_name,
    openai_api_key = openai_api_key
)

response = llm.invoke([HumanMessage(content="What's lastest news on India and Pakistan cease fire!")])
print(response.content)