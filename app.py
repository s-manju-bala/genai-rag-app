from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st 
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You're a helpful assisstant. Please respond to user queries"),
        ("user", "Question:{question}")
    ]
)

#streamlit framework
st.title("LangChain Demo with OPENAI API")
input_text = st.text_input("How can I help you?")

# OpenAI API
llm = ChatOpenAI(model = "gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))