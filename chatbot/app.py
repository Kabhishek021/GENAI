
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_API_KEY"]= os.getenv('LANGCHAIN_API_KEY')

os.environ["LANGCHAIN_TRACKING"] = 'true'


prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are helpfule assistant .Please response to the useerr question?'),
        ('user',"Question:{question}")
    ]
)

#streamlit frameword
st.title('Langchain Demo with Ollama ')
input_text = st.text_input('Serach the topic you want ')

# Ollama

llm = Ollama(model ='mistral')
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


if input_text:
    st.write(chain.invoke({'question':input_text}))