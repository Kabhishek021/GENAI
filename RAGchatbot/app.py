
import streamlit as st
import os
from langchain_groq import ChatGroq

from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_Api_key = os.getenv('GROQ_API_KEY')

# Initialize LLM
llm = ChatGroq( model_name="llama3-3b")

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)
from langchain_community.embeddings import HuggingFaceEmbeddings
def create_vector_embedding():
    embeddings = OllamaEmbeddings()
    

    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = PyPDFDirectoryLoader("research_papers")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
    final_documents = text_splitter.split_documents(docs[:10])
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

st.title("RAG Document Q&A with GROQ and Llama")

user_prompt = st.text_input('Enter your query from the research paper')

if st.button('Document Embeddings'):
    vectors = create_vector_embedding()
    st.write('Vector database is ready')

if user_prompt:
    vectors = create_vector_embedding()  # Ensure vectors are available
    retriever = vectors.as_retriever()
    documents_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, documents_chain)
    response = retrieval_chain.invoke({'input': user_prompt})

    st.write(response['answer'])

    with st.expander('Document Similarity Search'):
        for i, doc in enumerate(response.get('context', [])):
            st.write(doc.page_content)
            st.write('--------------------')






























from langchain_ollama import OllamaEmbeddings
