import os
import io

from PyPDF2 import PdfReader
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import streamlit as st

_ = load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

st.title("Gemma Model Document QnA")

model = ChatGroq(model="groq/compound")


prompt_template = """
Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer.
Context:\n{context}?\n
Question: \n{question}\n
Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
output_parser = StrOutputParser()

chain = prompt | model | output_parser


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        st.session_state.loader = PyPDFDirectoryLoader("./docs")  ## Data Ingestion
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(
            st.session_state.docs
        )
        st.session_state.vector_store = FAISS.from_documents(
            st.session_state.final_docs, st.session_state.embeddings
        )


input = st.text_input("What you want to ask from the docs?")

if st.button("Creating Vector Store"):
    vector_embedding()
    st.write("Vector Store DB is ready!")

if input:
    output_parser = StrOutputParser()
    retriever = st.session_state.vector_store.as_retriever()

    # Format retrieved documents into a single context string
    format_docs = lambda docs: "\n\n".join(doc.page_content for doc in docs)

    # Correct RAG chain using modern LCEL pattern
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | output_parser
    )

    response = rag_chain.invoke(input)

    st.write(response)

    with st.expander("Retrieved Documents"):
        retrieved_docs = st.session_state.vector_store.similarity_search(input, k=4)
        for i, doc in enumerate(retrieved_docs, 1):
            st.write(f"**Document {i}** (Source: {doc.metadata.get('source', 'unknown')})")
            st.write(doc.page_content)
            st.write("---")
