# rag_ui.py
import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit Config
st.set_page_config(page_title="GenAI RAG App", layout="wide")

# Sidebar Info
with st.sidebar:
    st.markdown("## 📌 About this App")
    st.write("This demo shows a **Retrieval-Augmented Generation (RAG)** pipeline.")
    st.markdown("**Tech Stack:** Python, Streamlit, LangChain, OpenAI, ChromaDB")
    st.markdown("👩‍💻 **Author:** Manju Bala S")

# Title
st.markdown("<h1 style='text-align: center;'>📚 GenAI RAG App – Context-Aware Q&A</h1>", unsafe_allow_html=True)

# User Query
query = st.text_input("💬 Ask a question from the ingested docs (e.g., *What is Attention in Transformers?*):")

if query:
    # --- Load Vector DB (must be created beforehand) ---
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    db = Chroma(persist_directory="rag/chroma_db", embedding_function=embeddings)

    retriever = db.as_retriever()

    # --- LLM ---
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # --- Retrieval QA Chain ---
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # --- Run Query ---
    response = qa_chain.run(query)

    # --- Display Answer ---
    st.subheader("📘 Answer")
    st.success(response)

    # --- Show Retrieved Context ---
    with st.expander("🔎 Retrieved Chunks"):
        docs = retriever.get_relevant_documents(query)
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Chunk {i}:** {doc.page_content[:400]}...")
