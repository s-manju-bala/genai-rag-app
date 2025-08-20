# **Retrieval-Augmented Generation (RAG) system using LangChain + OpenAI + ChromaDB for context-aware Q&A.**

## 📌 **Overview**
This project implements a **Retrieval-Augmented Generation (RAG)** system using **LangChain, OpenAI GPT models, and ChromaDB**.  
It ingests unstructured data (**documents, websites, speeches**), stores them as **embeddings in a vector database**, and generates **context-aware factual answers** to user queries.  

This app demonstrates how **Generative AI + Retrieval Pipelines** can reduce hallucinations and provide **reliable outputs**.  

---

## 🚀 **Features**
- 📄 **Document Ingestion** → Load text files, speeches, or web pages  
- ✂️ **Text Chunking** → Split documents into semantically meaningful chunks  
- 🔎 **Vector Store with ChromaDB** → Store embeddings for efficient semantic search  
- 💡 **Contextual Retrieval** → Fetch most relevant chunks for each query  
- 🤖 **Answer Generation with GPT** → Generate factual, context-rich answers  
- 📊 **LangSmith Tracing** → Monitor and debug RAG pipeline performance  

---

## 🛠 **Tech Stack**
- **Python** 🐍  
- **LangChain**  
- **OpenAI GPT Models**  
- **ChromaDB** 🗄️  
- **BeautifulSoup** (for HTML parsing)  
- **tiktoken** (token management)  
- **LangSmith** (monitoring & tracing)  

---

## 📂 **Project Structure**

genai-rag-app/

│── app.py

│── local_llama.py

│── requirements.txt

│── .env

│── /rag

│    ├── attention.pdf

│    ├── simplerag.ipynb

│    └── speech.txt

---

### 2️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

### 3️⃣ **Set up environment variables**

Create a .env file with:

OPENAI_API_KEY=your_api_key_here

LANGSMITH_API_KEY=your_langsmith_key_here   # optional

---

### 📊 **Example Query**
"Explain the Attention mechanism in Transformers."

### **Output (sample):**
✨ A context-rich explanation retrieved from the ingested “Attention Is All You Need” paper, citing relevant sections to reduce hallucinations.

---

### 🔮 **Future Roadmap**

🚀 Wrap RAG pipeline into a FastAPI backend

📑 Support PDF, Word, and multi-file ingestion

☁️ Deploy on Azure / AWS / GCP

🐳 Add Docker + CI/CD for production readiness

🤝 Extend with multi-agent workflows (LangGraph, Autogen)

---

### 👩‍💻 **Author**

Manju Bala S

AI Engineer & Data Scientist

Specializing in Generative AI, RAG Systems, SQL Optimization with LLMs, and AI Agents

---


