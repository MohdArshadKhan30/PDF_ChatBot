# 📚 PDF ChatBot — RAG-based Document Q&A

A conversational AI app that lets you upload any PDF and ask questions from it in natural language. Built using Retrieval-Augmented Generation (RAG) with LangChain, Groq LLM, FAISS vector store, and Streamlit.

---

## 🚀 Demo

> Upload a PDF → Ask a question → Get a context-aware answer instantly.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| LLM | Groq (gpt-oss-120b) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Framework | LangChain (LCEL) |
| PDF Parsing | PyPDF |

---

## ⚙️ How It Works

1. **Upload** a PDF via the sidebar
2. **Text extraction** — PyPDF reads all pages
3. **Chunking** — RecursiveCharacterTextSplitter splits text into 1000-character chunks with 150-character overlap
4. **Embedding** — HuggingFace sentence transformers convert chunks into vectors
5. **Vector store** — FAISS indexes all chunks for fast semantic search
6. **Query** — User question is matched against top 3 relevant chunks
7. **Answer** — Groq LLM generates a response grounded only in the retrieved context

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot
pip install -r requirements.txt
```

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
pdf-chatbot/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not committed)
└── README.md
```

---

## 📋 Requirements

```
streamlit
python-dotenv
pypdf
langchain
langchain-core
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-groq
faiss-cpu
sentence-transformers
huggingface-hub
```

---

## 🔑 Getting a Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up and navigate to API Keys
3. Generate a new key and paste it in your `.env` file

---

## 💡 Features

- Upload any text-based PDF
- Semantic search over document content
- Answers strictly grounded in uploaded document (no hallucination)
- Fast inference via Groq API
- Clean, minimal Streamlit UI

---

## 🙋 Author

**Mohd Arshad Khan**
