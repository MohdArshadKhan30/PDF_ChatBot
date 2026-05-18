import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

# LangChain (NEW STRUCTURE)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# -------------------- CONFIG --------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="📚")
st.header("📚 Chat with your PDF ")

# -------------------- LOAD ENV --------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env")
    st.stop()

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("📂 Upload Document")
    file = st.file_uploader("Upload a PDF", type="pdf")

# -------------------- FUNCTIONS --------------------
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_text(text)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="openai/gpt-oss-120b",
        temperature=0
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------- MAIN --------------------
if file is not None:

    # Extract text
    with st.spinner("📖 Reading PDF..."):
        text = extract_text(file)

    if not text.strip():
        st.error("❌ No readable text found in PDF.")
        st.stop()

    # Split text
    chunks = split_text(text)
    st.success(f"✅ PDF processed | Chunks: {len(chunks)}")

    # Create vector store
    with st.spinner("🔍 Creating embeddings..."):
        embeddings = load_embeddings()
        vector_store = FAISS.from_texts(chunks, embeddings)

    # User input
    user_question = st.text_input("💬 Ask a question from the document:")

    if user_question:

        llm = load_llm()

        # Retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Prompt
        prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context:
{context}

Question:
{question}
""")

        # RAG Chain (Modern LCEL)
        rag_chain = (
            {"context": retriever | format_docs, "question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Generate answer
        with st.spinner("🤖 Thinking..."):
            response = rag_chain.invoke(user_question)

        st.subheader("📌 Answer")
        st.write(response)

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("⚡ Powered by Groq + HuggingFace + LangChain")
