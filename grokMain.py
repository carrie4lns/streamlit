import streamlit as st
import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import requests
from pydantic import BaseModel
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Config Model
class AppConfig(BaseModel):
    xai_api_key: str
    grok_model: str = "grok-4"
    max_tokens: int = 500
    documents_dir: str = "./documents"

# Load config
@st.cache_data
def load_config():
    return AppConfig(
        xai_api_key=os.getenv("XAI_API_KEY", ""),
        grok_model=os.getenv("GROK_MODEL", "grok-4"),
        documents_dir=os.getenv("DOCUMENTS_DIR", "./documents")
    )

# Ingest PDFs
@st.cache_resource
def ingest_documents(documents_dir):
    documents = []
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        return []
    for filename in os.listdir(documents_dir):
        if filename.endswith(".pdf"):
            with pdfplumber.open(os.path.join(documents_dir, filename)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents

# Create vector store for ReAG
@st.cache_resource
def create_vector_store(documents):
    if not documents:
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# ReAG Logic: Combine local reasoning with optional xAI API augmentation
def process_prompt(prompt: str, config: AppConfig, vector_store, use_xai_api: bool):
    # Step 1: Retrieve relevant document chunks
    response_text = ""
    if vector_store:
        docs = vector_store.similarity_search(prompt, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        reasoning_prompt = (
            f"Based on the following context, reason through the user's prompt and provide a concise, accurate response.\n"
            f"Context:\n{context}\n\nPrompt: {prompt}"
        )
        response_text = f"Local ReAG Response:\n{reasoning_prompt[:500]}..."  # Simplified for demo
    else:
        response_text = "No documents available for local ReAG."

    # Step 2: Augment with xAI API if enabled
    if use_xai_api and config.xai_api_key:
        try:
            headers = {
                "Authorization": f"Bearer {config.xai_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": config.grok_model,
                "messages": [
                    {"role": "system", "content": "You are Grok, created by xAI. Provide helpful answers."},
                    {"role": "user", "content": prompt + (f"\nContext: {context}" if vector_store else "")}
                ],
                "max_tokens": config.max_tokens
            }
            response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            xai_response = response.json()["choices"][0]["message"]["content"]
            response_text += f"\n\nxAI API Response:\n{xai_response}"
        except Exception as e:
            response_text += f"\n\nxAI API Error: {str(e)}"

    return response_text

# Streamlit UI
st.title("Document Chat with FAA Pubs")

# Load config and documents
config = load_config()
documents = ingest_documents(config.documents_dir)
vector_store = create_vector_store(documents)

# Sidebar for config
st.sidebar.header("Configuration")
config.grok_model = st.sidebar.selectbox("LLM Model", ["grok-4", "grok-3"], index=0)
api_key = st.sidebar.text_input("API Key", type="password", value=config.xai_api_key)
use_xai_api = st.sidebar.checkbox("Augment with Grok xAI API", value=False)
if st.sidebar.button("Save Config"):
    with open(".env", "w") as f:
        f.write(f"XAI_API_KEY={api_key}\nGROK_MODEL={config.grok_model}\n")
    st.sidebar.success("Config saved!")

# Main UI
st.write(f"Loaded {len(documents)} PDFs from {config.documents_dir}")
prompt = st.text_area("Enter your prompt:", height=200)
if st.button("Submit"):
    if prompt:
        with st.spinner("Processing..."):
            response = process_prompt(prompt, config, vector_store, use_xai_api)
        st.write("**Response:**")
        st.markdown(response)
    else:
        st.error("Please enter a prompt.")