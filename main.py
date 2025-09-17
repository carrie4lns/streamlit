import streamlit as st
import openai
from pydantic import BaseModel
import os
import json

# Config Layer
class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    api_key: str
    max_tokens: int = 500

    # Load config from file or env
@st.cache_data
def load_config():
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return LLMConfig(**json.load(f))
    else:
        # Fallback to env vars
        return LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        )
    
    # Core Logic
def submit_prompt(prompt: str, config: LLMConfig) -> str:
    openai.api_key = config.api_key
    try:
        response = openai.ChatCompletion.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# UI Layer
st.title("Simple LLM Prompt App")

config = load_config()
st.sidebar.header("Config")
config.model = st.sidebar.selectbox("Model", ["grok-code-fast-1", "gpt-3.5-turbo", "gpt-4"], index=0)
api_key = st.sidebar.text_input("API Key", type="password", value=config.api_key)
if st.sidebar.button("Save Config"):
    with open("config.json", "w") as f:
        json.dump({"model": config.model, "api_key": api_key}, f)
    st.sidebar.success("Config saved!")

prompt = st.text_area("Enter your prompt:", height=200)
if st.button("Submit to LLM"):
    if prompt and api_key:
        with st.spinner("Generating response..."):
            response = submit_prompt(prompt, LLMConfig(model=config.model, api_key=api_key))
        st.write("**Response:**")
        st.write(response)
    else:
        st.error("Please enter a prompt and API key.")
