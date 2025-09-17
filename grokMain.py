import streamlit as st
import requests
from pydantic import BaseModel
import os
import json
from dotenv import load_dotenv

# Config Layer
class LLMConfig(BaseModel):
    provider: str = "xai"
    model: str = "grok-4"  # Set to grok-4 per the curl command
    api_key: str
    max_tokens: int = 500
    api_base: str = "https://api.x.ai/v1"

# Load config from file or env
@st.cache_data
def load_config():
    config_path = "config.json"
    load_dotenv() # Load environment variables from .env file if present

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = json.load(f)
            # Ensure XAI_API_KEY from .env overrides config.json (for security)
            config_data["api_key"] = os.getenv("XAI_API_KEY") # todo fix for when not XAI config
            return LLMConfig(**config_data)
    else:
        # Fallback to environment variables with hardcoded defaults
        return LLMConfig(
            api_key=os.getenv("XAI_API_KEY"),
            model=os.getenv("LLM_MODEL", "grok-4"),
            api_base=os.getenv("XAI_API_BASE", "https://api.x.ai/v1")
        )

# Core Logic
def submit_prompt(prompt: str, config: LLMConfig) -> str:
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": "You are Grok, a highly intelligent, helpful AI assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": config.max_tokens,
        "stream": False
    }
    try:
        response = requests.post(
            f"{config.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=3600  # Match curl's -m 3600
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# UI Layer
st.title("Grok Prompt App with xAI")

config = load_config()
st.sidebar.header("Config")
# Adjust model options; grok-4 is confirmed, others are placeholders
config.model = st.sidebar.selectbox("Model", ["grok-4", "grok"], index=0)
api_key = st.sidebar.text_input("xAI API Key", type="password", value=config.api_key)
if st.sidebar.button("Save Config"):
    with open("config.json", "w") as f:
        json.dump({"model": config.model, "api_key": api_key, "api_base": config.api_base}, f)
    st.sidebar.success("Config saved!")

prompt = st.text_area("Enter your prompt:", height=200, placeholder="e.g., What is the meaning of life, the universe, and everything?")
if st.button("Submit to Grok"):
    if prompt and api_key:
        with st.spinner("Generating response..."):
            response = submit_prompt(prompt, LLMConfig(model=config.model, api_key=api_key, api_base=config.api_base))
        st.write("**Response:**")
        st.write(response)
    else:
        st.error("Please enter a prompt and xAI API key.")

st.markdown("For xAI API details, visit [x.ai/api](https://x.ai/api).")