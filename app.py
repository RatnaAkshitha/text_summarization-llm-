import os
import validators
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Load environment variables from .env file
load_dotenv()

# Get Groq API key from environment
groq_api_key = os.getenv("GORQ_API_KEY")

# Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# User input in sidebar (optional override)
with st.sidebar:
    user_input_api_key = st.text_input("Groq API Key (optional, overrides .env)", value="", type="password")
    if user_input_api_key.strip():
        groq_api_key = user_input_api_key

generic_url = st.text_input("URL", label_visibility="collapsed")

# Groq LLM - ✅ FIX: use a supported model
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

# Prompt template
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Button logic
if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or Website)")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load content
                if "youtube.com" in generic_url or "https://youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                docs = loader.load()

                # Summarize
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)
                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
