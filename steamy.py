import streamlit as st
from streamlit_chat import message
import openai
# from azure.ai.openai import OpenAICredentials, OpenAIClient
import os
import tempfile
from langchain_community.llms import Ollama

# # Set up Azure OpenAI credentials
# openai.api_type = "azure"
# openai.api_base = "https://YOUR_AZURE_OPENAI_ENDPOINT.openai.azure.com"
# openai.api_version = "2023-03-15-preview"
# openai.api_key = "YOUR_AZURE_OPENAI_KEY"

# Create an OpenAI client
client = Ollama(model="mistral")
prompt="Act like a virtual assistant"
# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'file_content' not in st.session_state:
    st.session_state.file_content = None

# Function to handle file upload
def handle_file_upload():
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        with open(tmp_path, "r", encoding="utf-8") as f:
            file_content = f.read()
        st.session_state.file_content = file_content
        os.unlink(tmp_path)

# Function to generate response from the language model
def generate_response(prompt):
    if st.session_state.file_content:
        prompt += "\n\n" + st.session_state.file_content
    response = client.generate(prompt=prompt)
    return response.choices[0].text

# Streamlit app
st.title("ChatBot")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You: ", key="input")

if user_input.lower() == "file":
    handle_file_upload()
    st.session_state.history.append(("You", "file"))
    st.session_state.history.append(("Assistant", "Please upload a file to use for answering queries."))
elif user_input:
    output = generate_response(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Assistant", output))

for sender, message in st.session_state.history:
    message(sender + ": " + message, key=str(len(st.session_state.history)))