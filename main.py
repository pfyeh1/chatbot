import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.openai import OpenAI
import openai
from pypdf import PdfReader

import json

with open('config.json', 'r') as f:
    config = json.load(f)

openai.api_key = config['openai_api_key']

st.title("Chat with your uploaded file 🖐️ 💬 📚")

def get_pdf_text(pdf_file):
    """extracts text from .pdf file
    :param pdf_file: *.pdf
    :return: text
    """
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def load_data(text):
    """
    create VectoreStoreIndex from of text
    """
    with st.spinner(text = "Loading and indexing docs"):
        documents = [Document(text=text)]
        service_context = ServiceContext.from_defaults(llm = OpenAI(
                                                            model = "gpt-3.5-turbo",
                                                            temperature = .5,
                                                            system_prompt = 
                                                           "Your job is to answer questions on the relate to the documents. Keep your answers technical, based on facts, and explain your reasoning."))
        index = VectorStoreIndex.from_documents(documents, service_context = service_context)
        return index

def chat(index):
    """
    Chat with your document that's in VectorStoreIndex
    """

    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])
        
    # if last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Tinkin..."):
                response = chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history 

upload_option = st.radio("Choose an option:", ["Upload PDF", "Input Text", "Provide URL"])

if upload_option == "Upload PDF":
    pdf_file = st.file_uploader('Choose your .pdf file', type="pdf")

    if pdf_file is not None:
        st.write("Uploaded Filename: ", pdf_file.name)
        
        text = get_pdf_text(pdf_file) # extract text from pdf
        st.write("*PDF content extracted*...")
        st.write(text[:100])
            
        try:
            index = load_data(text = text)
            chat(index)
        except:
            st.warning("Uh oh! Something went wrong!")
    
if upload_option == "Input Text":
    text  = st.text_area("Input text here:")
    if st.button("Submit"):
        try:
            index = load_data(text = text)
            chat(index)
        except:
            st.warning("Uh oh! Something went wrong!")

# initialize session
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
    {"role": "assistant", "content": "Ask me a question about your text!"}
    ]    


