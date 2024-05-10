import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import AzureOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from googlesearch import search
import PyPDF2

# Load environment variables from .env file
load_dotenv()

OPENAI_API_VERSION = "2024-02-01"
azure_openai_api_key = os.environ.get('AZURE_OPENAI_API_KEY')

llm = AzureOpenAI(
    deployment_name="Dheeman",
    api_version=OPENAI_API_VERSION,
    azure_endpoint="https://d2912.openai.azure.com/",
    temperature=1
)

# Initialize session state
if 'msg' not in st.session_state:
    st.session_state['msg'] = [
        SystemMessage(content="Answer anything that human asks")
    ]

def ask_openai(question):
    st.session_state['msg'].append(HumanMessage(content=question))
    prompt = "\n".join(msg.content for msg in st.session_state['msg'])
    answer = llm(prompt)
    st.session_state['msg'].append(AIMessage(content=answer))
    return answer

def search_google(query):
    search_results = list(search(query, num=5, stop=5, pause=2))
    return search_results

def analyze_pdf(uploaded_file):
    if uploaded_file is not None:
        uploaded_file_bytes = uploaded_file.read()
        pdf_file = PyPDF2.PdfFileReader(uploaded_file_bytes)
        chunks = []
        for page_num in range(pdf_file.numPages):
            page = pdf_file.getPage(page_num)
            text = page.extract_text()
            chunks.extend(text.split("\n\n"))
        return chunks
    else:
        return None

st.set_page_config(page_title="Chatbot")
st.header("Ingram Chatbot Application")
st.markdown("Made with ❤️ in India")

input_type = st.radio("Select Input Type:", ("Text Input", "PDF Upload"))

if input_type == "Text Input":
    input_question = st.text_input("Ask your questions: ", key="input_question")
    if st.button("Ask"):
        # Get answer from OpenAI language model
        response = ask_openai(input_question)
        st.subheader("The Response from OpenAI:")
        st.write(response)
else:
    # File upload section
    st.sidebar.header("PDF Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file is not None:
        # Analyze the uploaded PDF file
        pdf_chunks = analyze_pdf(uploaded_file)
        if pdf_chunks:
            # Text input box for asking questions
            st.header("Ask Questions About the Uploaded PDF")
            pdf_question = st.text_input("Ask your questions here: ")
            if st.button("Ask"):
                # Get answer from OpenAI language model
                pdf_response = ask_openai(pdf_question)
                st.subheader("The Response from OpenAI:")
                st.write(pdf_response)
        else:
            st.write("Sorry, the uploaded file appears to be invalid or empty.")
