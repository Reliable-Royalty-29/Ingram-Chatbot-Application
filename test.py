import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import AzureOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from googlesearch import search
from io import BytesIO
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

def test(question):
    st.session_state['msg'].append(HumanMessage(content=question))
    prompt = "\n".join(msg.content for msg in st.session_state['msg'])
    answer = llm(prompt)
    st.session_state['msg'].append(AIMessage(content=answer))
    return answer

def search_google(query):
    search_results = list(search(query, num=5, stop=5, pause=2))
    return search_results

def analyze_pdf(uploaded_file):
    uploaded_file_bytes = uploaded_file.read()
    pdf_file = BytesIO(uploaded_file_bytes)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    chunks = []
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        chunks.extend(text.split("\n\n"))
    return chunks


st.set_page_config(page_title="Chatbot")
st.header("Ingram Chatbot Application")
st.markdown("Made with ❤️ in India")

input_type = st.radio("Select Input Type:", ("Text Input", "PDF Upload"))

if input_type == "Text Input":
    input_question = st.text_input("Ask your questions: ", key="input_question")
    if st.button("Ask"):
        # Get answer from OpenAI language model
        response = test(input_question)
        st.subheader("The Response from OpenAI:")
        st.write(response)
else:
    # File upload section
    st.sidebar.header("PDF Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file is not None:
        # Analyze the uploaded PDF file
        pdf_chunks = analyze_pdf(uploaded_file)
        
        # Text input box for asking questions
        st.header("Ask Questions About the Uploaded PDF")
        pdf_question = st.text_input("Ask your questions here: ")
        if st.button("Ask"):
            # Get answer from OpenAI language model
            pdf_response = test(pdf_question)
            st.subheader("The Response from OpenAI:")
            st.write(pdf_response)
