import streamlit as st
import base64
import requests
import time
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from requests.exceptions import RequestException
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(layout="wide")

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'summary' not in st.session_state:
    st.session_state.summary = None

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": "Bearer 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"} 

def query(payload, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            if attempt == max_retries - 1:
                st.error(f"API request failed after {max_retries} attempts: {str(e)}")
                return None
            else:
                st.warning(f"API request failed. Retrying in {delay} seconds...")
                time.sleep(delay)

def file_preprocessing(file):
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        texts = text_splitter.split_text(text)
        final_texts = " ".join(texts)
        return final_texts
    except Exception as e:
        st.error(f"Error in file preprocessing: {str(e)}")
        return None

def llm_pipeline(file):
    input_text = file_preprocessing(file)
    if input_text is None:
        return "Error in file preprocessing"
    
    chunks = [input_text[i:i+1000] for i in range(0, len(input_text), 1000)]
    
    summaries = []
    for chunk in chunks:
        output = query({
            "inputs": chunk,
            "parameters": {"max_length": 150, "min_length": 50}
        })
        if output is None:
            continue
        if isinstance(output, list) and len(output) > 0:
            summary = output[0].get('summary_text', '')
            summaries.append(summary)
        else:
            st.warning(f"Unexpected output format: {output}")
    
    if not summaries:
        return "Failed to generate summary"
    return " ".join(summaries)

def displayPDF(file):
    try:
        base64_pdf = base64.b64encode(file.read()).decode('utf-8')
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

def main():
    st.title("Document Summarization App")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.summary = None  # Reset summary when a new file is uploaded

    if st.session_state.uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("Uploaded File")
                displayPDF(st.session_state.uploaded_file)

            with col2:
                st.info("Generating summary...")
                st.session_state.summary = llm_pipeline(st.session_state.uploaded_file)
                if st.session_state.summary and st.session_state.summary != "Failed to generate summary":
                    st.success("Summarization Complete")
                    st.success(st.session_state.summary)
                else:
                    st.error("Failed to generate summary. Please try again later.")

if __name__ == "__main__":
    main()