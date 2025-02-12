import fitz  # PyMuPDF for PDF processing
import streamlit as st

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error processing {pdf_path}: {e}")
        return ""

def remove_headers_footers(text):
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if len(line.split()) > 5]  # Remove lines with less than 5 words
    return ' '.join(cleaned_lines)
