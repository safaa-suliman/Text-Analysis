# Imports
import streamlit as st
import os
import pandas as pd
import fitz  # PyMuPDF for PDF processing
import shutil  # For clearing temporary files
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from transformers import pipeline

# Download NLTK data
nltk.data.path.append('./nltk_data')  # Specify the path to pre-downloaded data
try:
    nltk.download('punkt', download_dir='./nltk_data')
    nltk.download('stopwords', download_dir='./nltk_data')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Initialize BERT summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Streamlit Config
st.set_page_config(page_title="Document Analysis Webpage", page_icon="📄", layout="wide")
st.subheader("Hi, This is a web for analyzing documents :wave:")
st.title("A Data Analyst From Sudan")
st.write("I am passionate about Data Science")
st.write("[My GitHub >](https://github.com/safa-suliman)")

# Define function to extract text from PDFs
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

# Summarize topics
def summarize_topic_words(topic_words):
    combined_text = " ".join(topic_words)
    summary = summarizer(combined_text, max_length=20, min_length=10, do_sample=False)
    return summary[0]["summary_text"]

# NMF Topic Modeling
def nmf_topic_modeling_with_summaries(texts, num_topics=3):
    vectorizer = CountVectorizer(stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(texts)
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(doc_term_matrix)
    feature_names = vectorizer.get_feature_names_out()

    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        summary = summarize_topic_words(top_words)
        topics.append(f"Topic {topic_idx + 1}: {summary}")
    return topics

# LDA Topic Modeling
def lda_topic_modeling_with_summaries(texts, num_topics=3):
    vectorizer = CountVectorizer(stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(texts)
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(doc_term_matrix)
    feature_names = vectorizer.get_feature_names_out()

    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        summary = summarize_topic_words(top_words)
        topics.append(f"Topic {topic_idx + 1}: {summary}")
    return topics

# Clear temporary files
def clear_temp_folder(folder="temp"):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# Streamlit App
st.title("📂 Document Analysis - Enhanced Features")
uploaded_files = st.file_uploader("Upload multiple PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    clear_temp_folder()
    pdf_texts = []

    for uploaded_file in uploaded_files:
        pdf_path = os.path.join("temp", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        if uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(pdf_path)
            if text.strip():
                pdf_texts.append({"filename": uploaded_file.name, "text": text})
            else:
                st.warning(f"{uploaded_file.name} seems to be a scanned PDF or image-only PDF.")
        else:
            st.warning(f"Skipping non-PDF file: {uploaded_file.name}")

    if not pdf_texts:
        st.error("No text could be extracted from the uploaded PDFs.")
    else:
        pdf_df = pd.DataFrame(pdf_texts)
        st.write("### Extracted Data:")
        st.dataframe(pdf_df)

        # Topic Modeling Tabs
        tabs = st.tabs(["NMF Topic Modeling", "LDA Topic Modeling"])
        
        with tabs[0]:
            st.header("NMF Topic Modeling")
            num_topics_nmf = st.slider("Select Number of Topics (NMF):", 2, 10, 3, key="num_topics_nmf")
            nmf_topics = nmf_topic_modeling_with_summaries([doc["text"] for doc in pdf_texts], num_topics=num_topics_nmf)
            st.write("### NMF Topics:")
            for topic in nmf_topics:
                st.write(topic)

        with tabs[1]:
            st.header("LDA Topic Modeling")
            num_topics_lda = st.slider("Select Number of Topics (LDA):", 2, 10, 3, key="num_topics_lda")
            lda_topics = lda_topic_modeling_with_summaries([doc["text"] for doc in pdf_texts], num_topics=num_topics_lda)
            st.write("### LDA Topics:")
            for topic in lda_topics:
                st.write(topic)
else:
    st.info("Please upload multiple PDF files.")
