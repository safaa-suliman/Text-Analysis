import streamlit as st
import os
import pandas as pd
import fitz  # PyMuPDF for PDF processing
import shutil  # For clearing temporary files
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Custom path for nltk_data
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Ensure required NLTK resources are downloaded
try:
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")

# Text preprocessing using NLTK
def preprocess_text(text, language='english'):
    stop_words = set(stopwords.words(language))
    words = word_tokenize(re.sub(r'\W+', ' ', text.lower()))  # Tokenize and clean text
    return [word for word in words if word.isalnum() and word not in stop_words]

# Extract text from PDFs
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

# Analyze texts
def analyze_texts(pdf_texts, top_n, language='english'):
    all_text = " ".join([doc["text"] for doc in pdf_texts])
    filtered_words = preprocess_text(all_text, language)
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(top_n)
    return top_words, word_counts

# Topic Modeling using NMF
def nmf_topic_modeling(texts, num_topics=3):
    vectorizer = CountVectorizer(stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(texts)
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(doc_term_matrix)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    return topics

# Clear temporary files
def clear_temp_folder(folder="temp"):
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
        except Exception as e:
            print(f"Error removing {folder}: {e}")
    os.makedirs(folder, exist_ok=True)

# Streamlit App
st.set_page_config(page_title="Document Analysis Webpage", page_icon="📄", layout="wide")
st.title("Document Analysis with Persistent Results")

# File uploader
uploaded_files = st.file_uploader("Upload multiple PDF files", type="pdf", accept_multiple_files=True)

if "pdf_texts" not in st.session_state:
    st.session_state["pdf_texts"] = []
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = {}
if "nmf_topics" not in st.session_state:
    st.session_state["nmf_topics"] = []

if uploaded_files:
    clear_temp_folder()
    pdf_texts = []

    for uploaded_file in uploaded_files:
        pdf_path = os.path.join("temp", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        text = extract_text_from_pdf(pdf_path)
        if text.strip():
            pdf_texts.append({"filename": uploaded_file.name, "text": text})
        else:
            st.warning(f"File {uploaded_file.name} contains no readable text.")

    if pdf_texts:
        st.session_state["pdf_texts"] = pdf_texts
        st.write("### Extracted Data")
        st.dataframe(pd.DataFrame(pdf_texts))

# Ensure texts are loaded in session state
pdf_texts = st.session_state.get("pdf_texts", [])

if pdf_texts:
    top_n = st.slider("Select number of top words to display", 1, 20, 10)

    if st.button("Analyze Texts"):
        top_words, word_counts = analyze_texts(pdf_texts, top_n)
        st.session_state["analysis_results"]["top_words"] = top_words
        st.session_state["analysis_results"]["word_counts"] = word_counts

    if "top_words" in st.session_state["analysis_results"]:
        st.write("### Top Words Across Documents")
        st.table(pd.DataFrame(st.session_state["analysis_results"]["top_words"], columns=["Word", "Frequency"]))

    specific_word = st.text_input("Enter a word to analyze its frequency:")
    if st.button("Calculate Frequency"):
        combined_text = " ".join([doc["text"].lower() for doc in pdf_texts])
        all_words = word_tokenize(re.sub(r'\W+', ' ', combined_text))
        total_count = Counter(all_words).get(specific_word.lower(), 0)

        st.write(f"The word **'{specific_word}'** appears **{total_count}** times across all documents.")
        doc_frequencies = [{"Document": doc["filename"], "Frequency": Counter(word_tokenize(doc["text"].lower())).get(specific_word.lower(), 0)} for doc in pdf_texts]
        st.table(pd.DataFrame(doc_frequencies))

    num_topics_nmf = st.slider("Select the Number of Topics (NMF):", 2, 10, 3, key="num_topics_nmf_specific_word")
    if st.button("Apply NMF Based on Specific Word"):
        filtered_texts = [doc["text"] for doc in pdf_texts if specific_word.lower() in doc["text"].lower()]
        if filtered_texts:
            nmf_topics = nmf_topic_modeling(filtered_texts, num_topics=num_topics_nmf)
            st.session_state["nmf_topics"] = nmf_topics
        else:
            st.warning(f"No documents contain the word '{specific_word}'.")

    if st.session_state["nmf_topics"]:
        st.write("### NMF Topic Modeling Results")
        for topic in st.session_state["nmf_topics"]:
            st.write(topic)
else:
    st.info("Please upload some PDF files.")
