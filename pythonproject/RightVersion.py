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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans

# Ensure NLTK resources are available
nltk_data_dir = "./nltk_data"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)

nltk.data.path.append(nltk_data_dir)
resources = ['punkt', 'stopwords']
for resource in resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

# Utility to get stopwords for a specific language
def get_stopwords(language='english'):
    try:
        return set(stopwords.words(language))
    except Exception:
        st.warning(f"Stopwords for language '{language}' are unavailable.")
        return set()

# Text preprocessing using NLTK
def preprocess_text(text, language='english'):
    stop_words = get_stopwords(language)
    words = word_tokenize(re.sub(r'\W+', ' ', text.lower()))
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

# Clear temporary files
def clear_temp_folder(folder="temp"):
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
        except Exception as e:
            print(f"Error removing {folder}: {e}")
    os.makedirs(folder, exist_ok=True)

# Analyze texts
def analyze_texts(pdf_texts, top_n, language='english'):
    all_text = " ".join([doc["text"] for doc in pdf_texts])
    filtered_words = preprocess_text(all_text, language)
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(top_n)
    return top_words, word_counts

# Topic Modeling using LDA
def topic_modeling(texts, num_topics=3):
    vectorizer = CountVectorizer(stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(texts)
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(doc_term_matrix)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    return topics

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

# Clustering using KMeans
def clustering(pdf_texts, num_clusters=3):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([doc["text"] for doc in pdf_texts])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(tfidf_matrix)
    return kmeans.labels_

# Streamlit App
st.set_page_config(page_title="Document Analysis Webpage", page_icon="📄", layout="wide")
st.subheader("Hi, This is a web for analyzing documents :wave:")
st.title("A Data Analyst From Sudan")
st.write("I am passionate about Data Science")
st.write("[My GitHub >](https://github.com/safa-suliman)")

# File uploader
uploaded_files = st.file_uploader("Upload multiple PDF files", type="pdf", accept_multiple_files=True)

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
        pdf_df = pd.DataFrame(pdf_texts)
        st.write("### Extracted Data")
        st.dataframe(pdf_df)

        # Text analysis
        top_n = st.slider("Select number of top words to display", 1, 20, 10)
        top_words, word_counts = analyze_texts(pdf_texts, top_n)
        st.write("### Top Words Across Documents")
        st.table(pd.DataFrame(top_words, columns=["Word", "Frequency"]))

        # Topic modeling and clustering
        tabs = st.tabs(["LDA Topic Modeling", "NMF Topic Modeling", "Clustering"])
        with tabs[0]:
            num_topics = st.slider("Select number of LDA topics", 2, 10, 3)
            lda_topics = topic_modeling([doc["text"] for doc in pdf_texts], num_topics)
            st.write("### LDA Topics")
            for topic in lda_topics:
                st.write(topic)

        with tabs[1]:
            num_topics = st.slider("Select number of NMF topics", 2, 10, 3)
            nmf_topics = nmf_topic_modeling([doc["text"] for doc in pdf_texts], num_topics)
            st.write("### NMF Topics")
            for topic in nmf_topics:
                st.write(topic)

        with tabs[2]:
            num_clusters = st.slider("Select number of clusters", 2, 10, 3)
            clusters = clustering(pdf_texts, num_clusters)
            pdf_df["Cluster"] = clusters
            st.write("### Clusters")
            st.dataframe(pdf_df)

else:
    st.info("Please upload some PDF files.")
