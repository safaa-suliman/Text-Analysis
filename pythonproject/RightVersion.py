import streamlit as st
import os
import pandas as pd
import fitz  # PyMuPDF for PDF processing
import shutil  # For clearing temporary files
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

words = word_tokenize(all_text.lower(), language='english')

import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
# Initialization block for NLTK resources

nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Download required NLTK data
try:
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")


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


# Function to preprocess and analyze text
def preprocess_text(text, language='english'):
    """
    Tokenizes and filters the input text, removing stopwords and non-alphanumeric characters.
    """
    stop_words = set(stopwords.words(language))
    words = word_tokenize(re.sub(r'\W+', ' ', text.lower()))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return filtered_words

def analyze_texts(pdf_texts, top_n, language='english'):
    """
    Analyzes the combined text from multiple documents and returns the top N words with their frequencies.
    """
    # Initialize NLTK resources (if needed)
    nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
    nltk.data.path.append(nltk_data_path)

    try:
        nltk.download('punkt', download_dir=nltk_data_path)
        nltk.download('stopwords', download_dir=nltk_data_path)
    except Exception as e:
        st.error(f"Error downloading NLTK resources: {e}")
        return [], Counter()

    # Combine all extracted text
    all_text = " ".join([doc["text"] for doc in pdf_texts])
    
    # Preprocess and filter text
    stop_words = set(nltk.corpus.stopwords.words(language))
    words = word_tokenize(re.sub(r'\W+', ' ', all_text.lower()), language=language)
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    # Get the top N most common words
    top_words = word_counts.most_common(top_n)
    
    return top_words, word_counts




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
        if st.button("Analyze Texts"):
            if pdf_texts:  # Ensure there are uploaded documents
                top_words, word_counts = analyze_texts(pdf_texts, top_n)
                st.write("### Top Words Across Documents")
                st.table(pd.DataFrame(top_words, columns=["Word", "Frequency"]))
            else:
                st.warning("No documents uploaded or text extracted. Please upload valid PDF files.")


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
