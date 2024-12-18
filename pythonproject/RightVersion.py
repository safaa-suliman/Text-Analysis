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
st.subheader("Hi, This is a web for analyzing documents :wave:")
st.title("A Data Analyst From Sudan")
st.write("I am passionate about Data Science")
st.write("[My GitHub >](https://github.com/safa-suliman)")

# Initialize session_state for results persistence
if "results" not in st.session_state:
    st.session_state.results = {
        "top_words": None,
        "lda_topics": None,
        "nmf_topics": None,
        "clusters": None,
        "word_frequency": None,
        "nmf_specific_word": None,
    }

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
        csv_data = pdf_df.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv_data, file_name="extracted_texts.csv", mime="text/csv")

        # Tabs
        tabs = st.tabs(["Topic Modeling", "Word Frequency"])

        with tabs[0]:
            # LDA Topics
            num_topics_lda = st.slider("Select number of LDA topics", 2, 10, 3)
            if st.button("Generate LDA Topics"):
                st.session_state.results["lda_topics"] = topic_modeling([doc["text"] for doc in pdf_texts], num_topics_lda)
            if st.session_state.results["lda_topics"]:
                st.write("### LDA Topics")
                for topic in st.session_state.results["lda_topics"]:
                    st.write(topic)

            # NMF Topics
            num_topics_nmf = st.slider("Select number of NMF topics", 2, 10, 3)
            if st.button("Generate NMF Topics"):
                st.session_state.results["nmf_topics"] = nmf_topic_modeling([doc["text"] for doc in pdf_texts], num_topics_nmf)
            if st.session_state.results["nmf_topics"]:
                st.write("### NMF Topics")
                for topic in st.session_state.results["nmf_topics"]:
                    st.write(topic)

            # Clustering
            num_clusters = st.slider("Select number of clusters", 2, 10, 3)
            if st.button("Generate Clusters"):
                st.session_state.results["clusters"] = clustering(pdf_texts, num_clusters)
                pdf_df["Cluster"] = st.session_state.results["clusters"]
            if st.session_state.results["clusters"] is not None:
                st.write("### Clusters")
                st.dataframe(pdf_df)

        with tabs[1]:  # Word Frequency Tab
            st.header("Text Analysis and Word Frequency")
        
            # Analyze Text Button
            top_n = st.slider("Select number of top words to display", 1, 20, 10)
            if st.button("Analyze Texts"):
                if pdf_texts:  # Ensure there are uploaded documents
                    top_words, word_counts = analyze_texts(pdf_texts, top_n)
                    st.session_state.results["top_words"] = top_words
                else:
                    st.warning("No documents uploaded or text extracted. Please upload valid PDF files.")
            
            # Display Top Words
            if st.session_state.results["top_words"]:
                st.write("### Top Words Across Documents")
                st.table(pd.DataFrame(st.session_state.results["top_words"], columns=["Word", "Frequency"]))
        
            # Word Frequency Analysis for a Specific Word
            specific_word = st.text_input("Enter a word to analyze its frequency:")
            if st.button("Calculate Word Frequency"):
                combined_text = " ".join([doc["text"].lower() for doc in pdf_texts])
                all_words = word_tokenize(re.sub(r'\W+', ' ', combined_text))
                total_count = Counter(all_words).get(specific_word.lower(), 0)
                doc_frequencies = [
                    {"Document": doc["filename"], "Frequency": Counter(word_tokenize(re.sub(r'\W+', ' ', doc["text"].lower()))).get(specific_word.lower(), 0)}
                    for doc in pdf_texts
                ]
                st.session_state.results["word_frequency"] = {"total": total_count, "per_doc": doc_frequencies}
        
            if st.session_state.results["word_frequency"]:
                st.write(f"### Word Frequency: '{specific_word}'")
                st.write(f"Total occurrences across documents: {st.session_state.results['word_frequency']['total']}")
                st.table(pd.DataFrame(st.session_state.results["word_frequency"]["per_doc"]))

else:
    st.info("Please upload some PDF files.")
