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

# Ensure punkt tab is available
try:
    nltk.download('punkt_tab', download_dir=nltk_data_path)
except Exception as e:
    st.error(f"Error downloading punkt_tab resource: {e}")

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
    """
    Analyzes the combined text from multiple documents and returns the top N words with their frequencies.
    """
    # Combine all extracted text
    all_text = " ".join([doc["text"] for doc in pdf_texts])
    
    # Preprocess and filter text
    filtered_words = preprocess_text(all_text, language)
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Get the top N most common words
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
    
    tabs = st.tabs(["Upload & Analyze", "Specific Word Analysis", "Topic Modeling"])
    if pdf_texts:  
        with tabs[0]:
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


        with tabs[1]:
            st.header("Specific Word Frequency")
            
            # Input for specific word analysis
            specific_word = st.text_input("Enter a word to analyze its frequency:")

            # "Calculate Frequency" button with a unique key
            if st.button("Calculate Frequency", key="calculate_frequency"):
                if specific_word:
                    if 'top_words' in st.session_state and 'word_counts' in st.session_state:
                        if st.session_state.top_words and st.session_state.word_counts:  # Check if they're not empty
                            specific_word_count = st.session_state.word_counts.get(specific_word.lower(), 0)
                            
                            # Store results in session state for persistence
                            st.session_state["specific_word_result"] = {
                                "word": specific_word,
                                "count": specific_word_count,
                                "frequencies_per_doc": []
                            }

                            # Calculate frequency for each document
                            word_frequencies_per_doc = []
                            for doc in pdf_texts:
                                # Tokenize and preprocess document text
                                words = word_tokenize(re.sub(r'\W+', ' ', doc["text"].lower()))
                                doc_specific_word_count = Counter(words).get(specific_word.lower(), 0)
                                word_frequencies_per_doc.append({
                                    "Document": doc["filename"],
                                    "Frequency": doc_specific_word_count
                                })

                            # Store document frequencies in session state
                            st.session_state["specific_word_result"]["frequencies_per_doc"] = word_frequencies_per_doc

                        else:
                            st.warning("The session state variables are empty. Please run the analysis first.")
                    else:
                        st.info("Click 'Analyze Texts' first to perform the analysis.")
                else:
                    st.warning("Please enter a word to analyze.")

            # Display previously calculated frequency result if available, with a unique button key
            if "specific_word_result" in st.session_state and not st.button("Recalculate Frequency", key="recalculate_frequency"):
                result = st.session_state["specific_word_result"]
                st.write(f"The word **'{result['word']}'** appears **{result['count']}** times.")
                frequency_df = pd.DataFrame(result["frequencies_per_doc"])
                st.write(f"The Frequency of **'{result['word']}'** in Each Document:")
                st.table(frequency_df)

            # Slider for number of topics (moved outside button logic)
            num_topics_nmf = st.slider("Select the Number of Topics (NMF):", 2, 10, 3, key="num_topics_nmf_specific_word")

            # Add button for applying NMF based on the specific word, with a unique key
            if st.button("Apply NMF Based on Specific Word", key="apply_nmf_specific_word"):
                if specific_word:
                    # Filter texts based on the specific word
                    filtered_texts = [doc["text"] for doc in pdf_texts if specific_word.lower() in doc["text"].lower()]

                    if filtered_texts:
                        # Apply NMF to the filtered texts
                        nmf_topics = nmf_topic_modeling_on_specific_word(filtered_texts, num_topics=num_topics_nmf)
                        st.write(f"### NMF Topic Modeling Results for documents containing the word '{specific_word}':")
                        for topic in nmf_topics:
                            st.write(topic)
                    else:
                        st.warning(f"No documents contain the word '{specific_word}'.")
                else:
                    st.warning("Please enter a word to perform NMF.")

        with tabs[2]:
            st.header(" Topic Modeling")
            # Perform Topic Modeling with LDA
            num_topics_lda = st.slider("Select the Number of Topics (LDA):", 2, 10, 3, key="num_topics_lda")
            topics_lda = topic_modeling([doc["text"] for doc in pdf_texts], num_topics=num_topics_lda)
            st.write("### LDA Topic Modeling Results:")
            for topic in topics_lda:
                st.write(topic)

            # Perform Clustering for LDA
            num_clusters_lda = st.slider("Select the Number of Clusters (LDA):", 2, 10, 3, key="num_clusters_lda")
            clusters_lda = clustering(pdf_texts, num_clusters=num_clusters_lda)
            pdf_df["Cluster (LDA)"] = clusters_lda
            st.write("### Clustered Documents (LDA):")
            st.dataframe(pdf_df)

            # Perform Topic Modeling with NMF
            num_topics_nmf = st.slider("Select the Number of Topics (NMF):", 2, 10, 3, key="num_topics_nmf")
            nmf_topics = nmf_topic_modeling([doc["text"] for doc in pdf_texts], num_topics=num_topics_nmf)
            st.write("### NMF Topic Modeling Results:")
            for topic in nmf_topics:
                st.write(topic)

            # Perform Clustering for NMF
            num_clusters_nmf = st.slider("Select the Number of Clusters (NMF):", 2, 10, 3, key="num_clusters_nmf")
            clusters_nmf = clustering(pdf_texts, num_clusters=num_clusters_nmf)
            pdf_df["Cluster (NMF)"] = clusters_nmf
            st.write("### Clustered Documents (NMF):")
            st.dataframe(pdf_df)

else:
    st.info("Please upload some PDF files.")
