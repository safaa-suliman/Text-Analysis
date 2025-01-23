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
from nltk.tokenize import word_tokenize, sent_tokenize

# Custom path for nltk_data
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Ensure required NLTK resources are downloaded
try:
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")

# Check if punkt data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

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

# Extract dates from text
def extract_dates(text):
    date_pattern = r'\b(?:\d{1,2}[-/th|st|nd|rd\s]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[-/,.\s]*\d{1,2}[-/,\s]*\d{2,4}\b|\b\d{1,2}[-/,\s]*\d{1,2}[-/,\s]*\d{2,4}\b'
    dates = re.findall(date_pattern, text, re.IGNORECASE)
    return dates

# Analyze texts
def analyze_texts_by_date(pdf_texts, top_n, language='english'):
    date_word_counts = {}
    for doc in pdf_texts:
        text = doc["text"]
        dates = extract_dates(text)
        filtered_words = preprocess_text(text, language)
        word_counts = Counter(filtered_words)
        for date in dates:
            if date not in date_word_counts:
                date_word_counts[date] = Counter()
            date_word_counts[date].update(word_counts)
    
    top_words_by_date = {date: counts.most_common(top_n) for date, counts in date_word_counts.items()}
    return top_words_by_date

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

    if pdf_texts:
        pdf_df = pd.DataFrame(pdf_texts)
        st.write("### Extracted Data")
        st.dataframe(pdf_df)
        # Option to download the DataFrame as a CSV
        csv_data = pdf_df.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv_data, file_name="extracted_texts.csv", mime="text/csv")

        # Text analysis
        top_n = st.slider("Select number of top words to display", 1, 20, 10)
        if st.button("Analyze Texts"):
            if pdf_texts:  # Ensure there are uploaded documents
                top_words_by_date = analyze_texts_by_date(pdf_texts, top_n)
                st.write("### Top Words by Date")
                for date, top_words in top_words_by_date.items():
                    st.write(f"**{date}**")
                    st.table(pd.DataFrame(top_words, columns=["Word", "Frequency"]))
            else:
                st.warning("No documents uploaded or text extracted. Please upload valid PDF files.")

        # Topic modeling and clustering
        tabs = st.tabs(["Topic Modeling", "Word Frequency"])

        with tabs[0]:
            num_topics = st.slider("Select number of LDA topics", 2, 10, 3)
            lda_topics = topic_modeling([doc["text"] for doc in pdf_texts], num_topics)
            st.write("### LDA Topics")
            for topic in lda_topics:
                st.write(topic)

            num_topics = st.slider("Select number of NMF topics", 2, 10, 3)
            nmf_topics = nmf_topic_modeling([doc["text"] for doc in pdf_texts], num_topics)
            st.write("### NMF Topics")
            for topic in nmf_topics:
                st.write(topic)

            num_clusters = st.slider("Select number of clusters", 2, 10, 3)
            clusters = clustering(pdf_texts, num_clusters)
            pdf_df["Cluster"] = clusters
            st.write("### Clusters")
            st.dataframe(pdf_df)

        with tabs[1]:
            # Streamlit App - Add Specific Word Frequency Analysis
            st.header("Specific Word Frequency Analysis")

            # Input for specific word analysis
            specific_word = st.text_input("Enter a word to analyze its frequency:")

            if st.button("Calculate Frequency"):
                if specific_word:
                    # Calculate frequency across all documents
                    combined_text = " ".join([doc["text"].lower() for doc in pdf_texts])
                    all_words = word_tokenize(re.sub(r'\W+', ' ', combined_text))
                    total_count = Counter(all_words).get(specific_word.lower(), 0)

                    st.write(f"The word **'{specific_word}'** appears **{total_count}** times across all documents.")
                    if total_count == 0:
                        st.write("")
                    else:
                        # Calculate frequency per document
                        doc_frequencies = []
                        for doc in pdf_texts:
                            words = word_tokenize(re.sub(r'\W+', ' ', doc["text"].lower()))
                            doc_count = Counter(words).get(specific_word.lower(), 0)
                            doc_frequencies.append({"Document": doc["filename"], "Frequency": doc_count})

                        # Display results
                        st.write("### Frequency in Each Document:")
                        st.table(pd.DataFrame(doc_frequencies))

            # Slider for number of topics (moved outside button logic)
            num_topics_nmf = st.slider("Select the Number of Topics (NMF):", 2, 10, 3, key="num_topics_nmf_specific_word")

            # Add button for applying NMF based on the specific word, with a unique key
            if st.button("Apply NMF Based on Specific Word", key="apply_nmf_specific_word"):
                if specific_word:
                    # Filter texts based on the specific word
                    filtered_texts = [doc["text"] for doc in pdf_texts if specific_word.lower() in doc["text"].lower()]

                    if filtered_texts:
                        # Apply NMF to the filtered texts
                        nmf_topics = nmf_topic_modeling(filtered_texts, num_topics=num_topics_nmf)
                        st.write(f"### NMF Topic Modeling Results for documents containing the word '{specific_word}':")
                        for topic in nmf_topics:
                            st.write(topic)
                    else:
                        st.warning(f"No documents contain the word '{specific_word}'.")
                else:
                    st.warning("Please enter a word to perform NMF.")
            else:
                st.warning("Please enter a word to analyze.")
else:
    st.info("Please upload some PDF files.")
