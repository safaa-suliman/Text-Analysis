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
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF  # Import NMF
nltk.data.path.append('./nltk_data')  # Specify the path to pre-downloaded data
try:
    nltk.download('punkt', download_dir='./nltk_data')
    nltk.download('stopwords', download_dir='./nltk_data')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

nltk.data.path.append('./nltk_data')
# Set page configuration
st.set_page_config(page_title="Document Analysis Webpage", page_icon="📄", layout="wide")
st.subheader("Hi, This is a web for analyzing documents :wave:")
st.title("A Data Analyst From Sudan")
st.write("I am passionate about Data Science")
st.write("[My GitHub >](https://github.com/safa-suliman)")

# Define the function to extract text from a PDF
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

# Topic Modeling with NMF
def nmf_topic_modeling(texts, num_topics=3):
    vectorizer = CountVectorizer(stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(texts)
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words_nmf = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words_nmf)}")
    return topics



def clear_temp_folder(folder="temp"):
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
        except Exception as e:
            print(f"Error removing {folder}: {e}")
    os.makedirs(folder, exist_ok=True)

# Function to perform text analysis
def analyze_texts(pdf_texts, top_n):
    # Combine all text for global word count
    all_text = " ".join([doc["text"] for doc in pdf_texts])

    # Preprocess and tokenize
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(re.sub(r'\W+', ' ', all_text.lower()))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    # Count word frequencies
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(top_n)

    return top_words, word_counts

# Topic Modeling with LDA (scikit-learn)
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

# Clustering using KMeans
def clustering(pdf_texts, num_clusters=3):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([doc["text"] for doc in pdf_texts])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(tfidf_matrix)
    return kmeans.labels_


# Function to filter texts containing a specific word
def filter_texts_by_word(texts, word):
    return [text for text in texts if word.lower() in text.lower()]

# Function for NMF Topic Modeling
def nmf_topic_modeling_on_specific_word(texts, num_topics=3):
    vectorizer = CountVectorizer(stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(texts)
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words_nmf = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words_nmf)}")
    return topics

# Streamlit App
st.title("📂 Document Analysis - Enhanced Features")

# File uploader for multiple PDFs
uploaded_files = st.file_uploader("Upload multiple PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Clear old temporary files
    clear_temp_folder()

    pdf_texts = []

    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary path
        pdf_path = os.path.join("temp", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Check file extension and process PDFs
        if uploaded_file.name.endswith(".pdf"):
            # Extract text from the PDF
            text = extract_text_from_pdf(pdf_path)
            if text.strip():  # Check if extracted text is not empty
                pdf_texts.append({"filename": uploaded_file.name, "text": text})
            else:
                st.warning(f"Seems your file {uploaded_file.name}. is pdf with image data")
        else:
            st.warning(f"Skipping non-PDF file: {uploaded_file.name}")

    # Check if pdf_texts contains data
    if not pdf_texts:
        st.error("No text could be extracted from the uploaded PDFs. Please check the files.")
    else:
        # Convert to a DataFrame
        pdf_df = pd.DataFrame(pdf_texts)
        # Define your tabs
        tabs = st.tabs(["Upload & Analyze", "Specific Word Analysis", "NMF Topic Modeling"])
        with tabs[0]:
            st.header("Upload & Analyze Documents")
            # Display the DataFrame
            st.write("### Extracted Data:")
            st.dataframe(pdf_df)

            # Option to download the DataFrame as a CSV
            csv_data = pdf_df.to_csv(index=False)
            st.download_button(label="Download CSV", data=csv_data, file_name="extracted_texts.csv", mime="text/csv")

            # Input for number of top words
            top_n = st.slider("Select the number of top words to display", min_value=1, max_value=20, value=10)

            if st.button("Analyze Texts"):
                # Perform analysis
                top_words, word_counts = analyze_texts(pdf_texts, top_n)
                # Store results in session state
                st.session_state.top_words = top_words
                st.session_state.word_counts = word_counts

            # Display top words in a table if they exist in session state
            if 'top_words' in st.session_state:
                st.write("### Top Words Across All Documents:")
                st.table(pd.DataFrame(st.session_state.top_words, columns=["Word", "Frequency"]))
        
    


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
    st.info("Please upload multiple PDF files.")
