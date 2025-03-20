import streamlit as st
import os
import pandas as pd
import fitz  # PyMuPDF for PDF processing
import shutil  # For clearing temporary files
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from datetime import datetime
nltk.download('punkt')
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

# Check if punkt data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

# Text preprocessing using NLTK
def preprocess_text(text, language='english'):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'\W+', ' ', text)
    # Tokenize the text into individual words
    words = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words(language))
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words



#def preprocess_text(text, language='english'):
 #   try:
  #      nltk.data.find('tokenizers/punkt')
   # except LookupError:
    #    nltk.download('punkt')
    
    # Tokenize and clean text
    #words = word_tokenize(re.sub(r'\W+', ' ', text.lower()))
    #return words


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

# Remove headers and footers
def remove_headers_footers(text):
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if len(line.split()) > 5]  # Remove lines with less than 5 words
    return ' '.join(cleaned_lines)

# Extract dates from text
def extract_dates(text):
    date_pattern = r'\b(?:\d{1,2}[-/th|st|nd|rd\s]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[-/,.\s]*\d{1,2}[-/,\s]*\d{2,4}\b|\b\d{1,2}[-/,\s]*\d{1,2}[-/,\s]*\d{2,4}\b'
    dates = re.findall(date_pattern, text, re.IGNORECASE)
    return dates

# Analyze texts
def analyze_texts(pdf_texts, top_n, language='english'):
    all_text = " ".join([doc["text"] for doc in pdf_texts])
    filtered_words = preprocess_text(all_text, language)
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(top_n)
    return top_words, word_counts


from collections import Counter
from datetime import datetime

def analyze_texts_by_date(pdf_texts, top_n, language='english', period='yearly', target_word=None, target_month=None):
    date_word_counts = {}
    target_word_counts = {}
    
    for doc in pdf_texts:
        text = doc["text"]
        dates = extract_dates(text)
        filtered_words = preprocess_text(text, language)
        word_counts = Counter(filtered_words)
        
        for date in dates:
            try:
                date_obj = datetime.strptime(date, '%d %b %Y')
            except ValueError:
                try:
                    date_obj = datetime.strptime(date, '%d/%m/%Y')
                except ValueError:
                    continue

            # Filter by month if specified
            if target_month and date_obj.month != target_month:
                continue

            if period == 'yearly':
                date_key = date_obj.year
            elif period == 'Monthly':
                date_key = f"{date_obj.year}-{date_obj.month:02d}"
            elif period == 'quarterly':
                date_key = f"{date_obj.year}-Q{(date_obj.month - 1) // 3 + 1}"
            elif period == 'half-yearly':
                date_key = f"{date_obj.year}-H{(date_obj.month - 1) // 6 + 1}"
            elif period == '3-years':
                date_key = f"{date_obj.year // 3 * 3}-{date_obj.year // 3 * 3 + 2}"
            elif period == '5-years':
                date_key = f"{date_obj.year // 5 * 5}-{date_obj.year // 5 * 5 + 4}"
            else:
                date_key = date_obj.year

            if date_key not in date_word_counts:
                date_word_counts[date_key] = Counter()
            date_word_counts[date_key].update(word_counts)

            # Count the frequency of the target word
            if target_word:
                target_word_lower = target_word.lower()
                if date_key not in target_word_counts:
                    target_word_counts[date_key] = 0
                target_word_counts[date_key] += word_counts.get(target_word_lower, 0)

    top_words_by_date = {date: counts.most_common(top_n) for date, counts in date_word_counts.items()}
    return top_words_by_date, target_word_counts
    


    
# Topic modeling using NMF
def nmf_topic_modeling_with_sentences(texts, num_topics=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(texts)
    nmf = NMF(n_components=num_topics, random_state=42)
    nmf.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        sentences = []
        for text in texts:
            for sentence in sent_tokenize(text):
                if any(word in sentence for word in topic_words):
                    sentences.append(sentence)
                    if len(sentences) >= 2:  # Limit to 2 sentences per topic
                        break
            if len(sentences) >= 2:
                break
        topics.append(f"Topic {topic_idx + 1}: {' '.join(sentences)}")
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
        text = remove_headers_footers(text)
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
                top_words, word_counts = analyze_texts(pdf_texts, top_n)
                st.write("### Top Words Across Documents")
                st.table(pd.DataFrame(top_words, columns=["Word", "Frequency"]))
            else:
                st.warning("No documents uploaded or text extracted. Please upload valid PDF files.")

        # Top words in each document
        if st.button("Analyze Texts in Each Document"):
            for doc in pdf_texts:
                top_words, _ = analyze_texts([doc], top_n)
                st.write(f"### Top Words in {doc['filename']}")
                st.table(pd.DataFrame(top_words, columns=["Word", "Frequency"]))

        # Topic modeling and clustering
        tabs = st.tabs(["Topic Modeling", "Word Frequency", "Top Words by Date"])

        with tabs[0]:
            num_topics = st.slider("Select number of NMF topics", 2, 10, 3)
            nmf_topics = nmf_topic_modeling_with_sentences([doc["text"] for doc in pdf_texts], num_topics)
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
                        nmf_topics = nmf_topic_modeling_with_sentences(filtered_texts, num_topics=num_topics_nmf)
                        st.write(f"### NMF Topic Modeling Results for documents containing the word '{specific_word}':")
                        for topic in nmf_topics:
                            st.write(topic)
                    else:
                        st.warning(f"No documents contain the word '{specific_word}'.")
                else:
                    st.warning("Please enter a word to perform NMF.")
            else:
                st.warning("Please enter a word to analyze.")

        with tabs[2]:
            st.header("Top Words by Date")
            period = st.selectbox("Select period for date analysis", ["Monthly","yearly", "quarterly", "half-yearly", "3-years", "5-years"])
            target_word = st.text_input("Enter a specific word to analyze its frequency (optional)")

            if st.button("Analyze Texts by Date"):
                if pdf_texts:  # Ensure there are uploaded documents
                    top_words_by_date, target_word_counts = analyze_texts_by_date(pdf_texts, top_n, period=period, target_word=target_word)
                    st.write(f"### Top Words by {period.capitalize()}")
                    for date, top_words in top_words_by_date.items():
                        st.write(f"**{date}**")
                        st.table(pd.DataFrame(top_words, columns=["Word", "Frequency"]))

                    if target_word:
                        st.write(f"### Frequency of the word '{target_word}' by {period.capitalize()}")
                        target_word_df = pd.DataFrame(list(target_word_counts.items()), columns=["Date", "Frequency"])
                        st.table(target_word_df)
            else:
                st.warning("No documents uploaded or text extracted. Please upload valid PDF files.")
            
else:
    st.info("Please upload some PDF files.")
