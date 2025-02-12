import streamlit as st
import os
import pandas as pd
from utils.pdf_processing import extract_text_from_pdf, remove_headers_footers
from utils.text_processing import preprocess_text
from utils.analysis import analyze_texts, analyze_texts_by_date, nmf_topic_modeling_with_sentences
from utils.clustering import clustering
import shutil

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
            period = st.selectbox("Select period for date analysis", ["yearly", "quarterly", "half-yearly", "3-years", "5-years"])
            if st.button("Analyze Texts by Date"):
                if pdf_texts:  # Ensure there are uploaded documents
                    top_words_by_date = analyze_texts_by_date(pdf_texts, top_n, period=period)
                    st.write(f"### Top Words by {period.capitalize()}")
                    for date, top_words in top_words_by_date.items():
                        st.write(f"**{date}**")
                        st.table(pd.DataFrame(top_words, columns=["Word", "Frequency"]))
                else:
                    st.warning("No documents uploaded or text extracted. Please upload valid PDF files.")
else:
    st.info("Please upload some PDF files.")
