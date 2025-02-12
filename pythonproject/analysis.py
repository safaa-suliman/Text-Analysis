import re
from collections import Counter
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.tokenize import sent_tokenize
from utils.text_processing import preprocess_text

def extract_dates(text):
    date_pattern = r'\b(?:\d{1,2}[-/th|st|nd|rd\s]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[-/,.\s]*\d{1,2}[-/,\s]*\d{2,4}\b|\b\d{1,2}[-/,\s]*\d{1,2}[-/,\s]*\d{2,4}\b'
    dates = re.findall(date_pattern, text, re.IGNORECASE)
    return dates

def analyze_texts(pdf_texts, top_n, language='english'):
    all_text = " ".join([doc["text"] for doc in pdf_texts])
    filtered_words = preprocess_text(all_text, language)
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(top_n)
    return top_words, word_counts

def analyze_texts_by_date(pdf_texts, top_n, language='english', period='yearly'):
    date_word_counts = {}
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

            if period == 'yearly':
                date_key = date_obj.year
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
    
    top_words_by_date = {date: counts.most_common(top_n) for date, counts in date_word_counts.items()}
    return top_words_by_date

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
