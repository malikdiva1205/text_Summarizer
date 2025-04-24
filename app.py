import streamlit as st
import spacy
import math
from nltk.stem import WordNetLemmatizer
import nltk
import PyPDF2
from bs4 import BeautifulSoup
import requests

# NLP setup
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')

nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

# ---------- Utility Functions ----------
def clean_and_lemmatize(doc):
    stop_words = nlp.Defaults.stop_words
    return [token.lemma_.lower() for token in doc if token.is_alpha and token.lemma_.lower() not in stop_words]

def frequency_matrix(sentences):
    freq_matrix = {}
    for sent in sentences:
        freq_table = {}
        words = clean_and_lemmatize(sent)
        for word in words:
            freq_table[word] = freq_table.get(word, 0) + 1
        freq_matrix[sent[:15]] = freq_table
    return freq_matrix

def tf_matrix(freq_matrix):
    tf_matrix = {}
    for sent, freq_table in freq_matrix.items():
        total_words = sum(freq_table.values())
        tf_table = {word: count / total_words for word, count in freq_table.items()}
        tf_matrix[sent] = tf_table
    return tf_matrix

def sentences_per_words(freq_matrix):
    sent_per_words = {}
    for f_table in freq_matrix.values():
        for word in f_table:
            sent_per_words[word] = sent_per_words.get(word, 0) + 1
    return sent_per_words

def idf_matrix(freq_matrix, sent_per_words, total_sentences):
    idf_matrix = {}
    for sent, f_table in freq_matrix.items():
        idf_table = {word: math.log10(total_sentences / float(sent_per_words[word])) for word in f_table}
        idf_matrix[sent] = idf_table
    return idf_matrix

def tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    for sent in tf_matrix:
        tf_idf_table = {word: tf_matrix[sent][word] * idf_matrix[sent][word] for word in tf_matrix[sent]}
        tf_idf_matrix[sent] = tf_idf_table
    return tf_idf_matrix

def score_sentences(tf_idf_matrix):
    sentence_score = {}
    for sent, f_table in tf_idf_matrix.items():
        if len(f_table) == 0:
            continue
        sentence_score[sent] = sum(f_table.values()) / len(f_table)
    return sentence_score

def named_entity_score(sentence):
    return len([ent for ent in sentence.ents if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]])

def auto_compression_percent(text_len):
    if text_len < 100:
        return 90
    elif text_len < 500:
        return 80
    elif text_len < 1500:
        return 70
    elif text_len < 3000:
        return 60
    else:
        return 50

def create_summary(sentences, sentence_score, compression_percent, entity_boost=False):
    sorted_scores = sorted(sentence_score.items(), key=lambda x: x[1], reverse=True)
    top_count = max(1, math.ceil(len(sorted_scores) * (compression_percent / 100)))
    top_sentences = dict(sorted_scores[:top_count])

    if entity_boost:
        for sentence in sentences:
            if sentence[:15] in top_sentences:
                top_sentences[sentence[:15]] += named_entity_score(sentence)

    final_sentences = sorted(top_sentences.items(), key=lambda x: x[1], reverse=True)
    summary = " ".join([s.text for s in sentences if s[:15] in dict(final_sentences)])
    return summary

# ---------- Input Handlers ----------
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    return "".join(page.extract_text() for page in reader.pages if page.extract_text())

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragraphs])
    except:
        return "Failed to retrieve URL content."

# ---------- Streamlit UI ----------
st.set_page_config(page_title="🧠 Smart TF-IDF Summarizer", layout="wide", initial_sidebar_state="expanded")

# Background styling
st.markdown("""
    <style>
        body {
            background-color: #111;
            color: #eee;
        }
        .stApp {
            background-color: #111;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Smart TF-IDF Summarizer")
st.markdown("Automatically summarize content from **Text**, **PDF**, **.txt files**, or **URLs** with intelligent compression and entity-awareness.")

with st.sidebar:
    st.header("📥 Input Options")
    input_option = st.radio("Choose input type:", ["Manual Text", "Upload PDF", "Upload .txt File", "Enter URL"])
    entity_boost = st.checkbox("⚡ Boost with named entities", value=True)

input_text = ""

if input_option == "Manual Text":
    input_text = st.text_area("Enter your text here:", height=250)

elif input_option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        input_text = extract_text_from_pdf(uploaded_file)

elif input_option == "Upload .txt File":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file:
        input_text = uploaded_file.read().decode('utf-8')

elif input_option == "Enter URL":
    url = st.text_input("Paste the URL here:")
    if url:
        input_text = extract_text_from_url(url)

# Run the summarizer
if st.button("🔍 Generate Summary") and input_text.strip():
    doc = nlp(input_text)
    sentences = list(doc.sents)
    total_sentences = len(sentences)

    compression_percent = auto_compression_percent(len(input_text.split()))
    
    freq_matrix = frequency_matrix(sentences)
    tf_mat = tf_matrix(freq_matrix)
    sent_per_word = sentences_per_words(freq_matrix)
    idf_mat = idf_matrix(freq_matrix, sent_per_word, total_sentences)
    tf_idf_mat = tf_idf_matrix(tf_mat, idf_mat)
    sentence_scores = score_sentences(tf_idf_mat)

    summary = create_summary(sentences, sentence_scores, compression_percent, entity_boost)

    st.markdown("### 📝 Summary:")
    st.success(summary)

    st.markdown("### 📊 Word Count Comparison:")
    st.write(f"**Original Text:** {len(input_text.split())} words")
    st.write(f"**Summary:** {len(summary.split())} words")

    st.download_button("📥 Download Summary", summary, file_name="summary.txt", mime="text/plain")
elif input_option != "Manual Text":
    st.info("Please upload a file or enter a valid URL to generate the summary.")
