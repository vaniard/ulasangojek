# ============================================
# KONFIGURASI STREAMLIT - HARUS DI AWAL
# ============================================
import streamlit as st

# Set page config harus menjadi perintah Streamlit pertama
st.set_page_config(page_title="Analisis Sentimen Ulasan Gojek", layout="wide", initial_sidebar_state="collapsed")

# ============================================
# IMPORT MODULES (setelah set_page_config)
# ============================================
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import sys
import subprocess
import requests
from io import StringIO
import csv
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Download nltk data - dilakukan sekali saat deploy
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

# Custom CSS untuk memperbagus tampilan
st.markdown("""
<style>
    /* Memperkecil padding dan margin global */
    .main > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Header styling */
    h1 {
        font-size: 2rem !important;
        margin-bottom: 1rem !important;
        color: #1f77b4 !important;
        border-bottom: 3px solid #1f77b4;
        display: inline-block;
        padding-bottom: 0.3rem;
    }
    
    h2, h3 {
        font-size: 1.3rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Card-like containers */
    .stDataFrame, .stMarkdown, .stPlotlyChart {
        background: white;
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
    }
    
    /* Metric cards styling */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 0.5rem;
        color: white;
    }
    
    .stMetric label {
        color: white !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.4rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Download button specific */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* Success message */
    .stAlert[data-baseweb="notification"] {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 0.2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.3rem 1rem;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Spacing reduction */
    .element-container {
        margin-bottom: 0.3rem !important;
    }
    
    /* DataFrame styling */
    .dataframe {
        font-size: 0.85rem !important;
    }
    
    .dataframe th {
        background-color: #f8f9fa !important;
        font-weight: 600 !important;
    }
    
    /* Caption styling */
    .caption {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: -0.5rem;
    }
    
    /* Custom card classes */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# TITLE (setelah set_page_config)
# ============================================
st.title("📊 Analisis Sentimen Ulasan Gojek")
st.markdown("<p class='caption'>Dashboard analisis sentimen otomatis dari ulasan pengguna aplikasi Gojek</p>", unsafe_allow_html=True)

# ============================================
# FUNGSI LOAD DATA
# ============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ulasan_gojek.csv')
        df_clean = pd.read_csv('clean_data_ulasan.csv')
        return df, df_clean
    except FileNotFoundError:
        st.error("File dataset tidak ditemukan. Pastikan file 'ulasan_gojek.csv' dan 'clean_data_ulasan.csv' berada di direktori yang sama.")
        sample_data = {
            'Review': ['Aplikasi bagus sekali', 'Lambat banget', 'Mantap', 'Kurang suka', 'Pelayanan oke'],
            'Rating': [5, 1, 5, 2, 4]
        }
        df = pd.DataFrame(sample_data)
        df_clean = df.copy()
        df_clean['text_final'] = df_clean['Review'].str.lower()
        return df, df_clean

@st.cache_data
def load_lexicon():
    pos = {}
    neg = {}
    try:
        resp_pos = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv', timeout=10)
        resp_neg = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv', timeout=10)
        
        if resp_pos.status_code == 200:
            reader = csv.reader(StringIO(resp_pos.text), delimiter=',')
            for row in reader:
                if row and len(row) > 1: pos[row[0]] = int(row[1])
                elif row: pos[row[0]] = 1
        
        if resp_neg.status_code == 200:
            reader = csv.reader(StringIO(resp_neg.text), delimiter=',')
            for row in reader:
                if row and len(row) > 1: neg[row[0]] = int(row[1])
                elif row: neg[row[0]] = -1
    except:
        pos = {'bagus': 1, 'mantap': 1, 'oke': 1}
        neg = {'buruk': -1, 'jelek': -1, 'lambat': -1}
    
    return pos, neg

def sentiment_analysis_lexicon_indonesia(text):
    score = 0
    if not text or pd.isna(text):
        return 0, 'neutral'
    words = str(text).lower().split()
    for word in words:
        if word in pos_dict: score += pos_dict[word]
        elif word in neg_dict: score += neg_dict[word]
    
    if score > 0: return score, 'positive'
    elif score < 0: return score, 'negative'
    else: return score, 'neutral'

# ============================================
# MAIN PROGRAM
# ============================================
df_raw, df_clean = load_data()
pos_dict, neg_dict = load_lexicon()

# 1. Tampilan Data Awal
st.header("1️⃣ Data Awal Ulasan")
st.dataframe(df_raw.head(10), use_container_width=True)

# 2. Text Preprocessing
st.header("2️⃣ Proses Preprocessing Teks")
if 'text_final' not in df_clean.columns:
    df_clean['text_final'] = df_clean['Review'].str.lower()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Hasil Preprocessing")
    st.dataframe(df_clean[['Review', 'text_final']].head(5), use_container_width=True)
with col2:
    st.subheader("Informasi Data")
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{df_clean.shape[0]}</div><div class="metric-label">Jumlah Data</div></div>""", unsafe_allow_html=True)

# 3. Pelabelan Sentimen
st.header("3️⃣ Pelabelan Sentimen (Lexicon-based)")
if 'polarity' not in df_clean.columns:
    results = df_clean['text_final'].apply(sentiment_analysis_lexicon_indonesia)
    df_clean['polarity_score'] = [r[0] for r in results]
    df_clean['polarity'] = [r[1] for r in results]

sentiment_counts = df_clean['polarity'].value_counts()
col1, col2, col3 = st.columns(3)
col1.metric("😊 Positif", sentiment_counts.get('positive', 0))
col2.metric("😞 Negatif", sentiment_counts.get('negative', 0))
col3.metric("😐 Netral", sentiment_counts.get('neutral', 0))

# 4. Distribusi Sentimen
st.header("4️⃣ Distribusi Sentimen")
col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    df_clean['polarity'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax1, colors=['#2ecc71', '#e74c3c', '#95a5a6'])
    st.pyplot(fig1)
with col2:
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_clean, x='polarity', palette=['#2ecc71', '#e74c3c', '#95a5a6'], ax=ax2)
    st.pyplot(fig2)

# 5. WordCloud
st.header("5️⃣ Word Cloud Berdasarkan Sentimen")
col1, col2 = st.columns(2)
with col1:
    st.subheader("☀️ Positif")
    text = ' '.join(df_clean[df_clean['polarity'] == 'positive']['text_final'].dropna())
    if text:
        wc = WordCloud(background_color='white', colormap='Greens').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis('off')
        st.pyplot(fig)
with col2:
    st.subheader("🌧️ Negatif")
    text = ' '.join(df_clean[df_clean['polarity'] == 'negative']['text_final'].dropna())
    if text:
        wc = WordCloud(background_color='white', colormap='Reds').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis('off')
        st.pyplot(fig)

# 6. Distribusi Panjang Teks
st.header("6️⃣ Distribusi Panjang Teks")
fig, ax = plt.subplots()
df_clean['text_final'].str.split().str.len().hist(bins=20, ax=ax, color='skyblue')
st.pyplot(fig)

# 7. TF-IDF
st.header("7️⃣ Top 20 Kata berdasarkan TF-IDF")
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(max_features=20)
tfidf_matrix = vec.fit_transform(df_clean['text_final'].dropna())
words = vec.get_feature_names_out()
sums = tfidf_matrix.sum(axis=0).A1
data_tfidf = pd.Series(sums, index=words).sort_values(ascending=False)
fig, ax = plt.subplots()
sns.barplot(x=data_tfidf.values, y=data_tfidf.index, ax=ax)
st.pyplot(fig)

# 8. Sample Data
st.header("8️⃣ Sample Ulasan per Sentimen")
stype = st.selectbox("Pilih sentimen:", ['positive', 'negative', 'neutral'])
st.dataframe(df_clean[df_clean['polarity'] == stype].head(10), use_container_width=True)

# 9. Download
st.header("9️⃣ Download Data Hasil Analisis")
csv_data = df_clean.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download CSV", data=csv_data, file_name='hasil_analisis.csv', mime='text/csv')

with st.expander("ℹ️ Informasi Sistem"):
    st.write(f"**Python:** {sys.version.split()[0]} | **Pandas:** {pd.__version__}")