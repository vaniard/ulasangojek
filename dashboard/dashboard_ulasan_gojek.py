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
import pkg_resources
st.write([pkg.key for pkg in pkg_resources.working_set])

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

# Function to install missing packages
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# Try importing nltk, install if not available
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import nltk
except ImportError:
    install_package('nltk')
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import nltk

# Try importing Sastrawi, install if not available
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
except ImportError:
    install_package('Sastrawi')
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Download nltk data with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

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
        # Create sample data for demonstration
        sample_data = {
            'Review': [
                'Aplikasi bagus sekali, sangat membantu',
                'Lambat banget, sering error',
                'Mantap, fitur lengkap',
                'Kurang suka, sering crash',
                'Pelayanan oke, driver ramah',
                'Aplikasi lemot, bikin kesal',
                'Sangat puas dengan layanan Gojek',
                'Banyak bug, perlu diperbaiki',
                'Keren banget, suka sekali',
                'Harga mahal, tidak sesuai'
            ],
            'Rating': [5, 1, 5, 2, 4, 2, 5, 1, 5, 3]
        }
        df = pd.DataFrame(sample_data)
        df_clean = df.copy()
        df_clean['text_final'] = df_clean['Review'].str.lower()
        return df, df_clean

# Load lexicon with error handling
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
                if row and len(row) > 1:
                    pos[row[0]] = int(row[1])
                elif row:
                    pos[row[0]] = 1
        else:
            st.warning("Gagal mengunduh lexicon positif. Menggunakan lexicon default.")
            
        if resp_neg.status_code == 200:
            reader = csv.reader(StringIO(resp_neg.text), delimiter=',')
            for row in reader:
                if row and len(row) > 1:
                    neg[row[0]] = int(row[1])
                elif row:
                    neg[row[0]] = -1
        else:
            st.warning("Gagal mengunduh lexicon negatif. Menggunakan lexicon default.")
            
    except requests.exceptions.RequestException as e:
        st.warning(f"Error mengunduh lexicon: {e}. Menggunakan lexicon default.")
        # Default lexicon
        pos = {
            'bagus': 1, 'baik': 1, 'mantap': 1, 'suka': 1, 'keren': 1, 
            'puas': 1, 'oke': 1, 'lengkap': 1, 'ramah': 1, 'cepat': 1,
            'senang': 1, 'recommended': 1, 'terbaik': 1, 'top': 1
        }
        neg = {
            'buruk': -1, 'jelek': -1, 'lambat': -1, 'error': -1, 'kurang': -1,
            'lemot': -1, 'kesal': -1, 'bug': -1, 'mahal': -1, 'crash': -1,
            'masalah': -1, 'gangguan': -1, 'parah': -1, 'kecewa': -1
        }
    
    return pos, neg

def sentiment_analysis_lexicon_indonesia(text):
    score = 0
    if not text or pd.isna(text):
        return 0, 'neutral'
    
    words = str(text).lower().split()
    for word in words:
        if word in pos_dict:
            score += pos_dict[word]
        elif word in neg_dict:
            score += neg_dict[word]
    
    if score > 0:
        return score, 'positive'
    elif score < 0:
        return score, 'negative'
    else:
        return score, 'neutral'

# ============================================
# MAIN PROGRAM
# ============================================

# Load data
df_raw, df_clean = load_data()

# Initialize lexicon
pos_dict, neg_dict = load_lexicon()

# ===============================
# 1. Tampilan Data Awal
# ===============================
st.header("1️⃣ Data Awal Ulasan")
with st.container():
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.caption(f"Total ulasan: {df_raw.shape[0]} baris | {df_raw.shape[1]} kolom")

# ===============================
# 2. Text Preprocessing
# ===============================
st.header("2️⃣ Proses Preprocessing Teks")

# Check if text_final column exists, if not create it
if 'text_final' not in df_clean.columns and 'Review' in df_clean.columns:
    df_clean['text_final'] = df_clean['Review'].str.lower()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Hasil Preprocessing")
    if 'text_final' in df_clean.columns and 'Review' in df_clean.columns:
        st.dataframe(df_clean[['Review', 'text_final']].head(5), use_container_width=True)
    else:
        st.dataframe(df_clean.head(5), use_container_width=True)
with col2:
    st.subheader("Informasi Data")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df_clean.shape[0]}</div>
        <div class="metric-label">Jumlah Data</div>
        <div class="metric-value" style="font-size: 1rem; margin-top: 0.5rem;">{df_clean.shape[1]}</div>
        <div class="metric-label">Jumlah Kolom</div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# 3. Pelabelan Sentimen
# ===============================
st.header("3️⃣ Pelabelan Sentimen (Lexicon-based)")

# Add sentiment analysis if not already present
if 'polarity' not in df_clean.columns:
    with st.spinner('Menganalisis sentimen...'):
        results = df_clean['text_final'].apply(sentiment_analysis_lexicon_indonesia)
        df_clean['polarity_score'] = [r[0] for r in results]
        df_clean['polarity'] = [r[1] for r in results]

# Display sentiment distribution
sentiment_counts = df_clean['polarity'].value_counts()
st.write("Distribusi Sentimen:")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("😊 Positif", sentiment_counts.get('positive', 0))
with col2:
    st.metric("😞 Negatif", sentiment_counts.get('negative', 0))
with col3:
    st.metric("😐 Netral", sentiment_counts.get('neutral', 0))

st.success("✅ Sentimen berhasil ditambahkan ke data")

# ===============================
# 4. Distribusi Sentimen
# ===============================
st.header("4️⃣ Distribusi Sentimen")

col1, col2 = st.columns(2)

with col1:
    if not sentiment_counts.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sizes = sentiment_counts
        labels = sizes.index
        explode = [0.05] * len(sizes)
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', explode=explode, colors=colors[:len(sizes)])
        ax.set_title("Proporsi Sentimen", fontsize=12, pad=10)
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Tidak ada data sentimen untuk ditampilkan")

with col2:
    if not sentiment_counts.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df_clean, x='polarity', ax=ax, palette=['#2ecc71', '#e74c3c', '#95a5a6'])
        ax.set_title("Jumlah per Kategori Sentimen", fontsize=12, pad=10)
        ax.set_xlabel("Sentimen")
        ax.set_ylabel("Jumlah")
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width()/2., p.get_height()),
                       ha='center', va='bottom', fontsize=9)
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Tidak ada data sentimen untuk ditampilkan")

# ===============================
# 5. WordCloud
# ===============================
st.header("5️⃣ Word Cloud Berdasarkan Sentimen")

col1, col2 = st.columns(2)

with col1:
    st.subheader("☀️ Sentimen Positif")
    pos_text = ' '.join(df_clean[df_clean['polarity'] == 'positive']['text_final'].dropna())
    if pos_text and pos_text.strip():
        wc_pos = WordCloud(width=600, height=300, background_color='white', 
                          colormap='Greens', max_words=100).generate(pos_text)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(wc_pos, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Tidak ada teks positif untuk membuat word cloud")

with col2:
    st.subheader("🌧️ Sentimen Negatif")
    neg_text = ' '.join(df_clean[df_clean['polarity'] == 'negative']['text_final'].dropna())
    if neg_text and neg_text.strip():
        wc_neg = WordCloud(width=600, height=300, background_color='white', 
                          colormap='Reds', max_words=100).generate(neg_text)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(wc_neg, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Tidak ada teks negatif untuk membuat word cloud")

# ===============================
# 6. Distribusi Panjang Teks
# ===============================
st.header("6️⃣ Distribusi Panjang Teks")

df_clean['text_length'] = df_clean['text_final'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df_clean['text_length'], bins=30, kde=True, ax=ax, color='skyblue')
ax.set_title("Distribusi Panjang Teks", fontsize=12, pad=10)
ax.set_xlabel("Jumlah Kata")
ax.set_ylabel("Frekuensi")
st.pyplot(fig)
plt.close()

# ===============================
# 7. TF-IDF Top Words
# ===============================
st.header("7️⃣ Top 20 Kata berdasarkan TF-IDF")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Filter out empty or NaN values
    valid_texts = df_clean['text_final'].dropna()
    valid_texts = valid_texts[valid_texts.str.strip() != '']
    
    if len(valid_texts) > 0:
        vectorizer = TfidfVectorizer(max_features=50, min_df=2)
        X = vectorizer.fit_transform(valid_texts)
        df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        word_scores = df_tfidf.sum().sort_values(ascending=False).head(20)
        
        if not word_scores.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = sns.color_palette("viridis", len(word_scores))
            sns.barplot(x=word_scores.values, y=word_scores.index, ax=ax, palette=colors)
            ax.set_title("Top 20 Kata berdasarkan TF-IDF", fontsize=12, pad=10)
            ax.set_xlabel("Skor TF-IDF")
            ax.set_ylabel("Kata")
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Tidak dapat menghitung TF-IDF scores")
    else:
        st.warning("Tidak ada teks valid untuk analisis TF-IDF")
        
except ImportError:
    st.error("Scikit-learn tidak terinstall. Install dengan: pip install scikit-learn")
except Exception as e:
    st.error(f"Error dalam analisis TF-IDF: {e}")

# ===============================
# 8. Sample Data by Sentiment
# ===============================
st.header("8️⃣ Sample Ulasan per Sentimen")

sentiment_type = st.selectbox("Pilih sentimen untuk dilihat:", 
                              ['positive', 'negative', 'neutral'])

sample_data = df_clean[df_clean['polarity'] == sentiment_type][['Review', 'text_final', 'polarity_score']].head(10)
if not sample_data.empty:
    st.dataframe(sample_data, use_container_width=True)
else:
    st.info(f"Tidak ada data dengan sentimen {sentiment_type}")

# ===============================
# 9. Opsi Ekspor Data
# ===============================
st.header("9️⃣ Download Data Hasil Analisis")

col1, col2 = st.columns([1, 3])
with col1:

        st.cache_data.clear()
        st.rerun()
with col2:
    csv = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name='hasil_analisis_sentimen_gojek.csv',
        mime='text/csv',
        use_container_width=True
    )

# Display session info
with st.expander("ℹ️ Informasi Sistem"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Python:** {sys.version.split()[0]}")
    with col2:
        st.write(f"**Pandas:** {pd.__version__}")
    with col3:
        st.write(f"**Data Size:** {df_clean.shape[0]} rows, {df_clean.shape[1]} cols")