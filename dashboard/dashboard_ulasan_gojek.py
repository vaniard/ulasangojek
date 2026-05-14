# ============================================
# KONFIGURASI STREAMLIT - HARUS DI AWAL
# ============================================
import streamlit as st

# Harus menjadi perintah Streamlit pertama
st.set_page_config(page_title="Analisis Sentimen Ulasan Gojek", layout="wide", initial_sidebar_state="collapsed")

# ============================================
# IMPORT MODULES
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import requests
from io import StringIO
import csv
import warnings
import sys
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

# Download nltk data secara otomatis saat deploy
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

# Custom CSS tetap dipertahankan sesuai keinginanmu
st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    h1 { font-size: 2rem !important; color: #1f77b4 !important; border-bottom: 3px solid #1f77b4; display: inline-block; }
    .stMetric { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; padding: 0.5rem; color: white; }
    .metric-card { background: white; border-radius: 10px; padding: 0.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .metric-value { font-size: 1.5rem; font-weight: bold; color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Analisis Sentimen Ulasan Gojek")
st.markdown("<p style='color: #6c757d;'>Dashboard analisis sentimen otomatis dari ulasan pengguna aplikasi Gojek</p>", unsafe_allow_html=True)

# ============================================
# FUNGSI & LOGIKA
# ============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ulasan_gojek.csv')
        df_clean = pd.read_csv('clean_data_ulasan.csv')
        return df, df_clean
    except FileNotFoundError:
        sample = pd.DataFrame({'Review': ['Bagus sekali', 'Jelek banget'], 'Rating': [5, 1]})
        df_clean = sample.copy()
        df_clean['text_final'] = df_clean['Review'].str.lower()
        return sample, df_clean

@st.cache_data
def load_lexicon():
    pos, neg = {}, {}
    try:
        r_pos = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv', timeout=10)
        r_neg = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv', timeout=10)
        if r_pos.status_code == 200:
            for row in csv.reader(StringIO(r_pos.text)):
                if row: pos[row[0]] = int(row[1]) if len(row) > 1 else 1
        if r_neg.status_code == 200:
            for row in csv.reader(StringIO(r_neg.text)):
                if row: neg[row[0]] = int(row[1]) if len(row) > 1 else -1
    except:
        pos, neg = {'bagus': 1}, {'jelek': -1}
    return pos, neg

def get_sentiment(text):
    score = 0
    words = str(text).lower().split()
    for w in words:
        if w in pos_dict: score += pos_dict[w]
        elif w in neg_dict: score += neg_dict[w]
    return (score, 'positive' if score > 0 else 'negative' if score < 0 else 'neutral')

# Load data dan lexicon
df_raw, df_clean = load_data()
pos_dict, neg_dict = load_lexicon()

# --- 1. Data Awal ---
st.header("1️⃣ Data Awal Ulasan")
st.dataframe(df_raw.head(10), use_container_width=True)

# --- 2. Preprocessing ---
st.header("2️⃣ Proses Preprocessing Teks")
if 'text_final' not in df_clean.columns: df_clean['text_final'] = df_clean['Review'].str.lower()
col1, col2 = st.columns(2)
col1.dataframe(df_clean[['Review', 'text_final']].head(5), use_container_width=True)
col2.markdown(f'<div class="metric-card"><div class="metric-value">{len(df_clean)}</div><div>Jumlah Data</div></div>', unsafe_allow_html=True)

# --- 3. Pelabelan ---
st.header("3️⃣ Pelabelan Sentimen")
if 'polarity' not in df_clean.columns:
    res = df_clean['text_final'].apply(get_sentiment)
    df_clean['polarity_score'] = [r[0] for r in res]
    df_clean['polarity'] = [r[1] for r in res]
counts = df_clean['polarity'].value_counts()
c1, c2, c3 = st.columns(3)
c1.metric("😊 Positif", counts.get('positive', 0))
c2.metric("😞 Negatif", counts.get('negative', 0))
c3.metric("😐 Netral", counts.get('neutral', 0))

# --- 4. Distribusi ---
st.header("4️⃣ Distribusi Sentimen")
col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    df_clean['polarity'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c', '#95a5a6'], ax=ax1)
    st.pyplot(fig1)
with col2:
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_clean, x='polarity', palette=['#2ecc71', '#e74c3c', '#95a5a6'], ax=ax2)
    st.pyplot(fig2)

# --- 5. WordCloud ---
st.header("5️⃣ Word Cloud")
col1, col2 = st.columns(2)
def plot_wc(sent, color, title):
    txt = ' '.join(df_clean[df_clean['polarity'] == sent]['text_final'].dropna())
    if txt:
        wc = WordCloud(background_color='white', colormap=color).generate(txt)
        fig, ax = plt.subplots()
        ax.imshow(wc); ax.axis('off'); st.pyplot(fig)
with col1: st.subheader("☀️ Positif"); plot_wc('positive', 'Greens', 'Positif')
with col2: st.subheader("🌧️ Negatif"); plot_wc('negative', 'Reds', 'Negatif')

# --- 6. Panjang Teks ---
st.header("6️⃣ Distribusi Panjang Teks")
fig3, ax3 = plt.subplots()
df_clean['text_final'].str.split().str.len().hist(bins=20, ax=ax3, color='skyblue')
st.pyplot(fig3)

# --- 7. TF-IDF ---
st.header("7️⃣ Top 20 Kata TF-IDF")
vec = TfidfVectorizer(max_features=20)
tfidf = vec.fit_transform(df_clean['text_final'].dropna())
feat_data = pd.Series(tfidf.sum(axis=0).A1, index=vec.get_feature_names_out()).sort_values(ascending=False)
fig4, ax4 = plt.subplots()
sns.barplot(x=feat_data.values, y=feat_data.index, ax=ax4)
st.pyplot(fig4)

# --- 8. Sample ---
st.header("8️⃣ Sample Ulasan")
stype = st.selectbox("Pilih sentimen:", ['positive', 'negative', 'neutral'])
st.dataframe(df_clean[df_clean['polarity'] == stype].head(10), use_container_width=True)

# --- 9. Download ---
st.header("9️⃣ Download Hasil")
st.download_button("📥 Download CSV", df_clean.to_csv(index=False).encode('utf-8'), "hasil_analisis.csv", "text/csv")