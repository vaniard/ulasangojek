# ============================================
# KONFIGURASI STREAMLIT - HARUS DI AWAL
# ============================================
import streamlit as st

# Set page config dengan sidebar terbuka (expanded)
st.set_page_config(page_title="Analisis Sentimen Gojek", layout="wide", initial_sidebar_state="expanded")

# ============================================
# IMPORT MODULES (Tanpa fungsi install otomatis agar stabil)
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
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

# Setup NLTK Data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# ============================================
# SIDEBAR NAVIGATION
# ============================================
with st.sidebar:
    st.title("🚀 Menu Navigasi")
    st.markdown("---")
    menu = st.radio(
        "Pilih Halaman:",
        ["Dashboard Analisis", "Cek Sentimen Mandiri"]
    )
    st.markdown("---")
    st.markdown("### Info Mahasiswa")
    st.text("Vania - Skripsi 2024")

# ============================================
# FUNGSI-FUNGSI UTAMA
# ============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ulasan_gojek.csv')
        df_clean = pd.read_csv('clean_data_ulasan.csv')
        return df, df_clean
    except FileNotFoundError:
        # Data dummy jika file tidak ditemukan saat awal deploy
        df = pd.DataFrame({'Review': ['Sangat bagus!'], 'Rating': [5]})
        df_clean = df.copy()
        df_clean['text_final'] = df_clean['Review'].str.lower()
        return df, df_clean

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
        pos, neg = {'bagus': 1}, {'lambat': -1}
    return pos, neg

def get_sentiment(text):
    score = 0
    if not text or pd.isna(text): return 0, 'neutral'
    words = str(text).lower().split()
    for w in words:
        if w in pos_dict: score += pos_dict[w]
        elif w in neg_dict: score += neg_dict[w]
    return (score, 'positive' if score > 0 else 'negative' if score < 0 else 'neutral')

# Pre-load data & lexicon
df_raw, df_clean = load_data()
pos_dict, neg_dict = load_lexicon()

# ============================================
# HALAMAN 1: DASHBOARD UTAMA
# ============================================
if menu == "Dashboard Analisis":
    st.title("📊 Dashboard Analisis Sentimen Gojek")
    
    # --- 1. Data Awal ---
    st.header("1️⃣ Data Awal Ulasan")
    st.dataframe(df_raw.head(5), use_container_width=True)

    # --- 2. Preprocessing ---
    st.header("2️⃣ Hasil Preprocessing")
    if 'text_final' not in df_clean.columns: 
        df_clean['text_final'] = df_clean['Review'].str.lower()
    st.dataframe(df_clean[['Review', 'text_final']].head(5), use_container_width=True)

    # --- 3. Pelabelan Sentimen ---
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

    # --- 4. Visualisasi Distribusi (DIPERKECIL) ---
    st.header("4️⃣ Distribusi Sentimen")
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        fig1, ax1 = plt.subplots(figsize=(5, 3)) # Ukuran diperkecil
        df_clean['polarity'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax1, colors=['#2ecc71', '#e74c3c', '#95a5a6'], fontsize=8)
        ax1.set_ylabel('')
        st.pyplot(fig1)

    with col_dist2:
        fig2, ax2 = plt.subplots(figsize=(5, 3)) # Ukuran diperkecil
        sns.countplot(data=df_clean, x='polarity', palette=['#2ecc71', '#e74c3c', '#95a5a6'], ax=ax2)
        ax2.tick_params(labelsize=8)
        st.pyplot(fig2)

    # --- 5. Word Cloud (DIPERKECIL) ---
    st.header("5️⃣ Word Cloud")
    col_wc1, col_wc2 = st.columns(2)
    
    with col_wc1:
        st.write("**Positif**")
        txt_pos = ' '.join(df_clean[df_clean['polarity'] == 'positive']['text_final'].dropna())
        if txt_pos:
            wc = WordCloud(background_color='white', width=400, height=200).generate(txt_pos)
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.imshow(wc); ax.axis('off'); st.pyplot(fig)

    with col_wc2:
        st.write("**Negatif**")
        txt_neg = ' '.join(df_clean[df_clean['polarity'] == 'negative']['text_final'].dropna())
        if txt_neg:
            wc = WordCloud(background_color='white', width=400, height=200, colormap='Reds').generate(txt_neg)
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.imshow(wc); ax.axis('off'); st.pyplot(fig)

    # --- 6. Distribusi Panjang Teks ---
    st.header("6️⃣ Statistik Panjang Teks")
    fig3, ax3 = plt.subplots(figsize=(8, 2.5))
    df_clean['text_final'].str.split().str.len().hist(bins=20, ax=ax3, color='skyblue')
    ax3.set_title("Jumlah Kata Per Ulasan", fontsize=10)
    st.pyplot(fig3)

    # --- 7. TF-IDF ---
    st.header("7️⃣ Top 20 Kata (TF-IDF)")
    vec = TfidfVectorizer(max_features=20)
    tfidf = vec.fit_transform(df_clean['text_final'].dropna())
    feat_data = pd.Series(tfidf.sum(axis=0).A1, index=vec.get_feature_names_out()).sort_values(ascending=False)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=feat_data.values, y=feat_data.index, ax=ax4, palette='viridis')
    ax4.tick_params(labelsize=8)
    st.pyplot(fig4)

    # --- 8. Sample Ulasan ---
    st.header("8️⃣ Lihat Sampel Data")
    stype = st.selectbox("Filter Sentimen:", ['positive', 'negative', 'neutral'])
    st.dataframe(df_clean[df_clean['polarity'] == stype].head(10), use_container_width=True)

    # --- 9. Download ---
    st.header("9️⃣ Ekspor Data")
    csv_res = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Hasil Analisis (.csv)", csv_res, "hasil_analisis_gojek.csv", "text/csv")

# ============================================
# HALAMAN 2: CEK SENTIMEN MANDIRI
# ============================================
else:
    st.title("🔍 Cek Sentimen Mandiri")
    st.write("Gunakan fitur ini untuk mencoba prediksi sentimen pada kalimat baru.")
    
    teks_input = st.text_area("Masukkan ulasan pelanggan di sini:", placeholder="Contoh: Gojek sangat memudahkan hidup saya!")
    
    if st.button("Analisis Sekarang"):
        if teks_input:
            skor, label