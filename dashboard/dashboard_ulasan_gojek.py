# ============================================
# KONFIGURASI STREAMLIT - HARUS DI AWAL
# ============================================
import streamlit as st

# Set page config
st.set_page_config(page_title="Analisis Sentimen Gojek", layout="wide", initial_sidebar_state="expanded")

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
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

# Download nltk data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# ============================================
# SIDEBAR NAVIGATION
# ============================================
with st.sidebar:
    st.title("🚀 Navigasi")
    st.markdown("---")
    menu = st.radio("Pilih Halaman:", ["Dashboard Utama", "Uji Sentimen Mandiri"])
    st.markdown("---")
    st.write("**Skripsi Vania**")

# ============================================
# FUNGSI-FUNGSI (Lexicon & Data)
# ============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ulasan_gojek.csv')
        df_clean = pd.read_csv('clean_data_ulasan.csv')
        return df, df_clean
    except:
        df = pd.DataFrame({'Review': ['Bagus'], 'Rating': [5]})
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
        pos, neg = {'bagus': 1}, {'jelek': -1}
    return pos, neg

def get_sentiment(text):
    score = 0
    words = str(text).lower().split()
    for w in words:
        if w in pos_dict: score += pos_dict[w]
        elif w in neg_dict: score += neg_dict[w]
    label = 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'
    return score, label

# Load Global
pos_dict, neg_dict = load_lexicon()
df_raw, df_clean = load_data()

# ============================================
# LOGIKA TAMPILAN
# ============================================

if menu == "Dashboard Utama":
    st.title("📊 Analisis Sentimen Ulasan Gojek")
    
    # 1. Data Awal
    st.header("1️⃣ Data Awal Ulasan")
    st.dataframe(df_raw.head(5), use_container_width=True)

    # 2. Preprocessing
    st.header("2️⃣ Hasil Preprocessing")
    if 'text_final' not in df_clean.columns: df_clean['text_final'] = df_clean['Review'].str.lower()
    st.dataframe(df_clean[['Review', 'text_final']].head(5), use_container_width=True)

    # 3. Pelabelan
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

    # 4. Distribusi (UKURAN DISESUAIKAN)
    st.header("4️⃣ Distribusi Sentimen")
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        fig1, ax1 = plt.subplots(figsize=(4, 3)) # Ukuran kecil
        df_clean['polarity'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax1, colors=['#2ecc71', '#e74c3c', '#95a5a6'])
        ax1.set_ylabel('')
        st.pyplot(fig1)
    with col_plot2:
        fig2, ax2 = plt.subplots(figsize=(4, 3)) # Ukuran kecil
        sns.countplot(data=df_clean, x='polarity', palette=['#2ecc71', '#e74c3c', '#95a5a6'], ax=ax2)
        st.pyplot(fig2)

    # 5. WordCloud (UKURAN DISESUAIKAN)
    st.header("5️⃣ Word Cloud")
    col_wc1, col_wc2 = st.columns(2)
    with col_wc1:
        st.write("☀️ **Positif**")
        txt = ' '.join(df_clean[df_clean['polarity'] == 'positive']['text_final'].dropna())
        if txt:
            wc = WordCloud(background_color='white', width=400, height=200).generate(txt)
            fig, ax = plt.subplots(figsize=(4, 2)); ax.imshow(wc); ax.axis('off'); st.pyplot(fig)
    with col_wc2:
        st.write("🌧️ **Negatif**")
        txt = ' '.join(df_clean[df_clean['polarity'] == 'negative']['text_final'].dropna())
        if txt:
            wc = WordCloud(background_color='white', width=400, height=200, colormap='Reds').generate(txt)
            fig, ax = plt.subplots(figsize=(4, 2)); ax.imshow(wc); ax.axis('off'); st.pyplot(fig)

    # 6. Panjang Teks
    st.header("6️⃣ Distribusi Panjang Teks")
    fig3, ax3 = plt.subplots(figsize=(8, 2.5))
    df_clean['text_final'].str.split().str.len().hist(bins=20, ax=ax3, color='skyblue')
    st.pyplot(fig3)

    # 7. TF-IDF
    st.header("7️⃣ Top 20 Kata (TF-IDF)")
    vec = TfidfVectorizer(max_features=20)
    tfidf = vec.fit_transform(df_clean['text_final'].dropna())
    feat_data = pd.Series(tfidf.sum(axis=0).A1, index=vec.get_feature_names_out()).sort_values(ascending=False)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=feat_data.values, y=feat_data.index, ax=ax4, palette='viridis')
    st.pyplot(fig4)

    # 8. Sample
    st.header("8️⃣ Sample Ulasan")
    stype = st.selectbox("Pilih sentimen:", ['positive', 'negative', 'neutral'])
    st.dataframe(df_clean[df_clean['polarity'] == stype].head(10), use_container_width=True)

    # 9. Download
    st.header("9️⃣ Download Hasil")
    st.download_button("📥 Download CSV", df_clean.to_csv(index=False).encode('utf-8'), "hasil_analisis.csv", "text/csv")

else:
    st.title("🔍 Uji Sentimen Mandiri")
    st.write("Masukkan kalimat untuk melihat hasil analisis lexicon.")
    
    user_text = st.text_area("Kalimat Review:", placeholder="Contoh: Gojek sangat membantu saya...")
    
    if st.button("Cek Sentimen"):
        if user_text:
            # SEMUA LOGIKA HARUS DI DALAM SINI (Agar tidak NameError)
            skor, label_hasil = get_sentiment(user_text)
            st.markdown("---")
            if label_hasil == 'positive':
                st.success(f"### Hasil: POSITIF 😊\n**Skor:** {skor}")
            elif label_hasil == 'negative':
                st.error(f"### Hasil: NEGATIF 😞\n**Skor:** {skor}")
            else:
                st.warning(f"### Hasil: NETRAL 😐\n**Skor:** {skor}")
        else:
            st.info("Silakan ketik sesuatu dulu ya.")