import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Konfigurasi Halaman
st.set_page_config(page_title="Traffic Prediction AI", layout="wide")
plt.style.use('dark_background')

st.title("🤖 AI Prediksi Kemacetan Lalu Lintas")
st.markdown("Dashboard ini melatih model Machine Learning untuk memprediksi status kemacetan berdasarkan data historis.")

# --- FUNGSI UTAMA ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def hitung_metrik(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    return acc, prec, rec, f1

# --- SIDEBAR ---
st.sidebar.header("1. Upload & Labeling")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # --- MENENTUKAN TARGET (LABEL) ---
    # Karena data mentah biasanya tidak punya kolom "Macet/Lancar", kita buat sendiri aturannya.
    st.sidebar.subheader("Definisi 'Macet'")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # User memilih kolom mana yang jadi patokan (misal: Occupancy)
    col_feature = st.sidebar.selectbox("Pilih Kolom Fitur Utama (misal: Occupancy/Flow)", numeric_cols, index=0)
    
    # User menentukan batas ambang
    max_val = float(df[col_feature].max())
    threshold = st.sidebar.slider(f"Jika {col_feature} di atas angka ini, dianggap MACET:", 0.0, max_val, max_val*0.3)
    
    # Membuat kolom Target (0 = Lancar, 1 = Macet)
    df['Status'] = df[col_feature].apply(lambda x: 1 if x > threshold else 0)
    df['Label_Status'] = df['Status'].apply(lambda x: 'Macet 🔴' if x == 1 else 'Lancar 🟢')

    # --- MEMBUAT TABS ---
    tab1, tab2 = st.tabs(["🧠 Training & Analisis Model", "🔮 Simulasi Prediksi"])

    # =========================================
    # TAB 1: ANALISIS PERFORMA MODEL
    # =========================================
    with tab1:
        st.header("Analisis Performa Model")
        st.info("Di sini kita membandingkan dua algoritma untuk melihat mana yang terbaik.")

        # Persiapan Data (X = Fitur, y = Target)
        X = df[[col_feature]] # Bisa ditambah kolom lain jika ada
        y = df['Status']

        # Split Data (80% Latihan, 20% Ujian)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        col_model1, col_model2 = st.columns(2)

        # --- MODEL A: Decision Tree ---
        with col_model1:
            st.subheader("Model A: Decision Tree")
            model_dt = DecisionTreeClassifier()
            model_dt.fit(X_train, y_train)
            y_pred_dt = model_dt.predict(X_test)
            
            acc_dt, prec_dt, rec_dt, f1_dt = hitung_metrik(y_test, y_pred_dt)
            st.write(f"**Akurasi:** {acc_dt:.2%}")
            st.write(f"**F1-Score:** {f1_dt:.2%}")
            st.caption("Kelebihan: Mudah diinterpretasi, tapi rawan overfitting.")

        # --- MODEL B: Random Forest ---
        with col_model2:
            st.subheader("Model B: Random Forest")
            model_rf = RandomForestClassifier(n_estimators=50)
            model_rf.fit(X_train, y_train)
            y_pred_rf = model_rf.predict(X_test)
            
            acc_rf, prec_rf, rec_rf, f1_rf = hitung_metrik(y_test, y_pred_rf)
            st.write(f"**Akurasi:** {acc_rf:.2%}")
            st.write(f"**F1-Score:** {f1_rf:.2%}")
            st.caption("Kelebihan: Lebih stabil dan akurat untuk data kompleks.")

        st.divider()
        
        # --- TABEL PERBANDINGAN METRIK (Analisis Wajib) ---
        st.subheader("📊 Tabel Perbandingan Metrik")
        metrics_df = pd.DataFrame({
            'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Decision Tree': [acc_dt, prec_dt, rec_dt, f1_dt],
            'Random Forest': [acc_rf, prec_rf, rec_rf, f1_rf]
        })
        st.dataframe(metrics_df.style.format("{:.2%}"), use_container_width=True)
        
        st.write("""
        **Penjelasan Metrik:**
        * **Accuracy:** Seberapa sering model menebak benar secara keseluruhan.
        * **Precision:** Dari semua yang ditebak "Macet", berapa yang benar-benar macet?
        * **Recall:** Dari kejadian yang aslinya "Macet", berapa yang berhasil dideteksi? (Penting agar tidak terlewat).
        * **F1-Score:** Rata-rata harmonis Precision & Recall. Gunakan ini jika data tidak seimbang.
        """)

    # =========================================
    # TAB 2: SIMULASI PREDIKSI (UI INPUT USER)
    # =========================================
    with tab2:
        st.header("Simulasi Prediksi Real-time")
        
        # Input User
        col_input, col_result = st.columns([1, 2])
        
        with col_input:
            st.subheader("🎚️ Input Data")
            input_val = st.number_input(f"Masukkan Nilai {col_feature}", min_value=0.0, value=float(threshold))
            
            # Tombol Prediksi
            if st.button("🔍 Prediksi Status"):
                # Kita pakai Random Forest karena biasanya lebih baik
                prediksi = model_rf.predict([[input_val]])[0]
                probabilitas = model_rf.predict_proba([[input_val]])[0][1] # Peluang macet
                
                with col_result:
                    st.subheader("Hasil Prediksi")
                    if prediksi == 1:
                        st.error(f"⚠️ Status: MACET PARAH (Confidence: {probabilitas:.1%})")
                    else:
                        st.success(f"✅ Status: LANCAR JAYA (Confidence: {1-probabilitas:.1%})")
                    
                    # --- VISUALISASI POSISI (Seperti request Anda) ---
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    # Plot Data Historis (Abu-abu)
                    sns.histplot(data=df, x=col_feature, hue='Label_Status', element="step", alpha=0.3, ax=ax)
                    
                    # Plot Posisi Input User (Titik Merah/Biru Besar)
                    warna_titik = 'red' if prediksi == 1 else '#00ff00'
                    ax.scatter(input_val, 50, color=warna_titik, s=200, zorder=5, label='Input Anda', edgecolors='white')
                    
                    # Garis Threshold
                    ax.axvline(threshold, color='yellow', linestyle='--', label='Batas Macet')
                    
                    ax.set_title(f"Posisi Input Anda ({input_val}) terhadap Data Historis")
                    ax.legend()
                    st.pyplot(fig)

else:
    st.info("Silakan upload file CSV di sidebar untuk memulai pelatihan AI.")
