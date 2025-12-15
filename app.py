import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Traffic Prediction AI", layout="wide")
plt.style.use('dark_background') # Tema Gelap untuk Grafik

st.title("🤖 AI Prediksi Kemacetan Lalu Lintas")
st.markdown("Dashboard ini melatih model Machine Learning untuk memprediksi status kemacetan dan mensimulasikan prediksi baru.")

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

# --- SIDEBAR: UPLOAD & KONFIGURASI ---
st.sidebar.header("1. Upload & Labeling")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = load_data(uploaded_file)
    
    # --- MENENTUKAN TARGET (LABEL OTOMATIS) ---
    st.sidebar.subheader("Definisi 'Macet'")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # User memilih kolom fitur utama
    col_feature = st.sidebar.selectbox("Pilih Kolom Fitur Utama (misal: Occupancy/Flow)", numeric_cols, index=0)
    
    # User menentukan batas ambang (Threshold)
    max_val = float(df[col_feature].max())
    default_thresh = max_val * 0.3
    threshold = st.sidebar.slider(f"Batas Nilai {col_feature} untuk dianggap MACET:", 0.0, max_val, default_thresh)
    
    # Membuat Label Target (0 = Lancar, 1 = Macet)
    df['Status'] = df[col_feature].apply(lambda x: 1 if x > threshold else 0)
    df['Label_Status'] = df['Status'].apply(lambda x: 'Macet 🔴' if x == 1 else 'Lancar 🟢')

    # --- MEMBUAT TABS ---
    tab1, tab2 = st.tabs(["🧠 Training & Analisis Model", "🔮 Simulasi Prediksi"])

    # =========================================
    # TAB 1: ANALISIS PERFORMA MODEL
    # =========================================
    with tab1:
        st.header("Analisis Performa Model")
        st.info("Membandingkan algoritma Decision Tree vs Random Forest.")

        # Persiapan Data (X = Fitur, y = Target)
        X = df[[col_feature]] 
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

        # --- MODEL B: Random Forest ---
        with col_model2:
            st.subheader("Model B: Random Forest")
            model_rf = RandomForestClassifier(n_estimators=50, random_state=42)
            model_rf.fit(X_train, y_train)
            y_pred_rf = model_rf.predict(X_test)
            
            acc_rf, prec_rf, rec_rf, f1_rf = hitung_metrik(y_test, y_pred_rf)
            st.write(f"**Akurasi:** {acc_rf:.2%}")
            st.write(f"**F1-Score:** {f1_rf:.2%}")

        st.divider()
        
        # --- TABEL PERBANDINGAN METRIK (SUDAH DIPERBAIKI) ---
        st.subheader("📊 Tabel Perbandingan Metrik")
        
        metrics_df = pd.DataFrame({
            'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Decision Tree': [acc_dt, prec_dt, rec_dt, f1_dt],
            'Random Forest': [acc_rf, prec_rf, rec_rf, f1_rf]
        })
        
        # PERBAIKAN: Set index ke kolom teks agar formatting % berjalan lancar
        metrics_df.set_index('Metrik', inplace=True)
        
        st.dataframe(metrics_df.style.format("{:.2%}"), use_container_width=True)
        
        st.caption("""
        **Keterangan:**
        * **Accuracy:** Ketepatan prediksi secara umum.
        * **Precision:** Ketepatan saat memprediksi 'Macet' (seberapa valid peringatan macetnya).
        * **Recall:** Kemampuan mendeteksi seluruh kejadian macet (agar tidak ada macet yang terlewat).
        """)

    # =========================================
    # TAB 2: SIMULASI PREDIKSI (UI INPUT USER)
    # =========================================
    with tab2:
        st.header("Simulasi Prediksi Real-time")
        
        col_input, col_result = st.columns([1, 2])
        
        with col_input:
            st.subheader("🎚️ Input Data Baru")
            # Default value diambil dari rata-rata biar user ga bingung
            default_input = float(df[col_feature].mean())
            input_val = st.number_input(f"Masukkan Nilai {col_feature}", min_value=0.0, value=default_input)
            
            tombol_prediksi = st.button("🔍 Prediksi Status", type="primary")
            
        with col_result:
            if tombol_prediksi:
                # Prediksi menggunakan Random Forest (Model B)
                prediksi = model_rf.predict([[input_val]])[0]
                probabilitas = model_rf.predict_proba([[input_val]])[0][1] # Ambil probabilitas kelas "1" (Macet)
                
                st.subheader("Hasil Prediksi")
                
                if prediksi == 1:
                    st.error(f"⚠️ Status: MACET (Keyakinan AI: {probabilitas:.1%})")
                else:
                    st.success(f"✅ Status: LANCAR (Keyakinan AI: {1-probabilitas:.1%})")
                
                # --- VISUALISASI POSISI (STYLE MATPLOTLIB DARK) ---
                st.divider()
                st.write(f"**Posisi Input ({input_val}) terhadap Data Historis:**")
                
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # 1. Plot Data Historis (Histogram / Area)
                sns.histplot(data=df, x=col_feature, hue='Label_Status', element="step", alpha=0.3, ax=ax, palette=['green', 'red'])
                
                # 2. Plot Garis Threshold (Kuning Putus-putus)
                ax.axvline(threshold, color='#ffcc00', linestyle='--', linewidth=2, label=f'Batas Macet ({threshold})')
                
                # 3. Plot Titik Input User (Lingkaran Besar)
                warna_titik = 'red' if prediksi == 1 else '#00ff00' # Merah atau Hijau Terang
                # Kita taruh titiknya di tengah ketinggian grafik (y=50) agar terlihat
                y_pos = ax.get_ylim()[1] / 2 
                ax.scatter(input_val, y_pos, color=warna_titik, s=300, zorder=10, label='Input Anda', edgecolors='white', linewidth=2)
                
                # Kosmetik Grafik
                ax.set_title(f"Visualisasi Posisi Data: {col_feature}", fontsize=14, color='white')
                ax.set_xlabel(col_feature, color='white')
                ax.set_ylabel("Frekuensi Jumlah Data", color='white')
                ax.tick_params(colors='white') # Warna angka sumbu jadi putih
                ax.legend(facecolor='#333333', labelcolor='white') # Legenda gelap
                ax.grid(False) # Hilangkan grid biar bersih
                
                # Background transparan agar menyatu dengan Streamlit Dark Mode
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)
                
                st.pyplot(fig)

else:
    st.info("👋 Silakan upload file CSV di sidebar sebelah kiri untuk memulai pelatihan AI.")
