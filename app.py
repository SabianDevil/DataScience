import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Prediksi Kemacetan", layout="wide")
plt.style.use('dark_background')

# --- SIDEBAR: LOGO & JUDUL ---
st.sidebar.title("🚦 Sistem Prediksi Kemacetan")
st.sidebar.markdown("Input detik (0-86400) untuk memprediksi kondisi lalu lintas.")

# --- 1. UPLOAD DATA ---
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # --- 2. PERSIAPAN DATA (TIDAK BUTUH TANGGAL) ---
        # Kita anggap 'index' baris sebagai representasi waktu (urutan detik/interval)
        # X = Fitur Waktu (Kita buat urutan angka dari 0 sampai jumlah data)
        df['interval_index'] = np.arange(len(df))
        
        # Konfigurasi Kolom Target
        st.sidebar.subheader("⚙️ Konfigurasi Data")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        col_flow = st.sidebar.selectbox("Kolom Flow:", numeric_cols, index=0)
        idx_occ = 1 if len(numeric_cols) > 1 else 0
        col_occ = st.sidebar.selectbox("Kolom Occupancy:", numeric_cols, index=idx_occ)

        # --- 3. PILIH MODEL AI ---
        st.sidebar.divider()
        st.sidebar.subheader("🧠 Pengaturan Model AI")
        model_option = st.sidebar.selectbox(
            "Pilih Model AI:",
            ("Decision Tree (Akurat)", "Linear Regression (Simpel)", "Polynomial Regression (Melengkung)")
        )

        # --- 4. INPUT DETIK (SEPERTI TEMAN ANDA) ---
        st.sidebar.divider()
        st.sidebar.subheader("⏱️ Metode Input")
        
        # Slider/Input angka dari 0 sampai jumlah data (maksimum detik)
        max_val = len(df)
        input_detik = st.sidebar.number_input(f"Input Detik/Interval (0 - {max_val})", min_value=0, max_value=max_val, value=int(max_val/2))

        if st.sidebar.button("🚀 Prediksi Sekarang", type="primary"):
            
            # --- PROSES PELATIHAN MODEL (TRAINING) ---
            X = df[['interval_index']] # Data input (Waktu)
            y_flow = df[col_flow]      # Target 1
            y_occ = df[col_occ]        # Target 2
            
            # Logika pemilihan model
            if "Decision Tree" in model_option:
                model_flow = DecisionTreeRegressor(max_depth=10)
                model_occ = DecisionTreeRegressor(max_depth=10)
            elif "Polynomial" in model_option:
                # Derajat 4 agar bisa melengkung naik turun
                model_flow = make_pipeline(PolynomialFeatures(4), LinearRegression())
                model_occ = make_pipeline(PolynomialFeatures(4), LinearRegression())
            else:
                model_flow = LinearRegression()
                model_occ = LinearRegression()
            
            # Latih Model (Fit)
            model_flow.fit(X, y_flow)
            model_occ.fit(X, y_occ)
            
            # --- LAKUKAN PREDIKSI ---
            input_data = pd.DataFrame({'interval_index': [input_detik]})
            pred_flow = model_flow.predict(input_data)[0]
            pred_occ = model_occ.predict(input_data)[0]
            
            # Hitung Threshold Otomatis
            thresh_flow = df[col_flow].max() * 0.7
            thresh_occ = df[col_occ].max() * 0.3
            
            # --- TAMPILAN UTAMA ---
            st.title("Sistem Prediksi Kemacetan Munich")
            
            # 1. KARTU STATUS (Metric Besar)
            col_stat, col_f, col_o, col_int, col_mod = st.columns(5)
            
            # Tentukan Status
            status_text = "LANCAR"
            status_color = "green"
            if pred_occ > thresh_occ:
                status_text = "MACET TOTAL"
                status_color = "red"
            elif pred_flow > thresh_flow:
                status_text = "PADAT MERAYAP"
                status_color = "orange"
                
            col_stat.markdown(f"### Status:\n## :{status_color}[{status_text}]")
            col_f.metric("Prediksi Flow", f"{pred_flow:.1f}")
            col_o.metric("Prediksi Occupancy", f"{pred_occ:.2f}%")
            col_int.metric("Input Interval", input_detik)
            col_mod.markdown(f"**Model AI:**\n{model_option}")
            
            st.divider()

            # 2. GRAFIK POSISI PREDIKSI
            st.subheader("📈 Posisi Prediksi pada Data Historis")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.patch.set_facecolor('black')
            fig.patch.set_alpha(0.0)

            # Grafik Flow
            ax1.plot(df['interval_index'], df[col_flow], color='white', alpha=0.1, label='Data Asli')
            # Garis Prediksi Model (Opsional: Menggambar garis pola AI)
            y_flow_pred_all = model_flow.predict(X)
            ax1.plot(df['interval_index'], y_flow_pred_all, color='cyan', alpha=0.4, linewidth=1, label='Pola AI')
            
            # Titik Merah Prediksi
            ax1.scatter(input_detik, pred_flow, color='red', s=200, zorder=10, label='Prediksi Saat Ini', edgecolors='white')
            ax1.axhline(thresh_flow, color='#ffcc00', linestyle='--', label='Threshold')
            
            ax1.set_title("Flow Historis vs Prediksi AI", color='white')
            ax1.set_facecolor('black')
            ax1.grid(False)
            legend1 = ax1.legend(facecolor='#262730', edgecolor='white')
            for text in legend1.get_texts(): text.set_color("white")
            ax1.tick_params(colors='white')

            # Grafik Occupancy
            ax2.plot(df['interval_index'], df[col_occ], color='white', alpha=0.1, label='Data Asli')
            y_occ_pred_all = model_occ.predict(X)
            ax2.plot(df['interval_index'], y_occ_pred_all, color='cyan', alpha=0.4, linewidth=1, label='Pola AI')
            
            ax2.scatter(input_detik, pred_occ, color='red', s=200, zorder=10, label='Prediksi Saat Ini', edgecolors='white')
            ax2.axhline(thresh_occ, color='#ffcc00', linestyle='--', label='Threshold')
            
            ax2.set_title("Occupancy Historis vs Prediksi AI", color='white')
            ax2.set_facecolor('black')
            ax2.grid(False)
            legend2 = ax2.legend(facecolor='#262730', edgecolor='white')
            for text in legend2.get_texts(): text.set_color("white")
            ax2.tick_params(colors='white')

            st.pyplot(fig)

        else:
            st.info("👈 Masukkan Detik/Interval di Sidebar lalu klik tombol 'Prediksi Sekarang'.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

else:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Duke_University_Logo.svg/1200px-Duke_University_Logo.svg.png", width=100) # Logo placeholder
    st.title("Sistem Prediksi Kemacetan Munich")
    st.info("👋 Silakan upload file CSV di sebelah kiri.")
