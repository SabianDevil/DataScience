import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(page_title="Posisi Prediksi Data", layout="wide")

# Mengatur tema Matplotlib menjadi Gelap (agar sesuai gambar Anda)
plt.style.use('dark_background')

st.title("📈 Posisi Prediksi pada Data Historis")
st.markdown("Visualisasi posisi data saat ini terhadap keseluruhan data historis.")

# --- SIDEBAR ---
st.sidebar.header("📂 Panel Kontrol")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # 1. BACA DATA
        df = pd.read_csv(uploaded_file)
        
        # 2. PILIH KOLOM (Agar user bisa atur sendiri)
        st.sidebar.subheader("⚙️ Konfigurasi Kolom")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        col_flow = st.sidebar.selectbox("Pilih Kolom Flow:", numeric_cols, index=0)
        col_occ = st.sidebar.selectbox("Pilih Kolom Occupancy:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

        # 3. KONTROL INTERAKTIF (Slider)
        st.sidebar.divider()
        st.sidebar.subheader("🎚️ Simulasi Prediksi")
        
        # Slider untuk menentukan posisi titik MERAH (Interval)
        max_idx = len(df) - 1
        posisi_saat_ini = st.sidebar.slider("Posisi Interval (Titik Merah)", 0, max_idx, int(max_idx/2))
        
        # Slider untuk menentukan batas ambang KUNING (Threshold)
        max_flow = int(df[col_flow].max())
        max_occ = float(df[col_occ].max())
        
        threshold_flow = st.sidebar.slider("Threshold Macet (Flow)", 0, max_flow, int(max_flow * 0.7))
        threshold_occ = st.sidebar.slider("Threshold Macet (Occupancy)", 0.0, max_occ, max_occ * 0.3)

        # --- VISUALISASI MATPLOTLIB ---
        
        # Membuat canvas gambar (2 kolom: Kiri Flow, Kanan Occupancy)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # === GRAFIK 1: FLOW ===
        # A. Plot Data Historis (Abu-abu transparan)
        ax1.plot(df.index, df[col_flow], color='white', alpha=0.15, label='Data Historis', linewidth=1)
        
        # B. Plot Titik Merah (Prediksi Saat Ini)
        nilai_flow_saat_ini = df.iloc[posisi_saat_ini][col_flow]
        ax1.scatter(posisi_saat_ini, nilai_flow_saat_ini, color='red', s=100, zorder=5, label='Prediksi Saat Ini')
        
        # C. Plot Garis Kuning (Threshold)
        ax1.axhline(y=threshold_flow, color='#ffcc00', linestyle='--', linewidth=1.5, label='Threshold Macet')

        # D. Kosmetik Grafik
        ax1.set_title("Flow Historis dengan Prediksi Saat Ini", fontsize=14, pad=15)
        ax1.set_ylabel("Flow")
        ax1.set_xlabel("interval")
        ax1.legend(loc='upper right', frameon=True, facecolor='white', labelcolor='black')
        ax1.grid(False) # Hilangkan grid agar mirip gambar referensi

        # === GRAFIK 2: OCCUPANCY ===
        # A. Plot Data Historis
        ax2.plot(df.index, df[col_occ], color='white', alpha=0.15, label='Data Historis', linewidth=1)
        
        # B. Plot Titik Merah
        nilai_occ_saat_ini = df.iloc[posisi_saat_ini][col_occ]
        ax2.scatter(posisi_saat_ini, nilai_occ_saat_ini, color='red', s=100, zorder=5, label='Prediksi Saat Ini')
        
        # C. Plot Garis Kuning
        ax2.axhline(y=threshold_occ, color='#ffcc00', linestyle='--', linewidth=1.5, label='Threshold Macet')

        # D. Kosmetik Grafik
        ax2.set_title("Occupancy Historis dengan Prediksi Saat Ini", fontsize=14, pad=15)
        ax2.set_ylabel("Occupancy")
        ax2.set_xlabel("interval")
        ax2.legend(loc='upper right', frameon=True, facecolor='white', labelcolor='black')
        ax2.grid(False)

        # Tampilkan ke Streamlit
        st.pyplot(fig)

        # Tampilkan Angka Detail di Bawah Grafik
        c1, c2, c3 = st.columns(3)
        c1.metric("Posisi Interval", posisi_saat_ini)
        c2.metric("Nilai Flow Saat Ini", f"{nilai_flow_saat_ini:.2f}", delta=f"{nilai_flow_saat_ini - threshold_flow:.2f} dari batas")
        c3.metric("Nilai Occupancy Saat Ini", f"{nilai_occ_saat_ini:.4f}", delta=f"{nilai_occ_saat_ini - threshold_occ:.4f} dari batas")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.info("Tips: Pastikan file CSV Anda memiliki kolom angka untuk Flow dan Occupancy.")

else:
    st.info("👋 Silakan upload file CSV Birmingham/Traffic Anda.")
