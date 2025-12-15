import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(page_title="Posisi Prediksi Data", layout="wide")

# Mengatur tema Matplotlib menjadi Gelap
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
        
        # 2. PILIH KOLOM
        st.sidebar.subheader("⚙️ Konfigurasi Kolom")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        col_flow = st.sidebar.selectbox("Pilih Kolom Flow:", numeric_cols, index=0)
        # Otomatis pilih kolom ke-2 untuk occupancy jika ada
        index_occ = 1 if len(numeric_cols) > 1 else 0
        col_occ = st.sidebar.selectbox("Pilih Kolom Occupancy:", numeric_cols, index=index_occ)

        # --- LOGIKA OTOMATIS (PENGGANTI SIMULASI PREDIKSI) ---
        # Karena slider dihapus, kita set nilai secara otomatis:
        
        # 1. Posisi saat ini = Data Terakhir (Paling Baru)
        posisi_saat_ini = len(df) - 1 
        
        # 2. Threshold Flow = 70% dari Flow Maksimal
        max_flow = int(df[col_flow].max())
        threshold_flow = int(max_flow * 0.7)
        
        # 3. Threshold Occupancy = 30% dari Occupancy Maksimal
        max_occ = float(df[col_occ].max())
        threshold_occ = max_occ * 0.3

        # --- VISUALISASI MATPLOTLIB ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # === GRAFIK 1: FLOW ===
        ax1.plot(df.index, df[col_flow], color='white', alpha=0.15, label='Data Historis', linewidth=1)
        nilai_flow_saat_ini = df.iloc[posisi_saat_ini][col_flow]
        ax1.scatter(posisi_saat_ini, nilai_flow_saat_ini, color='red', s=100, zorder=5, label='Data Terkini')
        ax1.axhline(y=threshold_flow, color='#ffcc00', linestyle='--', linewidth=1.5, label='Batas Macet')
        
        ax1.set_title("Flow Historis", fontsize=14, pad=15)
        ax1.set_ylabel("Flow")
        ax1.set_xlabel("interval")
        ax1.legend(loc='upper right', frameon=True, facecolor='white', labelcolor='black')
        ax1.grid(False)

        # === GRAFIK 2: OCCUPANCY ===
        ax2.plot(df.index, df[col_occ], color='white', alpha=0.15, label='Data Historis', linewidth=1)
        nilai_occ_saat_ini = df.iloc[posisi_saat_ini][col_occ]
        ax2.scatter(posisi_saat_ini, nilai_occ_saat_ini, color='red', s=100, zorder=5, label='Data Terkini')
        ax2.axhline(y=threshold_occ, color='#ffcc00', linestyle='--', linewidth=1.5, label='Batas Macet')
        
        ax2.set_title("Occupancy Historis", fontsize=14, pad=15)
        ax2.set_ylabel("Occupancy (%)") # Satuan Persen
        ax2.set_xlabel("interval")
        ax2.legend(loc='upper right', frameon=True, facecolor='white', labelcolor='black')
        ax2.grid(False)

        st.pyplot(fig)

        # --- LOGIKA STATUS & PREDIKSI ---
        st.divider()
        st.subheader("🏁 Status Data Terkini")

        # Logika Penentuan Status
        status_text = "LANCAR 🟢"
        warna_pesan = "success"

        if nilai_occ_saat_ini > threshold_occ:
            status_text = "MACET (Occupancy Tinggi) 🔴"
            warna_pesan = "error"
        elif nilai_flow_saat_ini > threshold_flow:
            status_text = "PADAT (Flow Tinggi) 🟠"
            warna_pesan = "warning"

        # Tampilkan Status
        if warna_pesan == "error":
            st.error(f"Status Lalu Lintas: **{status_text}**")
        elif warna_pesan == "warning":
            st.warning(f"Status Lalu Lintas: **{status_text}**")
        else:
            st.success(f"Status Lalu Lintas: **{status_text}**")

        # Tampilkan Angka Prediksi
        c1, c2, c3 = st.columns(3)
        
        c1.metric("Posisi Interval", posisi_saat_ini)
        
        c2.metric(
            label="Flow Terkini", 
            value=f"{nilai_flow_saat_ini:.2f}", 
            delta=f"{nilai_flow_saat_ini - threshold_flow:.2f} dari batas",
            delta_color="inverse" 
        )
        
        c3.metric(
            label="Occupancy Terkini", 
            value=f"{nilai_occ_saat_ini:.2f}%",  # Format Persen
            delta=f"{nilai_occ_saat_ini - threshold_occ:.2f}% dari batas",
            delta_color="inverse"
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.info("Tips: Pastikan file CSV Anda memiliki kolom angka untuk Flow dan Occupancy.")

else:
    st.info("👋 Silakan upload file CSV Birmingham/Traffic Anda.")
