import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Kemacetan", layout="wide")
plt.style.use('dark_background')

st.title("📈 Simulasi Prediksi Kemacetan")
st.markdown("Input jam berapa saja, sistem akan mencari pola data yang sesuai di posisi tersebut.")

# --- 2. SIDEBAR PANEL ---
st.sidebar.header("📂 Panel Kontrol")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # BACA DATA
        df = pd.read_csv(uploaded_file)
        
        # --- KONFIGURASI MANUAL (Hanya Flow & Occupancy) ---
        st.sidebar.subheader("⚙️ Konfigurasi Kolom")
        
        # Kita ambil semua kolom angka
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        # User pilih manual kolom mana yang Flow, mana yang Occupancy
        col_flow = st.sidebar.selectbox("Pilih Kolom Flow:", numeric_cols, index=0)
        # Otomatis coba pilih index ke-1 untuk occupancy
        idx_occ = 1 if len(numeric_cols) > 1 else 0
        col_occ = st.sidebar.selectbox("Pilih Kolom Occupancy:", numeric_cols, index=idx_occ)

        # --- INPUT WAKTU (JAM MENIT DETIK) ---
        st.sidebar.divider()
        st.sidebar.subheader("🎚️ Input Waktu Simulasi")
        
        c1, c2, c3 = st.sidebar.columns(3)
        with c1: input_jam = st.number_input("Jam", 0, 23, 12)   # Default jam 12
        with c2: input_menit = st.number_input("Menit", 0, 59, 0)
        with c3: input_detik = st.number_input("Detik", 0, 59, 0)

        # --- LOGIKA MAPPING (TRIK TEMAN ANDA) ---
        # Mengubah Input Waktu menjadi Posisi Baris Data (Index)
        
        # 1. Hitung total detik dari input user (0 s/d 86400)
        input_total_seconds = (input_jam * 3600) + (input_menit * 60) + input_detik
        total_seconds_in_day = 24 * 3600
        
        # 2. Hitung Persentase Waktu (Contoh: Jam 12 siang = 50%)
        ratio_waktu = input_total_seconds / total_seconds_in_day
        
        # 3. Terapkan ratio ke jumlah total baris data
        # Jika data ada 1000 baris, jam 12 siang berarti baris ke-500
        total_rows = len(df)
        target_index = int(ratio_waktu * (total_rows - 1))
        
        # Pastikan tidak error kalau index kebablasan
        target_index = max(0, min(target_index, total_rows - 1))

        # --- AMBIL DATA ---
        nilai_flow = df.iloc[target_index][col_flow]
        nilai_occ = df.iloc[target_index][col_occ]

        # --- HITUNG THRESHOLD (BATAS MACET) OTOMATIS ---
        max_flow = int(df[col_flow].max())
        max_occ = float(df[col_occ].max())
        
        # Batas Macet = 30% dari nilai tertinggi Occupancy
        thresh_occ = max_occ * 0.3
        # Batas Padat = 70% dari nilai tertinggi Flow
        thresh_flow = max_flow * 0.7

        # --- VISUALISASI ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Formatting String Waktu untuk Label
        waktu_str = f"{input_jam:02d}:{input_menit:02d}:{input_detik:02d}"

        # Grafik 1: Flow
        ax1.plot(df.index, df[col_flow], color='white', alpha=0.15, label='Data Historis')
        ax1.scatter(target_index, nilai_flow, color='red', s=150, zorder=10, label=f'Pukul {waktu_str}', edgecolors='white')
        ax1.axhline(thresh_flow, color='#ffcc00', linestyle='--', label='Batas Padat')
        ax1.set_title("Grafik Flow", fontsize=14, pad=10)
        ax1.legend(facecolor='#262730', labelcolor='white')
        ax1.grid(False)

        # Grafik 2: Occupancy
        ax2.plot(df.index, df[col_occ], color='white', alpha=0.15, label='Data Historis')
        ax2.scatter(target_index, nilai_occ, color='red', s=150, zorder=10, label=f'Pukul {waktu_str}', edgecolors='white')
        ax2.axhline(thresh_occ, color='#ffcc00', linestyle='--', label='Batas Macet')
        ax2.set_title("Grafik Occupancy", fontsize=14, pad=10)
        ax2.legend(facecolor='#262730', labelcolor='white')
        ax2.grid(False)

        st.pyplot(fig)

        # --- HASIL STATUS ---
        st.divider()
        st.subheader(f"🏁 Analisis Pukul {waktu_str}")
        
        # Logika Status
        status = "LANCAR 🟢"
        warna = "success"

        if nilai_occ > thresh_occ:
            status = "MACET (Occupancy Tinggi) 🔴"
            warna = "error"
        elif nilai_flow > thresh_flow:
            status = "PADAT (Flow Tinggi) 🟠"
            warna = "warning"

        if warna == "error": st.error(f"Status: **{status}**")
        elif warna == "warning": st.warning(f"Status: **{status}**")
        else: st.success(f"Status: **{status}**")

        # Metric Cards
        c1, c2, c3 = st.columns(3)
        c1.metric("Waktu Simulasi", waktu_str)
        c2.metric("Prediksi Flow", f"{nilai_flow:.0f}", delta=f"{nilai_flow - thresh_flow:.0f} dr batas", delta_color="inverse")
        c3.metric("Prediksi Occupancy", f"{nilai_occ:.2f}%", delta=f"{nilai_occ - thresh_occ:.2f}% dr batas", delta_color="inverse")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.info("Pastikan kolom Flow dan Occupancy yang dipilih benar.")

else:
    st.info("👋 Silakan upload file CSV Traffic Anda.")
