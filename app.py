import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Simulasi Kemacetan", layout="wide")
plt.style.use('dark_background') # Tema Gelap

st.title("📈 Simulasi & Prediksi Kemacetan")
st.markdown("Cari status kemacetan berdasarkan Hari dan Waktu spesifik.")

# --- 2. SIDEBAR PANEL ---
st.sidebar.header("📂 Panel Kontrol")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # BACA DATA
        df = pd.read_csv(uploaded_file)
        
        # --- A. LOGIKA OTOMATIS CARI KOLOM (Tanpa Input User) ---
        all_cols = df.columns
        
        # 1. Cari Kolom Waktu (Otomatis)
        # Mencari kolom yang mengandung kata 'date', 'time', 'tgl', atau 'waktu'
        col_time = next((c for c in all_cols if any(x in c.lower() for x in ['date', 'time', 'tgl', 'waktu'])), None)
        
        if col_time is None:
            st.error("❌ Tidak dapat menemukan kolom Waktu/Tanggal otomatis. Pastikan ada kolom bernama 'date' atau 'time' di CSV.")
            st.stop()

        # Proses Konversi Waktu & Ekstrak Hari
        df['datetime_obj'] = pd.to_datetime(df[col_time], errors='coerce')
        df = df.dropna(subset=['datetime_obj']) # Hapus baris yang gagal convert
        
        # Tambahkan kolom 'Hari' (0=Senin, 6=Minggu) dan 'Total Detik'
        df['day_of_week'] = df['datetime_obj'].dt.dayofweek 
        df['total_seconds'] = df['datetime_obj'].dt.hour * 3600 + \
                              df['datetime_obj'].dt.minute * 60 + \
                              df['datetime_obj'].dt.second

        # 2. Cari Kolom Flow & Occupancy (Konfigurasi Manual agar Akurat)
        st.sidebar.subheader("⚙️ Konfigurasi Data")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        # Filter agar kolom hasil olahan tidak muncul
        numeric_cols_clean = [c for c in numeric_cols if c not in ['day_of_week', 'total_seconds']]
        
        col_flow = st.sidebar.selectbox("Pilih Kolom Flow:", numeric_cols_clean, index=0)
        idx_occ = 1 if len(numeric_cols_clean) > 1 else 0
        col_occ = st.sidebar.selectbox("Pilih Kolom Occupancy:", numeric_cols_clean, index=idx_occ)

        # --- B. INPUT HARI & WAKTU (SESUAI REQUEST) ---
        st.sidebar.divider()
        st.sidebar.subheader("🎚️ Input Spesifik")
        
        # 1. Input Hari (Senin - Minggu)
        hari_mapping = {
            "Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3, 
            "Jumat": 4, "Sabtu": 5, "Minggu": 6
        }
        pilihan_hari = st.sidebar.selectbox("Pilih Hari:", list(hari_mapping.keys()))
        target_day_idx = hari_mapping[pilihan_hari]

        # 2. Input Jam Detik Terpisah
        c1, c2, c3 = st.sidebar.columns(3)
        with c1: input_jam = st.number_input("Jam", 0, 23, 7)
        with c2: input_menit = st.number_input("Menit", 0, 59, 0)
        with c3: input_detik = st.number_input("Detik", 0, 59, 0)

        # --- C. ALGORITMA PENCARIAN DATA ---
        
        # Langkah 1: Filter data hanya ambil HARI yang dipilih (Misal: Hanya ambil data hari Senin)
        df_filtered = df[df['day_of_week'] == target_day_idx]

        if df_filtered.empty:
            st.warning(f"⚠️ Tidak ada data untuk hari {pilihan_hari} di file CSV ini.")
            st.stop()

        # Langkah 2: Di data hari tersebut, cari JAM yang paling mirip
        input_total_seconds = (input_jam * 3600) + (input_menit * 60) + input_detik
        
        # Cari selisih waktu terkecil
        # idxmin() akan mengembalikan index asli dari dataframe utama
        idx_ketemu = (df_filtered['total_seconds'] - input_total_seconds).abs().idxmin()
        
        # Ambil nilai Flow & Occupancy dari baris yang ditemukan
        nilai_flow = df.loc[idx_ketemu, col_flow]
        nilai_occ = df.loc[idx_ketemu, col_occ]
        waktu_aktual = df.loc[idx_ketemu, 'datetime_obj'].strftime("%H:%M:%S")
        tanggal_aktual = df.loc[idx_ketemu, 'datetime_obj'].strftime("%d-%m-%Y")

        # --- D. VISUALISASI ---
        
        # Hitung Threshold Otomatis
        max_flow = int(df[col_flow].max())
        max_occ = float(df[col_occ].max())
        thresh_flow = max_flow * 0.7
        thresh_occ = max_occ * 0.3

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Grafik 1: Flow
        # Kita plot seluruh data historis sebagai background
        ax1.plot(df.index, df[col_flow], color='white', alpha=0.15, label='Data Historis')
        # Titik Merah
        ax1.scatter(idx_ketemu, nilai_flow, color='red', s=150, zorder=10, label=f'{pilihan_hari}, {waktu_aktual}', edgecolors='white')
        ax1.axhline(thresh_flow, color='#ffcc00', linestyle='--', label='Batas Padat')
        ax1.set_title("Grafik Flow", fontsize=14, pad=10)
        ax1.legend(facecolor='#262730', labelcolor='white')
        ax1.grid(False)

        # Grafik 2: Occupancy
        ax2.plot(df.index, df[col_occ], color='white', alpha=0.15, label='Data Historis')
        ax2.scatter(idx_ketemu, nilai_occ, color='red', s=150, zorder=10, label=f'{pilihan_hari}, {waktu_aktual}', edgecolors='white')
        ax2.axhline(thresh_occ, color='#ffcc00', linestyle='--', label='Batas Macet')
        ax2.set_title("Grafik Occupancy", fontsize=14, pad=10)
        ax2.legend(facecolor='#262730', labelcolor='white')
        ax2.grid(False)

        st.pyplot(fig)

        # --- E. HASIL ANALISIS STATUS ---
        st.divider()
        st.subheader(f"🏁 Hasil Analisis: {pilihan_hari}, Pukul {waktu_aktual}")
        st.caption(f"Data diambil dari tanggal: {tanggal_aktual}")

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
        c1.metric("Waktu Ditemukan", f"{pilihan_hari}, {waktu_aktual}")
        c2.metric("Flow", f"{nilai_flow:.0f}", delta=f"{nilai_flow - thresh_flow:.0f} dr batas", delta_color="inverse")
        c3.metric("Occupancy", f"{nilai_occ:.2f}%", delta=f"{nilai_occ - thresh_occ:.2f}% dr batas", delta_color="inverse")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.warning("Tips: Pastikan file CSV memiliki kolom Waktu (format date/time) serta kolom Flow dan Occupancy.")

else:
    st.info("👋 Silakan upload file CSV Traffic Anda.")
