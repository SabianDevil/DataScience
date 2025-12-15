import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Konfigurasi Halaman
st.set_page_config(page_title="Posisi Prediksi Data", layout="wide")

# Mengatur tema Matplotlib menjadi Gelap
plt.style.use('dark_background')

st.title("📈 Posisi Prediksi pada Data Historis")
st.markdown("Cari data kemacetan berdasarkan input Jam, Menit, dan Detik spesifik.")

# --- SIDEBAR ---
st.sidebar.header("📂 Panel Kontrol")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # 1. BACA DATA
        df = pd.read_csv(uploaded_file)
        
        # 2. KONFIGURASI KOLOM
        st.sidebar.subheader("⚙️ Konfigurasi Kolom")
        all_cols = df.columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        # A. Kolom Waktu
        default_time = next((i for i, c in enumerate(all_cols) if 'date' in c.lower() or 'time' in c.lower()), 0)
        col_time = st.sidebar.selectbox("Pilih Kolom Waktu/Tanggal:", all_cols, index=default_time)

        # Proses Konversi Waktu
        df['datetime_convert'] = pd.to_datetime(df[col_time], errors='coerce')
        df = df.dropna(subset=['datetime_convert']) 

        if df.empty:
            st.error("Gagal membaca kolom waktu.")
            st.stop()

        # B. Kolom Flow & Occupancy
        numeric_cols_clean = [c for c in numeric_cols if c != 'datetime_convert']
        col_flow = st.sidebar.selectbox("Pilih Kolom Flow:", numeric_cols_clean, index=0)
        idx_occ = 1 if len(numeric_cols_clean) > 1 else 0
        col_occ = st.sidebar.selectbox("Pilih Kolom Occupancy:", numeric_cols_clean, index=idx_occ)

        # 3. METODE INPUT WAKTU TERPISAH
        st.sidebar.divider()
        st.sidebar.subheader("🎚️ Input Waktu Spesifik")
        
        # Default value dari data awal
        waktu_awal = df['datetime_convert'].iloc[0]
        
        c_jam, c_menit, c_detik = st.sidebar.columns(3)
        with c_jam:
            input_jam = st.number_input("Jam", 0, 23, int(waktu_awal.hour))
        with c_menit:
            input_menit = st.number_input("Menit", 0, 59, int(waktu_awal.minute))
        with c_detik:
            input_detik = st.number_input("Detik", 0, 59, int(waktu_awal.second))

        # --- LOGIKA PENCARIAN DATA BARU (Total Detik) ---
        # 1. Hitung total detik dari input user (Jam * 3600 + Menit * 60 + Detik)
        user_total_seconds = (input_jam * 3600) + (input_menit * 60) + input_detik
        
        # 2. Hitung total detik untuk SETIAP baris data di CSV
        # (Kita abaikan tanggalnya, fokus ke jam-nya saja)
        df['seconds_from_midnight'] = df['datetime_convert'].dt.hour * 3600 + \
                                      df['datetime_convert'].dt.minute * 60 + \
                                      df['datetime_convert'].dt.second
        
        # 3. Cari selisih terkecil antara Input User vs Data
        # idxmin() akan mengambil index baris yang paling mirip jam-nya
        posisi_saat_ini = (df['seconds_from_midnight'] - user_total_seconds).abs().idxmin()
        
        # Ambil Waktu Aktual string untuk ditampilkan
        waktu_aktual_str = df.loc[posisi_saat_ini, 'datetime_convert'].strftime("%H:%M:%S")

        # 4. SET THRESHOLD OTOMATIS
        max_flow = int(df[col_flow].max())
        max_occ = float(df[col_occ].max())
        threshold_flow = max_flow * 0.7 
        threshold_occ = max_occ * 0.3

        # --- VISUALISASI MATPLOTLIB ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # === GRAFIK 1: FLOW ===
        ax1.plot(df.index, df[col_flow], color='white', alpha=0.15, label='Data Historis', linewidth=1)
        nilai_flow_saat_ini = df.iloc[posisi_saat_ini][col_flow]
        
        ax1.scatter(posisi_saat_ini, nilai_flow_saat_ini, color='red', s=120, zorder=5, label=f'Pukul {waktu_aktual_str}')
        ax1.axhline(y=threshold_flow, color='#ffcc00', linestyle='--', linewidth=1.5, label='Batas Padat')
        
        ax1.set_title("Grafik Flow", fontsize=14, pad=15)
        ax1.set_ylabel("Flow")
        ax1.legend(loc='upper right', frameon=True, facecolor='white', labelcolor='black')
        ax1.grid(False)

        # === GRAFIK 2: OCCUPANCY ===
        ax2.plot(df.index, df[col_occ], color='white', alpha=0.15, label='Data Historis', linewidth=1)
        nilai_occ_saat_ini = df.iloc[posisi_saat_ini][col_occ]
        
        ax2.scatter(posisi_saat_ini, nilai_occ_saat_ini, color='red', s=120, zorder=5, label=f'Pukul {waktu_aktual_str}')
        ax2.axhline(y=threshold_occ, color='#ffcc00', linestyle='--', linewidth=1.5, label='Batas Macet')
        
        ax2.set_title("Grafik Occupancy", fontsize=14, pad=15)
        ax2.set_ylabel("Occupancy (%)")
        ax2.legend(loc='upper right', frameon=True, facecolor='white', labelcolor='black')
        ax2.grid(False)

        st.pyplot(fig)

        # --- HASIL ANALISIS ---
        st.divider()
        st.subheader(f"🏁 Analisis Pukul {waktu_aktual_str}")

        # Logika Status
        status_text = "LANCAR 🟢"
        warna_pesan = "success"

        if nilai_occ_saat_ini > threshold_occ:
            status_text = "MACET (Occupancy Tinggi) 🔴"
            warna_pesan = "error"
        elif nilai_flow_saat_ini > threshold_flow:
            status_text = "PADAT (Flow Tinggi) 🟠"
            warna_pesan = "warning"

        if warna_pesan == "error":
            st.error(f"Status: **{status_text}**")
        elif warna_pesan == "warning":
            st.warning(f"Status: **{status_text}**")
        else:
            st.success(f"Status: **{status_text}**")

        # Metric Cards
        c1, c2, c3 = st.columns(3)
        c1.metric("Waktu Data Ditemukan", waktu_aktual_str)
        c2.metric("Flow", f"{nilai_flow_saat_ini:.0f}")
        c3.metric("Occupancy", f"{nilai_occ_saat_ini:.2f}%")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.info("Pastikan kolom Waktu berisi format yang benar.")

else:
    st.info("👋 Silakan upload file CSV Traffic Anda.")
