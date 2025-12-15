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
        
        # A. Kolom Waktu (Wajib dipilih untuk fitur Jam/Menit/Detik)
        # Cari default yang namanya mirip 'date' atau 'time'
        default_time = next((i for i, c in enumerate(all_cols) if 'date' in c.lower() or 'time' in c.lower()), 0)
        col_time = st.sidebar.selectbox("Pilih Kolom Waktu/Tanggal:", all_cols, index=default_time)

        # Proses Konversi Waktu (Penting!)
        # errors='coerce' akan mengubah data yang bukan waktu menjadi NaT (error), lalu kita buang (dropna)
        df['datetime_convert'] = pd.to_datetime(df[col_time], errors='coerce')
        df = df.dropna(subset=['datetime_convert']) 

        if df.empty:
            st.error("Gagal membaca kolom waktu. Pastikan format kolom yang dipilih adalah Tanggal/Jam.")
            st.stop()

        # B. Kolom Flow & Occupancy
        numeric_cols_clean = [c for c in numeric_cols if c != 'datetime_convert']
        col_flow = st.sidebar.selectbox("Pilih Kolom Flow:", numeric_cols_clean, index=0)
        # Cari default occupancy (biasanya kolom ke-2)
        idx_occ = 1 if len(numeric_cols_clean) > 1 else 0
        col_occ = st.sidebar.selectbox("Pilih Kolom Occupancy:", numeric_cols_clean, index=idx_occ)

        # 3. METODE INPUT WAKTU TERPISAH
        st.sidebar.divider()
        st.sidebar.subheader("🎚️ Input Waktu Spesifik")
        
        # Ambil waktu awal data sebagai default value agar tidak null
        waktu_awal = df['datetime_convert'].iloc[0]
        
        # Membuat 3 Kolom Input Berdampingan
        c_jam, c_menit, c_detik = st.sidebar.columns(3)
        
        with c_jam:
            input_jam = st.number_input("Jam", min_value=0, max_value=23, value=waktu_awal.hour)
        with c_menit:
            input_menit = st.number_input("Menit", min_value=0, max_value=59, value=waktu_awal.minute)
        with c_detik:
            input_detik = st.number_input("Detik", min_value=0, max_value=59, value=waktu_awal.second)

        # --- LOGIKA PENCARIAN DATA (Nearest Match) ---
        # 1. Buat object waktu dari input user
        waktu_user = datetime.time(input_jam, input_menit, input_detik)
        
        # 2. Gabungkan tanggal dari data pertama + waktu user
        # (Asumsi kita mencari jam tersebut pada hari yang ada di data)
        tanggal_ref = df['datetime_convert'].iloc[0].date()
        target_timestamp = pd.Timestamp.combine(tanggal_ref, waktu_user)
        
        # 3. Cari baris data yang selisih waktunya paling kecil
        posisi_saat_ini = (df['datetime_convert'] - target_timestamp).abs().idxmin()
        
        # Ambil data aktual yang ditemukan
        waktu_aktual_str = df.loc[posisi_saat_ini, 'datetime_convert'].strftime("%H:%M:%S")

        # 4. SET THRESHOLD OTOMATIS (Agar simple tanpa slider manual)
        max_flow = int(df[col_flow].max())
        max_occ = float(df[col_occ].max())
        
        threshold_flow = max_flow * 0.7 # 70% dari max dianggap batas padat
        threshold_occ = max_occ * 0.3   # 30% dari max dianggap batas macet

        # --- VISUALISASI MATPLOTLIB ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # === GRAFIK 1: FLOW ===
        ax1.plot(df.index, df[col_flow], color='white', alpha=0.15, label='Data Historis', linewidth=1)
        nilai_flow_saat_ini = df.iloc[posisi_saat_ini][col_flow]
        
        # Titik Merah
        ax1.scatter(posisi_saat_ini, nilai_flow_saat_ini, color='red', s=120, zorder=5, label=f'Input: {waktu_aktual_str}')
        # Garis Threshold
        ax1.axhline(y=threshold_flow, color='#ffcc00', linestyle='--', linewidth=1.5, label='Batas Padat')
        
        ax1.set_title("Grafik Flow", fontsize=14, pad=15)
        ax1.set_ylabel("Flow")
        ax1.legend(loc='upper right', frameon=True, facecolor='white', labelcolor='black')
        ax1.grid(False)

        # === GRAFIK 2: OCCUPANCY ===
        ax2.plot(df.index, df[col_occ], color='white', alpha=0.15, label='Data Historis', linewidth=1)
        nilai_occ_saat_ini = df.iloc[posisi_saat_ini][col_occ]
        
        ax2.scatter(posisi_saat_ini, nilai_occ_saat_ini, color='red', s=120, zorder=5, label=f'Input: {waktu_aktual_str}')
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

        # Tampilkan Status
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
        st.info("Tips: Pastikan kolom Waktu berisi format tanggal/jam yang benar (contoh: 2023-10-27 08:30:00).")

else:
    st.info("👋 Silakan upload file CSV Traffic Anda.")
