import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
        
        # 2. KONFIGURASI KOLOM
        st.sidebar.subheader("⚙️ Konfigurasi Kolom")
        
        # A. Kolom Waktu (PENTING untuk fitur Jam/Menit/Detik)
        # Cari kolom yang namanya mirip 'date' atau 'time' untuk default
        cols = df.columns
        default_time_idx = next((i for i, c in enumerate(cols) if 'date' in c.lower() or 'time' in c.lower()), 0)
        
        col_time = st.sidebar.selectbox("Pilih Kolom Waktu (Timestamp):", cols, index=default_time_idx)
        
        # Konversi kolom waktu ke format datetime yang bisa dibaca komputer
        # errors='coerce' akan mengubah data yang error menjadi NaT (Not a Time)
        df['datetime_convert'] = pd.to_datetime(df[col_time], errors='coerce')
        
        # Hapus baris yang gagal dikonversi waktunya (agar tidak error)
        df = df.dropna(subset=['datetime_convert'])
        
        # B. Kolom Flow & Occupancy
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        # Hapus kolom datetime_convert dari pilihan numeric biar bersih
        numeric_cols = [c for c in numeric_cols if c != 'datetime_convert']
        
        col_flow = st.sidebar.selectbox("Pilih Kolom Flow:", numeric_cols, index=0)
        col_occ = st.sidebar.selectbox("Pilih Kolom Occupancy:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

        # 3. KONTROL INPUT WAKTU (JAM/MENIT/DETIK)
        st.sidebar.divider()
        st.sidebar.subheader("🎚️ Input Waktu Simulasi")
        
        # Ambil waktu paling awal dan akhir dari data untuk referensi
        min_time = df['datetime_convert'].min().time()
        max_time = df['datetime_convert'].max().time()
        st.sidebar.caption(f"Rentang Data: {min_time} s/d {max_time}")

        # Widget Input Waktu
        # value=min_time agar defaultnya mulai dari awal data
        waktu_input = st.sidebar.time_input("Masukkan Jam, Menit, Detik:", value=min_time, step=1)
        
        # --- LOGIKA PENCARIAN WAKTU TERDEKAT ---
        # Kita harus mencari baris data mana yang jam-nya paling mirip dengan input user
        
        # 1. Ambil tanggal dari data pertama (asumsi data harian)
        tanggal_referensi = df['datetime_convert'].iloc[0].date()
        
        # 2. Gabungkan tanggal data + waktu input user
        target_timestamp = pd.Timestamp.combine(tanggal_referensi, waktu_input)
        
        # 3. Cari index dengan selisih waktu terkecil (Nearest Match)
        # (df['datetime_convert'] - target_timestamp).abs() menghitung jarak setiap data ke input user
        # .idxmin() mengambil index yang jaraknya paling dekat (0 detik atau milidetik)
        posisi_saat_ini = (df['datetime_convert'] - target_timestamp).abs().idxmin()

        # --- SLIDER THRESHOLD ---
        st.sidebar.divider()
        st.sidebar.subheader("⚠️ Batas Ambang (Threshold)")
        
        max_flow = int(df[col_flow].max())
        max_occ = float(df[col_occ].max())
        
        threshold_flow = st.sidebar.slider("Batas Macet (Flow)", 0, max_flow, int(max_flow * 0.7))
        threshold_occ = st.sidebar.slider("Batas Macet (Occupancy)", 0.0, max_occ, max_occ * 0.3)

        # --- VISUALISASI MATPLOTLIB ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # === GRAFIK 1: FLOW ===
        # Kita plot index sebagai sumbu X agar grafik tetap rapi
        ax1.plot(df.index, df[col_flow], color='white', alpha=0.15, label='Data Historis', linewidth=1)
        nilai_flow_saat_ini = df.iloc[posisi_saat_ini][col_flow]
        ax1.scatter(posisi_saat_ini, nilai_flow_saat_ini, color='red', s=100, zorder=5, label='Prediksi Saat Ini')
        ax1.axhline(y=threshold_flow, color='#ffcc00', linestyle='--', linewidth=1.5, label='Threshold Macet')
        
        ax1.set_title("Flow Historis dengan Prediksi Saat Ini", fontsize=14, pad=15)
        ax1.set_ylabel("Flow")
        ax1.set_xlabel("interval (urutan waktu)")
        ax1.legend(loc='upper right', frameon=True, facecolor='white', labelcolor='black')
        ax1.grid(False)

        # === GRAFIK 2: OCCUPANCY ===
        ax2.plot(df.index, df[col_occ], color='white', alpha=0.15, label='Data Historis', linewidth=1)
        nilai_occ_saat_ini = df.iloc[posisi_saat_ini][col_occ]
        ax2.scatter(posisi_saat_ini, nilai_occ_saat_ini, color='red', s=100, zorder=5, label='Prediksi Saat Ini')
        ax2.axhline(y=threshold_occ, color='#ffcc00', linestyle='--', linewidth=1.5, label='Threshold Macet')
        
        ax2.set_title("Occupancy Historis dengan Prediksi Saat Ini", fontsize=14, pad=15)
        ax2.set_ylabel("Occupancy (%)")
        ax2.set_xlabel("interval (urutan waktu)")
        ax2.legend(loc='upper right', frameon=True, facecolor='white', labelcolor='black')
        ax2.grid(False)

        st.pyplot(fig)

        # --- LOGIKA STATUS & PREDIKSI ---
        st.divider()
        st.subheader("🏁 Hasil Analisis Status")

        # Ambil waktu aktual dari data yang ditemukan
        waktu_aktual_data = df.iloc[posisi_saat_ini]['datetime_convert'].strftime("%H:%M:%S")

        status_text = "LANCAR 🟢"
        warna_pesan = "success"

        if nilai_occ_saat_ini > threshold_occ:
            status_text = "MACET (Occupancy Tinggi) 🔴"
            warna_pesan = "error"
        elif nilai_flow_saat_ini > threshold_flow:
            status_text = "PADAT (Flow Tinggi) 🟠"
            warna_pesan = "warning"

        if warna_pesan == "error":
            st.error(f"Status Pukul **{waktu_aktual_data}**: **{status_text}**")
        elif warna_pesan == "warning":
            st.warning(f"Status Pukul **{waktu_aktual_data}**: **{status_text}**")
        else:
            st.success(f"Status Pukul **{waktu_aktual_data}**: **{status_text}**")

        # Tampilkan Angka Prediksi
        c1, c2, c3 = st.columns(3)
        
        c1.metric("Waktu Terpilih", waktu_aktual_data)
        
        c2.metric(
            label="Prediksi Flow", 
            value=f"{nilai_flow_saat_ini:.2f}", 
            delta=f"{nilai_flow_saat_ini - threshold_flow:.2f} dari batas",
            delta_color="inverse" 
        )
        
        c3.metric(
            label="Prediksi Occupancy", 
            value=f"{nilai_occ_saat_ini:.2f}%", 
            delta=f"{nilai_occ_saat_ini - threshold_occ:.2f}% dari batas",
            delta_color="inverse"
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.info("Tips: Pastikan file CSV Anda memiliki kolom Waktu/Tanggal yang valid, serta kolom angka Flow dan Occupancy.")

else:
    st.info("👋 Silakan upload file CSV Birmingham/Traffic Anda.")
