import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tsfel
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import io

# --- 1. Konfigurasi dan Pemuatan Model ---
try:
    # Model dan Scaler HARUS dilatih dengan set fitur TSFEL 'statistical' yang sama.
    model = joblib.load('model_randomforest_tsfel.pkl')
    scaler = joblib.load('scaler_tsfel.pkl')
    st.sidebar.success("Model dan Scaler berhasil dimuat.")
except FileNotFoundError:
    st.sidebar.error("File model atau scaler tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Gagal memuat model/scaler: {e}")
    st.stop()

# --- 2. Load konfigurasi TSFEL (HANYA STATISTICAL) ---
# MEMENUHI PERSYARATAN TUGAS KULIAH: HANYA MENGGUNAKAN DOMAIN STATISTICAL
try:
    cfg = tsfel.get_features_by_domain('statistical') 
    st.sidebar.info(f"TSFEL Features: Hanya {len(cfg.keys())} fitur statistik yang dimuat.")
except Exception as e:
    st.sidebar.error(f"Gagal memuat konfigurasi TSFEL: {e}")
    st.stop()

# --- PARAMETER PENTING UNTUK KONSISTENSI SINYAL ---
# Definisikan target durasi sinyal (dalam detik)
# Sinyal saat pelatihan model HARUS memiliki panjang yang sama dengan ini!
TARGET_DURATION_SECONDS = 2 

# --- 3. UI Streamlit Utama ---
st.title("🎙️ Voice Command Recognition - Buka/Tutup")
st.markdown("""
**Mode Tugas Kuliah:** Prediksi menggunakan **HANYA** fitur statistik TSFEL.

Untuk meningkatkan akurasi, pastikan Anda telah **melatih ulang** model Anda dengan set fitur statistik yang sama.
""")

# --- 4. Pilihan input ---
st.subheader("🎧 Pilih metode input suara:")
input_choice = st.radio("Pilih salah satu:", ["🎤 Rekam Langsung", "📂 Upload File"], horizontal=True)

audio_bytes = None

# --- 5. Mode rekam suara ---
if input_choice == "🎤 Rekam Langsung":
    st.info("Tekan tombol di bawah untuk mulai rekam, lalu tekan lagi untuk berhenti. Rekam suara dengan jelas.")
    audio_data = mic_recorder(
        start_prompt="Mulai Rekam 🎙️",
        stop_prompt="Berhenti ⏹️",
        just_once=False,
        use_container_width=True,
    )

    if audio_data and audio_data["bytes"]:
        audio_bytes = audio_data["bytes"]
        st.audio(audio_bytes, format='audio/wav')

# --- 6. Mode upload file ---
elif input_choice == "📂 Upload File":
    uploaded_file = st.file_uploader("Unggah file suara (.wav / .mp3)", type=["wav", "mp3"])
    if uploaded_file:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format=uploaded_file.type)

# --- 7. Jika ada data audio, lakukan prediksi ---
if audio_bytes:
    with st.spinner("Memproses dan memprediksi..."):
        # Load audio data using pydub from bytes
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            sr = audio.frame_rate
            
            # Konversi AudioSegment ke numpy array (Penting untuk float32)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            
            # Normalisasi sinyal (Penting untuk konsistensi)
            signal = samples / np.max(np.abs(samples))

        except Exception as e:
            st.error(f"Gagal membaca file audio atau konversi: {e}")
            st.stop()
        
        # --- LANGKAH OPTIMASI UTAMA: PENYAMAAN PANJANG SINYAL ---
        target_samples = TARGET_DURATION_SECONDS * sr
        
        if len(signal) > target_samples:
            # Potong (Truncate) sinyal jika terlalu panjang
            signal = signal[:target_samples]
            st.warning(f"Sinyal dipotong menjadi {TARGET_DURATION_SECONDS} detik.")
        elif len(signal) < target_samples:
            # Tambahkan zero-padding jika terlalu pendek
            padding_needed = target_samples - len(signal)
            signal = np.pad(signal, (0, padding_needed), 'constant')
            st.warning(f"Sinyal ditambahkan zero-padding hingga {TARGET_DURATION_SECONDS} detik.")
        
        # Ekstraksi fitur TSFEL
        try:
            # TSFEL bekerja lebih baik jika data berbentuk 1 dimensi (series)
            df_features = tsfel.time_series_features_extractor(cfg, signal, fs=sr)
            
            # Pengecekan KONSISTENSI DIMENSI (PENTING untuk Model)
            if df_features.shape[1] != scaler.n_features_in_:
                st.error(f"""
                Kesalahan Dimensi Fitur!
                Model Anda dilatih dengan {scaler.n_features_in_} fitur, tetapi aplikasi
                mengekstraksi {df_features.shape[1]} fitur.
                
                Ini adalah penyebab akurasi rendah. Pastikan konfigurasi 'statistical' 
                Anda dan data pelatihan memiliki jumlah fitur yang sama!
                """)
                st.stop()
                
            X_new = df_features.values

            # Normalisasi
            X_new_scaled = scaler.transform(X_new)

            # Prediksi
            prediction = model.predict(X_new_scaled)
            proba = model.predict_proba(X_new_scaled)[0]

            label = prediction[0]
            
            # Mengambil confidence dari label yang diprediksi
            class_labels = model.classes_
            predicted_index = list(class_labels).index(label)
            confidence = proba[predicted_index] * 100

            # Hasil prediksi
            st.balloons()
            st.success(f"### 🎉 Prediksi Berhasil!")
            st.markdown(f"**🎯 Perintah Dikenali:** `<span style='font-size: 2.5rem; color: #4CAF50;'>{label.upper()}</span>`", unsafe_allow_html=True)
            st.markdown(f"**Tingkat Keyakinan (Confidence):** `{confidence:.2f}%`")

        except Exception as e:
            st.error(f"Terjadi kesalahan selama ekstraksi fitur atau prediksi. Pastikan data pelatihan konsisten. Error: {e}")

else:
    st.info("Silakan rekam atau unggah file suara terlebih dahulu untuk diprediksi.")
