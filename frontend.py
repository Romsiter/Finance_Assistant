# streamlit_app.py

import streamlit as st
import sounddevice as sd
import soundfile as sf
import requests
from scipy.io.wavfile import write
import wavio as wv

st.set_page_config(page_title="Voice Finance Assistant", layout="centered")

st.title("🎙️ Voice-Based Financial Assistant")

# 1. Record Audio
st.header("Record your Query")
duration = st.slider("Recording duration (seconds)", 3, 30, 5)

if st.button("🎤 Record Audio"):
    fs = 44100
    st.info("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write("audio.wav", recording, fs)  # ✅ RIGHT
    st.success("Done! Saved as audio.wav")

# 2. Trigger full backend pipeline
if st.button("🧠 Get Market Brief"):
    with open("audio.wav", "rb") as f:
        files = {"audio": ("audio.wav", f, "audio/wav")}
        response = requests.post("http://127.0.0.1:8000/process_query", files=files)

    if response.status_code == 200:
        with open("response.wav", "wb") as out:
            out.write(response.content)
        st.success("✅ Got your result!")
        st.audio("response.wav")
    else:
        st.error("❌ Something went wrong")
