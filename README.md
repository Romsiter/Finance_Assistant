# 🧠 Voice-Based Finance Assistant

An AI-powered assistant that lets users ask financial questions by voice. The system transcribes audio using Whisper, performs intelligent financial data retrieval and analysis using autonomous agents, and replies back with synthesized speech using OpenAI TTS.

---

## 🚀 Features

- 🎙️ Voice input via microphone (Streamlit frontend)
- 📝 Automatic transcription using Whisper
- 🧠 CrewAI agents that retrieve market data, run analysis, and generate a natural language summary
- 🔈 TTS audio response using OpenAI's `tts-1` model
- 🌐 FastAPI backend for handling the pipeline
- 🧩 Modular, agent-based architecture for extensibility

---

## 🧑‍💼 Use Case: Financial Voice Assistant

This assistant is built for retail investors or analysts who want spoken updates on stocks, economic indicators, or news summaries without typing.

---

## 🛠️ CrewAI Agent Roles

| Agent           | Purpose |
|-----------------|---------|
| **Stock Agent** | Retrieves stock price and performance data |
| **Scraper Agent** | Collects financial headlines or news |
| **Retriever Agent** | Finds relevant context using vector databases |
| **Analysis Agent** | Synthesizes all info into coherent insights |
| **Language Agent** | Refines language for clarity and quality |
| **TTS Agent** | Converts the final summary into spoken audio (`.wav` format) |

---

## 🧩 Workflow Overview

1. 🎤 User records a voice query using the Streamlit frontend.
2. 📤 Audio is sent to FastAPI `/process_query` endpoint.
3. 🔊 Whisper transcribes the audio to text.
4. 🧠 CrewAI agents collaborate to understand, retrieve, and synthesize financial data.
5. 📄 The final summary is written to `final_answer.txt`.
6. 🔈 The TTS Agent reads it and creates `tts_xxx.wav`.
7. 📥 The frontend plays the response as audio.

---

## 📦 Installation

### 1. Clone the repo
```bash
git clone https://github.com/Romsiter/Finance_Assistant.git
cd Finance_Assistant
