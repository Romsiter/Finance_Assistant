# api/main.py
from mangum import Mangum

from fastapi import FastAPI, Depends, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import os
from crewai import Crew, Process
import importlib
import whisper
# --- top of main.py ---------------------------
import os, shutil, whisper
from fastapi import FastAPI, UploadFile, File, Depends
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent          # <project-root>

# Make absolutely sure ffmpeg.exe is reachable
FFMPEG_DIR = r"C:\Users\PRANAV BHARDWAJ\Downloads\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin"   # <-- put *your* bin path here
if FFMPEG_DIR not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + FFMPEG_DIR

# Quick sanity-check; remove in production
assert shutil.which("ffmpeg"), "ffmpeg still not found – PATH is: " + os.environ["PATH"]
app = FastAPI()
def get_whisper_model():
    model = whisper.load_model("base")
    return model
@app.post("/process_query")
async def process_query(audio: UploadFile = File(...)):
    # Save uploaded audio
    with open("audio.wav", "wb") as f:
        shutil.copyfileobj(audio.file, f)
        
    def transcribe_audio(audio_path: str, model=get_whisper_model()):       
        print(audio_path)
        result = model.transcribe(audio_path)
        return {"text": result["text"]}
  
  
    query = transcribe_audio("audio.wav")["text"]
    with open("query.txt", "w", encoding="utf-8") as f:
        f.write(query)
        
    mod_names = ["Agents.api_agent", "Agents.scraping_agent", "Agents.retriever_agent",
                 "Agents.analysis_agent", "Agents.language_agent", "Agents.voice_agent"]
    
    mods = [importlib.import_module(m) for m in mod_names]
    for m in mods: importlib.reload(m)
    stock_agent,   stock_task   = mods[0].stock_agent,   mods[0].stock_task
    scraper_agent, news_task    = mods[1].scraper_agent, mods[1].news_task
    retriever_agent,retrieval_task = mods[2].retriever_agent,mods[2].retrieval_task
    analysis_agent,analysis_task  = mods[3].analysis_agent, mods[3].analysis_task
    language_agent,synthesis_task = mods[4].language_agent, mods[4].synthesis_task
    tts_agent,     tts_task       = mods[5].tts_agent,      mods[5].tts_task
    # Run full pipeline as Crew
    
    crew = Crew(
        agents=[
            stock_agent,
            scraper_agent,
            retriever_agent,
            analysis_agent,
            language_agent,
            tts_agent,
        ],
        tasks=[
            stock_task,
            news_task,
            retrieval_task,
            analysis_task,
            synthesis_task,
            tts_task,
        ],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    print("🔄 Crew completed. Returning result...")
    wav_file = BASE_DIR / "tts_answer.wav"
    return FileResponse(wav_file, media_type="audio/wav", filename="response.wav")


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI"}

# 👇 This line exposes a Lambda-compatible handler
handler = Mangum(app)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost:8000", port=8000)
