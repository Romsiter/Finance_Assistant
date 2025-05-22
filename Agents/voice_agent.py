from crewai.tools import tool
from crewai import Agent, Task, Crew, Process
import openai
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
openai.api_key=os.getenv('OPENAI_API_KEY')

@tool("text_to_speech")
def text_to_speech(text: str | None = None) -> str:
    """
    Converts text (or final_answer.txt) to speech and saves as a WAV file.
    Returns the absolute path to the output file.
    """

    if text is None:
        try:
            script_dir = Path(__file__).resolve().parent
            with open(script_dir / "final_answer.txt", "r", encoding="utf-8") as fh:
                text = fh.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError("final_answer.txt not found next to script")

    if not text:
        raise ValueError("No text provided for TTS")

    # Create a unique output path
    out_name = f"tts_answer.wav"
    out_path = Path(__file__).resolve().parent / out_name

    # Generate speech
    speech = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    speech.stream_to_file(out_path)

    if not out_path.exists():
        raise RuntimeError("TTS generation failed: file not created")

    return str(out_path)

tts_agent = Agent(
    role="Speech Synthesizer",
    goal="Convert the final written response into spoken audio",
    backstory="An AI voice assistant that speaks financial summaries aloud.",
    tools=[text_to_speech],
    verbose=True
)

tts_task = Task(
    description="Convert the final user-facing response into speech by reading final_answer.txt and generating an audio file.",
    expected_output="Absolute path of the generated WAV file",
    agent=tts_agent,
    inputs={},  # No dynamic input needed since tool reads from file
    output_file="tts_log.txt"
)
    