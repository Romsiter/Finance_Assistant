from crewai.tools import tool
from crewai import Agent, Task, Crew, Process
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key=os.getenv('OPENAI_API_KEY')

@tool("text_to_speech")
def text_to_speech(_: str = "") -> str:
    """
    Converts the content of 'final_answer.txt' to speech and saves it as 'output_tts.wav'.
    """
    try:
        with open("final_answer.txt", "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            return "❌ final_answer.txt is empty."

        # Load TTS model
        speech = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",  # or nova, echo, etc.
            input=text
        )
        # Generate and save speech
        speech.stream_to_file("output_tts.wav")

        return "✅ OpenAI TTS saved speech to 'output_tts.wav'"

    except FileNotFoundError:
        return "❌ 'final_answer.txt' not found."
    except Exception as e:
        return f"⚠️ Error during OpenAI TTS: {str(e)}"


tts_agent = Agent(
    role="Speech Synthesizer",
    goal="Convert the final written response into spoken audio",
    backstory="An AI voice assistant that speaks financial summaries aloud.",
    tools=[text_to_speech],
    verbose=True
)

tts_task = Task(
    description="Convert the final user-facing response into speech by reading final_answer.txt and generating an audio file.",
    expected_output="Audio saved as output_tts.wav",
    agent=tts_agent,
    inputs={},  # No dynamic input needed since tool reads from file
    output_file="tts_log.txt"
)