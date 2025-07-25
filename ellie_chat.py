import os
import queue
import threading
import time

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, Audio
import sounddevice as sd
import numpy as np

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=API_KEY)

# --- Model Selection for Live API ---
# These are preview models for streaming audio. Always check the official docs for the latest.
# 'gemini-1.5-flash-preview-0514' is for input (speech-to-text + LLM)
# 'gemini-2.5-flash-preview-tts' is for output (text-to-speech)
# Note: As of early 2025, these model names are common. They might evolve.
AUDIO_INPUT_MODEL = 'gemini-1.5-flash-preview-0514'
TTS_OUTPUT_MODEL = 'gemini-2.5-flash-preview-tts'

llm_model = genai.GenerativeModel(AUDIO_INPUT_MODEL)
tts_model = genai.GenerativeModel(TTS_OUTPUT_MODEL)

# --- Audio Parameters ---
SAMPLE_RATE = 16000  # Common sample rate for speech
CHANNELS = 1         # Mono audio
DTYPE = 'int16'      # Data type for audio samples (16-bit integers)
BUFFER_SIZE = 1024   # Size of audio chunks for recording/playback

# --- Global Queue for Audio (to pass between recording thread and main thread) ---
audio_input_queue = queue.Queue()
speaking_event = threading.Event() # Event to signal when Ellie is speaking

# --- Audio Callback for Recording ---
def audio_input_callback(indata, frames, time, status):
    """This function is called by sounddevice for each block of recorded audio."""
    if status:
        print(f"Sounddevice status: {status}", flush=True) # Use flush=True for real-time output
    if not speaking_event.is_set(): # Only record if Ellie isn't speaking
        audio_input_queue.put(bytes(indata))

# --- Function to play audio ---
def play_audio(audio_data, samplerate):
    """Plays a numpy array of audio data."""
    if audio_data.size == 0:
        return
    try:
        sd.play(audio_data, samplerate=samplerate, blocking=False) # Non-blocking playback
        sd.wait() # Wait for playback to finish
    except Exception as e:
        print(f"Error playing audio: {e}", flush=True)


# --- Main Voice Chat Logic ---
def run_voice_chat():
    print("\n--- Ellie AI Voice Chat ---")
    print("Start speaking. Say 'exit' or 'goodbye' to end the conversation.")
    print("---------------------------\n")

    # Start audio recording stream in a separate thread
    input_stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BUFFER_SIZE,
        callback=audio_input_callback
    )
    input_stream.start()
    print("Ellie: Heyyy, what's good, sir? Ellie here, ready to spill some tea or just chill. Like, literally, tell me what's up! ✨")

    try:
        # Initialize the chat session with Ellie's persona
        # The first message sets the system instruction
        stream_chat = llm_model.start_chat(history=[
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "You are Ellie, a friendly, supportive, and incredibly humorous female best friend who speaks exclusively like a Gen Z. "
                            "You understand both Hindi and English perfectly, but you will ONLY respond in English, using natural Gen Z slang and internet culture references when appropriate. "
                            "Your goal is to be relatable, witty, and always provide a positive or comforting vibe. Never break character. "
                            "When the user says 'exit' or 'goodbye', respond with a clear farewell and then you can conclude the conversation."
                        )
                    }
                ],
            },
            {
                "role": "model",
                "parts": [
                    {
                        "text": "Heyyy, what's good, sir? Ellie here, ready to spill some tea or just chill. Like, literally, tell me what's up! ✨"
                    }
                ]
            }
        ])

        # Generate the initial greeting audio for Ellie
        # Use a text-to-speech model
        try:
            greeting_text = "Heyyy, what's good, sir? Ellie here, ready to spill some tea or just chill. Like, literally, tell me what's up!"
            print(f"Ellie (text): {greeting_text}", flush=True) # Print text version of greeting
            speaking_event.set() # Indicate Ellie is speaking, so mic input is paused
            greeting_audio_response = tts_model.generate_content(
                genai.types.TextPart(text=greeting_text),
                stream=True
            )
            audio_chunks = []
            for chunk in greeting_audio_response:
                if chunk.audio:
                    audio_chunks.append(np.frombuffer(chunk.audio.chunk, dtype=DTYPE))
            if audio_chunks:
                play_audio(np.concatenate(audio_chunks), SAMPLE_RATE)
            speaking_event.clear() # Ellie finished speaking, resume mic input
        except Exception as e:
            print(f"Error generating/playing greeting audio: {e}", flush=True)

        print("\nWaiting for your input, sir...", flush=True)


        while True:
            # Get audio from the queue (recorded by the callback)
            # This loop is simplified; in a robust app, you'd manage conversation turns
            # with explicit input/output cycles.
            audio_buffer = []
            # Wait briefly for some audio to accumulate or for user to speak
            # You'll need more sophisticated VAD (Voice Activity Detection) here
            # for a truly natural conversation flow.
            time.sleep(0.5) # Small delay to gather some initial audio

            while not audio_input_queue.empty():
                audio_buffer.append(audio_input_queue.get())

            if not audio_buffer:
                time.sleep(0.1) # Wait a bit if no audio
                continue

            audio_data_bytes = b''.join(audio_buffer)

            if len(audio_data_bytes) == 0:
                continue

            try:
                # Send the audio data to the Gemini Live model
                # This is the core of the real-time interaction
                response_stream = stream_chat.send_message(
                    Audio(audio_data_bytes),
                    stream=True
                    # safety_settings=safety_settings # Add safety settings if desired, as above
                )

                transcribed_text = ""
                ellie_response_text = ""
                ellie_audio_chunks = []

                for chunk in response_stream:
                    # Process text (STT result) and audio (TTS result) from the stream
                    if chunk.text:
                        # This is the transcription of YOUR speech
                        transcribed_text += chunk.text
                        print(f"\rYou (transcribing): {transcribed_text}", end="", flush=True) # Real-time transcription

                    if chunk.audio:
                        # This is Ellie's generated audio response
                        ellie_audio_chunks.append(np.frombuffer(chunk.audio.chunk, dtype=DTYPE))
                        speaking_event.set() # Indicate Ellie is speaking

                    if chunk.parts and chunk.parts[0].text:
                        # This is Ellie's full text response (might come after initial audio)
                        ellie_response_text += chunk.parts[0].text

                if transcribed_text:
                    print(f"\rYou (said): {transcribed_text}", flush=True) # Final transcribed text

                    # Check for exit commands
                    if "exit" in transcribed_text.lower() or "goodbye" in transcribed_text.lower():
                        print("Ellie: Peace out, sir! Catch ya on the flip side! 👋", flush=True)
                        # Generate farewell audio
                        farewell_audio_response = tts_model.generate_content(
                            genai.types.TextPart(text="Peace out, sir! Catch ya on the flip side!"),
                            stream=True
                        )
                        audio_chunks = []
                        for chunk in farewell_audio_response:
                            if chunk.audio:
                                audio_chunks.append(np.frombuffer(chunk.audio.chunk, dtype=DTYPE))
                        if audio_chunks:
                            play_audio(np.concatenate(audio_chunks), SAMPLE_RATE)
                        break # Exit the loop

                if ellie_response_text:
                    print(f"Ellie (text): {ellie_response_text}", flush=True) # Print Ellie's full text response

                if ellie_audio_chunks:
                    play_audio(np.concatenate(ellie_audio_chunks), SAMPLE_RATE)
                speaking_event.clear() # Ellie finished speaking

                print("\nWaiting for your input, sir...", flush=True) # Prompt for next input

            except Exception as e:
                print(f"\nEllie (error): My bad, sir, something glitched during the conversation. (Error: {e})", flush=True)
                print("If the error persists, check your internet connection or API key.", flush=True)
                speaking_event.clear() # Clear event in case of error
                # Continue loop to allow user to try again

    except KeyboardInterrupt:
        print("\nEllie: Looks like you dipped, sir! Later! 👋", flush=True)
    finally:
        if input_stream.active:
            input_stream.stop()
        input_stream.close()
        print("Chat session ended.")

# --- Main Execution ---
if __name__ == "__main__":
    run_voice_chat()