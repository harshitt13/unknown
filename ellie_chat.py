import os
import queue
import threading
import time

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import sounddevice as sd
import numpy as np

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=API_KEY)

# --- Model Selection ---
# These are models that support text generation
# Note: Audio streaming features may not be available in all versions
AUDIO_INPUT_MODEL = 'gemini-1.5-flash'  # Use stable model name

llm_model = genai.GenerativeModel(AUDIO_INPUT_MODEL)

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
        # For now, just print the text greeting since TTS might not be available
        try:
            greeting_text = "Heyyy, what's good, sir? Ellie here, ready to spill some tea or just chill. Like, literally, tell me what's up!"
            print(f"Ellie (text): {greeting_text}", flush=True) # Print text version of greeting
            speaking_event.set() # Indicate Ellie is speaking, so mic input is paused
            
            # Note: TTS functionality might require different model or API approach
            # For now, we'll just use text responses
            
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
                # For now, we'll skip audio processing since the current API version
                # might not support real-time audio streaming as implemented here
                # Instead, we'll wait for text input or implement a different audio approach
                
                # Placeholder for audio processing - you could implement:
                # 1. Save audio to a file and send it as a file upload
                # 2. Use a speech-to-text service first, then send text
                # 3. Wait for Google to release the full audio streaming API
                
                print("Audio captured, but audio processing not fully implemented yet.")
                print("Please use the text version (ellie_chat_text.py) for now.")
                time.sleep(1)  # Brief pause before next iteration

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