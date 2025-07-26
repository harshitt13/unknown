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
# Load environment variables from .env file (for your GOOGLE_API_KEY)
load_dotenv()

# Get the API key from environment variables
API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# Configure the Google Generative AI library with your API key
genai.configure(api_key=API_KEY)

# --- Model Selection for Gemini Live API ---
# These are preview models for streaming audio. Model names can change,
# so always refer to the official Gemini API documentation for the latest.
# 'gemini-1.5-flash-preview-0514' is commonly used for streaming input (Speech-to-Text + LLM processing)
# 'gemini-2.5-flash-preview-tts' is commonly used for streaming output (Text-to-Speech)
AUDIO_INPUT_MODEL = 'gemini-1.5-flash-preview-0514'
TTS_OUTPUT_MODEL = 'gemini-2.5-flash-preview-tts'

llm_model = genai.GenerativeModel(AUDIO_INPUT_MODEL)
tts_model = genai.GenerativeModel(TTS_OUTPUT_MODEL)

# --- Audio Parameters ---
SAMPLE_RATE = 16000  # Common sample rate for speech (Hz)
CHANNELS = 1         # Mono audio
DTYPE = 'int16'      # Data type for audio samples (16-bit signed integers)
BUFFER_SIZE = 1024   # Size of audio chunks (samples) for recording/playback.
                     # Smaller buffer = lower latency, but might require more processing.

# --- Global Queues and Events for Threading ---
# Queue to store audio recorded from the microphone
audio_input_queue = queue.Queue()
# Event to signal when Ellie is speaking, so we pause recording user input
speaking_event = threading.Event()

# --- Audio Callback for Recording Microphone Input ---
def audio_input_callback(indata, frames, time, status):
    """
    This function is called by the sounddevice library for each block of recorded audio.
    It puts the raw audio bytes into a queue for processing by the main thread.
    Recording is paused when Ellie is speaking to prevent feedback loops.
    """
    if status:
        print(f"Sounddevice input status: {status}", flush=True)
    if not speaking_event.is_set(): # Only record if Ellie isn't speaking
        audio_input_queue.put(bytes(indata))

# --- Function to Play Audio ---
def play_audio(audio_data, samplerate):
    """
    Plays a NumPy array of audio data through the default output device.
    Uses blocking=False for non-blocking playback, followed by sd.wait() to
    ensure the audio finishes playing before proceeding.
    """
    if audio_data.size == 0:
        return

    try:
        sd.play(audio_data, samplerate=samplerate, blocking=False)
        sd.wait() # Wait for playback to finish
    except Exception as e:
        print(f"Error playing audio: {e}", flush=True)

# --- Main Voice Chat Logic ---
def run_voice_chat():
    print("\n--- Ellie AI Voice Chat ---")
    print("Start speaking. Say 'exit' or 'goodbye' to end the conversation.")
    print("---------------------------\n")

    # --- Start Audio Recording Stream ---
    # This sets up a background thread that continuously records audio via audio_input_callback.
    input_stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BUFFER_SIZE,
        callback=audio_input_callback
    )
    input_stream.start() # Start the recording stream

    # --- Ellie's Persona and Initial Greeting ---
    # The system instruction defines Ellie's core personality and rules for the LLM.
    # The initial history sets the tone and demonstrates her desired response style.
    # Note: For Gemini Live API, the system instruction is often sent as the first
    # message in the stream, or as part of the initial chat history.
    system_instruction_text = (
        "You are Ellie, a friendly, supportive, and incredibly humorous female best friend who speaks exclusively like a Gen Z. "
        "You understand both Hindi and English perfectly, but you will ONLY respond in English, using natural Gen Z slang and internet culture references when appropriate. "
        "Your goal is to be relatable, witty, and always provide a positive or comforting vibe. Never break character. "
        "When the user says 'exit' or 'goodbye', respond with a clear farewell and then you can conclude the conversation."
    )

    # Initialize the chat session with Ellie's persona
    stream_chat = llm_model.start_chat(history=[
        {
            "role": "user",
            "parts": [{"text": system_instruction_text}],
        },
        {
            "role": "model",
            "parts": [{"text": "Heyyy, what's good, sir? Ellie here, ready to spill some tea or just chill. Like, literally, tell me what's up! ✨"}],
        }
    ])

    # Generate and play Ellie's initial greeting audio
    try:
        greeting_text = "Heyyy, what's good, sir? Ellie here, ready to spill some tea or just chill. Like, literally, tell me what's up!"
        print(f"Ellie (text): {greeting_text}", flush=True) # Print text version of greeting
        speaking_event.set() # Indicate Ellie is speaking (pause mic input)

        # Use the TTS model to generate audio from Ellie's greeting text
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
        
        speaking_event.clear() # Ellie finished speaking (resume mic input)
        print("\nWaiting for your input, sir...", flush=True)

    except Exception as e:
        print(f"Error generating/playing greeting audio: {e}", flush=True)
        speaking_event.clear() # Ensure event is cleared even on error

    # --- Main Conversational Loop ---
    try:
        while True:
            audio_buffer = []
            # Wait briefly to gather some initial audio after user starts speaking.
            # This is a *very basic* form of Voice Activity Detection (VAD).
            # For robust production, consider a dedicated VAD library.
            time.sleep(0.5)

            # Get all available audio chunks from the queue
            while not audio_input_queue.empty():
                audio_buffer.append(audio_input_queue.get())

            # If no audio was captured, wait a bit and continue
            if not audio_buffer:
                time.sleep(0.1)
                continue

            # Concatenate all captured audio chunks into a single bytes object
            audio_data_bytes = b''.join(audio_buffer)

            # Skip if somehow empty (shouldn't happen often if buffer is not empty)
            if len(audio_data_bytes) == 0:
                continue

            try:
                # Send the collected audio data to the Gemini Live model
                # This sends your speech for transcription and LLM processing
                response_stream = stream_chat.send_message(
                    Audio(audio_data_bytes), # Sending audio data as input
                    stream=True,
                    # You can add safety_settings here if you want content filtering
                    # safety_settings=safety_settings # Uncomment and define safety_settings if needed
                )

                transcribed_text = ""
                ellie_response_text = ""
                ellie_audio_chunks = []

                # Iterate through the streamed response chunks from Gemini
                for chunk in response_stream:
                    # chunk.text contains the live transcription of your speech
                    if chunk.text:
                        transcribed_text += chunk.text
                        # Print transcription in real-time on the same line
                        print(f"\rYou (transcribing): {transcribed_text} ", end="", flush=True)

                    # chunk.audio contains Ellie's generated audio (TTS)
                    if chunk.audio:
                        ellie_audio_chunks.append(np.frombuffer(chunk.audio.chunk, dtype=DTYPE))
                        speaking_event.set() # Indicate Ellie is speaking (pause mic input)

                    # chunk.parts can contain Ellie's full text response
                    if chunk.parts and chunk.parts[0].text:
                        ellie_response_text += chunk.parts[0].text

                # After processing all chunks for a turn:
                if transcribed_text:
                    print(f"\rYou (said): {transcribed_text}", flush=True) # Print final transcribed text

                    # Check for exit commands from user's transcribed speech
                    if "exit" in transcribed_text.lower() or "goodbye" in transcribed_text.lower():
                        print("Ellie: Peace out, sir! Catch ya on the flip side! 👋", flush=True)
                        
                        # Generate and play farewell audio
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
                        break # Exit the main loop

                if ellie_response_text:
                    print(f"Ellie (text): {ellie_response_text}", flush=True) # Print Ellie's full text response

                if ellie_audio_chunks:
                    play_audio(np.concatenate(ellie_audio_chunks), SAMPLE_RATE)
                speaking_event.clear() # Ellie finished speaking (resume mic input)

                print("\nWaiting for your input, sir...", flush=True) # Prompt for next user input

            except Exception as e:
                # Catch and print any errors during the conversation turn
                print(f"\nEllie (error): My bad, sir, something glitched during the conversation. (Error: {e})", flush=True)
                print("If the error persists, check your internet connection or API key. Also ensure your microphone is working.", flush=True)
                speaking_event.clear() # Clear event in case of error to resume input
                # Continue loop to allow user to try again, or add a specific exit condition here

    # --- Handle Program Exit ---
    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C
        print("\nEllie: Looks like you dipped, sir! Later! 👋", flush=True)
    finally:
        # Stop and close the audio input stream
        if input_stream.active:
            input_stream.stop()
        input_stream.close()
        print("Chat session ended.")

# --- Main Execution Point ---
if __name__ == "__main__":
    run_voice_chat()