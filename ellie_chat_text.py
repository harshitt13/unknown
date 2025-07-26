import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=API_KEY)

# --- Model Selection ---
MODEL_NAME = 'gemini-1.5-flash'  # Use stable model name
llm_model = genai.GenerativeModel(MODEL_NAME)

# --- Main Chat Logic ---
def run_text_chat():
    print("\n--- Ellie AI Text Chat ---")
    print("Start typing. Type 'exit' or 'goodbye' to end the conversation.")
    print("---------------------------\n")

    # Initialize the chat session with Ellie's persona
    chat = llm_model.start_chat(history=[
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

    print("Ellie: Heyyy, what's good, sir? Ellie here, ready to spill some tea or just chill. Like, literally, tell me what's up! ✨")

    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            # Check for exit commands
            if user_input.lower() in ['exit', 'goodbye', 'quit', 'bye']:
                farewell_response = chat.send_message("The user said goodbye. Please respond with a farewell.")
                print(f"Ellie: {farewell_response.text}")
                break

            try:
                # Send message to Ellie
                response = chat.send_message(user_input)
                print(f"Ellie: {response.text}")

            except Exception as e:
                print(f"Ellie (error): My bad, sir, something glitched during the conversation. (Error: {e})")
                print("If the error persists, check your internet connection or API key.")

    except KeyboardInterrupt:
        print("\nEllie: Looks like you dipped, sir! Later! 👋")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("Chat session ended.")

# --- Main Execution ---
if __name__ == "__main__":
    run_text_chat()
