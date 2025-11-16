import json
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

_ = load_dotenv()

# Initialize Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load history from file for cross-session persistence
HISTORY_FILE = "chat_history.json"


def load_history_from_file():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_history_to_file(history: list[tuple[str, str]]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# Session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history_from_file()
    if st.session_state.chat_history:
        st.toast("Loaded previous chat from file", icon="info")


loaded_history = load_history_from_file()
if loaded_history:
    st.session_state.chat_history = loaded_history
    st.info("Loaded previous chat history from file.")


# Function to build Gemini contents from history
def build_contents(history):
    """Convert chat_history to list of types.Content for Gemini"""
    contents = []
    for role, text in history:
        content = types.Content(
            role="user" if role == "You" else "model", parts=[types.Part(text=text)]
        )
        contents.append(content)

    return contents


# chat = client.chats.create(model="gemini-2.5-flash")


# Function to load Gemini model and get response
def get_gemini_response(user_input):
    # Append user message to history
    st.session_state.chat_history.append(("You", user_input))

    # Build full contents (history + current input)
    contents = build_contents(st.session_state.chat_history)

    try:
        response = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=1024),
        )

        full_response = ""
        placeholder = st.empty()
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                placeholder.markdown(full_response + "▌")

        # Save bot response to history
        st.session_state.chat_history.append(("Bot", full_response.strip()))
        return full_response

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, something went wrong."


# -------------------------------------------------
# UI
# -------------------------------------------------
st.set_page_config(page_title="Gemini Chat – Persistent", layout="centered")
st.title("Gemini Chat")

# ---- Show previous messages -------------------------------------------------
for role, text in st.session_state.chat_history:
    with st.chat_message("user" if role == "You" else "assistant"):
        st.markdown(text)

# ---- Input -------------------------------------------------
if prompt := st.chat_input("Ask Gemini…"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            get_gemini_response(prompt)


# ---- Sidebar controls -------------------------------------------------
with st.sidebar:
    st.header("Controls")

    if st.button("Clear Chat", type="primary"):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.session_state.chat_history = []
        st.success("Chat cleared – file removed")
        st.rerun()

    if st.button("Save History Now"):
        save_history_to_file(st.session_state.chat_history)
        st.success("Saved to `chat_history.json`")

    if st.button("Load History from File"):
        fresh = load_history_from_file()
        st.session_state.chat_history = fresh
        st.success("History loaded!")
        st.rerun()

    st.caption(
        f"**File:** `{HISTORY_FILE}`  \n"
        f"**Messages:** {len(st.session_state.chat_history)}"
    )
