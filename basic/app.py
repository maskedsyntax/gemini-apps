import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

_ = load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# Function to load Gemini model and get response
def get_gemini_response(question):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
        ),
    )

    return response


# Initialize streamlit app
st.set_page_config(page_title="Basic")
st.header("Gemini Basic App")
input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

if submit:
    response = get_gemini_response(input)
    st.subheader("response:")
    st.write(response.text)
