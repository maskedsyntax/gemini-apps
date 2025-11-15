import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

_ = load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# Function to load Gemini model and get response
def get_gemini_response(input, image):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[input, image],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
        ),
    )

    return response.text


# Initialize streamlit app
st.set_page_config(page_title="Basic")
st.header("Gemini Basic Image App")
input = st.text_input("Input: ", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

submit = st.button("Ask the question")

if submit:
    response = get_gemini_response(input, image)
    st.subheader("response:")
    st.write(response)
