import json
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

_ = load_dotenv()

# Initialize Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# Function to load Gemini model and get response
def get_gemini_response(input, image, prompt):
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[input, image, prompt],
        config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=3072),
    )

    return response.text


# Initialize streamlit app
st.set_page_config(page_title="Multilang Invoice Extractor")
st.header("Multilang Invoice Extractor")
input = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

submit = st.button("Describe the invoice")

input_prompt = """
You are an expert in understanding invoices. We will upload an image as invoice and you will have to answer any questions based on the uploaded invoice image.
"""

if submit:
    response = get_gemini_response(input_prompt, image, input)
    st.subheader("response:")
    st.write(response)
