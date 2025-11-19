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
        model="gemini-2.5-pro",
        contents=[input, image[0]],
        config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=3072),
    )

    return response.text


def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        # image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return [types.Part.from_bytes(data=bytes_data, mime_type=uploaded_file.type)]

    else:
        raise FileNotFoundError("No file uploaded")


# Initialize streamlit app
st.set_page_config(page_title="Calorie App")
st.header("Calorie App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

submit = st.button("Ask the question")

input_prompt = """
You are a certified clinical dietitian and nutritional scientist. Analyze the provided image of a meal and return a precise, evidence-based nutritional breakdown.

Follow these rules without exception:
- Identify only food items that are clearly visible and distinguishable.
- For every item, explicitly state the assumed portion size (e.g., “2 large eggs”, “1 medium slice”, “100 g”).
- Use only standard USDA or internationally accepted database values — never estimate or hallucinate calories/macros.
- Be conservative: when portion is ambiguous, choose the smaller common serving size.
- Output must follow this exact structure with the specified section headers, separators, and numbering. Do not add any extra text, explanations, or commentary outside these sections.

=== IDENTIFIED FOOD ITEMS & CALORIES ===
1. [Clear food name] – [X] kcal ([exact portion assumption])
2. [Clear food name] – [X] kcal ([exact portion assumption])
...
──────────────────────────────────
TOTAL ESTIMATED CALORIES: [X] kcal

=== MACRONUTRIENT BREAKDOWN ===
Protein:    XX% ([XX] g)
Carbohydrates: XX% ([XX] g)
Fat:        XX% ([XX] g)

=== HEALTH ASSESSMENT ===
Overall classification: [Very Healthy / Healthy / Moderately Healthy / Unhealthy / Very Unhealthy]

Justification (exactly 3 short sentences):
• Positive aspects: [one concise sentence]
• Areas of concern: [one concise sentence]
• Overall balance: [one concise sentence]

Recommendations (exactly 3 bullets):
• [specific, actionable change 1]
• [specific, actionable change 2]
• [specific, actionable change 3]

Disclaimer: This is an estimate based on visual identification and standard portion sizes. For precise dietary or medical needs, consult a registered dietitian with weighed measurements.
"""

if submit:
    image_data = input_image_setup(uploaded_file)
    response = get_gemini_response(input_prompt, image_data)
    st.subheader("response:")
    st.write(response)
