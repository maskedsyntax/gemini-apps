import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from youtube_transcript_api import YouTubeTranscriptApi

_ = load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

prompt = """You are Yotube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 250 words. Please provide the summary of the text given here:  """


# Function to load Gemini model and get response
def get_gemini_response(prompt, transcript_text):
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[prompt, transcript_text],
        config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=3072),
    )

    return response.text


def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]

        ytt_api = YouTubeTranscriptApi()
        transcript_text = ytt_api.fetch(video_id=video_id)

        transcript = ""
        for snippet in transcript_text:
            transcript += " " + snippet.text

        return transcript

    except Exception as e:
        raise e


st.title("YouTube Transcript to Detailed Notes Converter")
youtube_video_url = st.text_input("Enter YouTube Video Link:")

if youtube_video_url:
    video_id = youtube_video_url.split("=")[1]
    print(video_id)
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

if st.button("Get Detailed Notes"):
    transcript_text = extract_transcript_details(youtube_video_url)

    if transcript_text:
        summary = get_gemini_response(prompt, transcript_text)
        st.markdown("## Detailed Notes:")
        st.write(summary)
