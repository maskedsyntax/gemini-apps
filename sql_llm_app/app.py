import json
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
import sqlite3

_ = load_dotenv()

# Initialize Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# Function to load Gemini model and get response
def get_gemini_response(input,prompt):
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[input, prompt],
        config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=3072),
    )

    return response.text


def read_sql_query(sql, db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.commit()
    conn.close()

    for row in rows:
        print(row)

    return rows


prompt = [
    """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name STUDENT and has the following columns - NAME, CLASS, 
    SECTION \n\nFor example,\nExample 1 - How many entries of records are present?, 
    the SQL command will be something like this SELECT COUNT(*) FROM STUDENT ;
    \nExample 2 - Tell me all the students studying in Data Science class?, 
    the SQL command will be something like this SELECT * FROM STUDENT 
    where CLASS="Data Science"; 
    also the sql code should not have ``` in beginning or end and sql word in output

    """
]


## Streamlit App

st.set_page_config(page_title="I can Retrieve Any SQL query")
st.header("Gemini App To Retrieve SQL Data")

question = st.text_input("Input: ", key="input")

submit = st.button("Ask the question")

# if submit is clicked
if submit:
    response = get_gemini_response(question, prompt)
    print(response)
    response = read_sql_query(response, "student.db")
    st.subheader("The Response is")
    for row in response:
        print(row)
        st.header(row)
