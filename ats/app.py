import io

from PyPDF2 import PdfReader
import streamlit as st
from dotenv import load_dotenv

# from google import genai
from langchain_groq import ChatGroq

_ = load_dotenv()

# Initialize Gemini
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = ChatGroq(model="groq/compound", temperature=0.7)


# Function to load Gemini model and get response
def get_gemini_response(input, pdf_content, prompt):
    # response = client.models.generate_content(
    #     model="gemini-2.5-flash",
    #     contents=[input, pdf_content, prompt],
    #     config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=3072),
    # )
    full_prompt = f"""
    Job Description:
    {input}

    Resume Content:
    {pdf_content}

    Instructions:
    {prompt}
    """

    response = model.invoke(full_prompt)

    return response.content


def get_pdf_text(pdf):
    text = ""
    pdf_bytes = pdf.getvalue()
    pdf_stream = io.BytesIO(pdf_bytes)
    pdf_reader = PdfReader(pdf_stream)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


def main():
    st.set_page_config(page_title="ATS")
    st.header("ATS")
    job_desc = st.text_area("Job Description", key="job_desc")
    uploaded_resume = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    request1 = st.button("Describe the Resume")
    request2 = st.button("How can i improve my skills?")
    request3 = st.button("Whats the percentage match?")

    input_prompt1 = """
    You are a strict, impartial senior Technical Recruiter with 15+ years of experience hiring for the exact role described. 
    Your task is to perform an objective, evidence-based evaluation of the candidate's resume against the job description.

    CRITICAL RULES:
    - Base every statement exclusively on explicit content present in the resume and job description.
    - Ignore candidate name, gender, age, nationality, photo, university prestige, or any personal identifiers.
    - Never infer or assume unstated skills/experience.
    - You MUST list an equal or greater number of weaknesses than strengths if gaps exist.
    - Be direct and critical when evidence is missing or weak.

    Output strictly in the following format:

    **Overall Fit:** Strong / Moderate / Weak / Poor (one phrase only)

    **Strengths (explicitly matched requirements):**
    • Requirement met – evidence from resume
    • ...

    **Critical Gaps / Weaknesses (missing or insufficient):**
    • Required skill/experience not demonstrated – impact on role
    • ...

    **Final Recommendation:** Advance to interview / Request additional screening / Reject (one phrase only)
    """

    input_prompt2 = """
    You are a strict, impartial Technical Hiring Manager. First perform an objective gap analysis of the resume against the job description, then provide concrete, actionable improvement advice.

    CRITICAL RULES:
    - Evaluate based solely on explicit evidence in the provided texts.
    - Ignore all personal identifiers (name, photo, demographics, etc.).
    - Prioritize technical and experience gaps over formatting unless explicitly required.
    - Every suggestion must be directly tied to a missing or weak requirement in the job description.

    Output strictly in this format:

    **Objective Gap Summary**
    Matches: X out of Y key requirements fully met

    **Non-Negotiable Gaps (will likely cause rejection):**
    • Missing requirement – suggested remediation
    • ...

    **Important but Fixable Gaps:**
    • Weak/under-developed area – specific action to strengthen
    • ...

    **Resume Presentation Improvements (only if relevant to JD):**
    • Bullet point structure / quantification / keyword usage issues

    **Recommended 30-Day Action Plan:**
    1. ...
    2. ...
    """

    input_prompt3 = """
    You are a calibrated ATS scanner combined with an expert Technical Sourcer. Your task is to calculate a precise, conservative match score.

    STEPS YOU MUST FOLLOW:
    1. Extract ALL hard skills, tools, frameworks, years of experience, certifications, and domain requirements explicitly stated in the job description.
    2. For each item, determine if the resume contains:
    - Full match (explicitly stated with similar or greater proficiency/level)
    - Partial match (mentioned but weaker/less experience)
    - No match
    3. Scoring: Full = 1.0 point, Partial = 0.5 points, None = 0
    4. Final % = (total points / total requirements) × 100 → round down to nearest integer

    CRITICAL RULES:
    - Never assume unstated experience
    - Company names/projects alone do not count as skill proof
    - Be strictly conservative on partial matches

    Output EXACTLY in this format with no additional text:

    Match Percentage: XX%

    Fully Matched Requirements:
    • skill/tool (evidence)
    • ...

    Partially Matched (count as 50%):
    • skill/tool – weakness in resume
    • ...

    Missing Requirements (0%):
    • skill/tool/experience
    • ...

    Final Verdict: Strong Match (80%+) / Moderate (65–79%) / Weak (50–64%) / Poor (<50%)
    """

    if uploaded_resume is not None:
        st.write("PDF Uploaded Successfully!")
        # st.write(get_pdf_text(uploaded_resume))
        content = get_pdf_text(uploaded_resume)

        if request1:
            prompt = input_prompt1
        elif request2:
            prompt = input_prompt2
        else:
            prompt = input_prompt3

        response = get_gemini_response(job_desc, content, prompt)
        st.write(response)
    else:
        st.write("Please upload the resume.")


if __name__ == "__main__":
    main()
