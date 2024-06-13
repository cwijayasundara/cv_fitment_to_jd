import warnings
import streamlit as st

from dotenv import load_dotenv
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from util import pdf_to_text

warnings.filterwarnings('ignore')

load_dotenv()

model_name = "gpt-4o"

llm = ChatOpenAI(model=model_name,
                 streaming=True,
                 temperature=0,
                 callbacks=[StreamingStdOutCallbackHandler()])

st.title("CV Evaluator")

uploaded_jd = st.file_uploader("Choose the JD", type="txt")
uploaded_cv = st.file_uploader("Choose the CV", type=['pdf'])

evaluate_cv = st.button("Evaluate CV")


def get_jd_text(uploaded_jd):
    if uploaded_jd:
        jd_text = uploaded_jd.read().decode("utf-8")
        return jd_text


def get_cv_text(uploaded_cv):
    if uploaded_cv:
        cv_text = pdf_to_text(uploaded_cv)
        return cv_text


prompt = """
Here is the job description to evaluate the candidate against:

<job_description>
{JOB_DESCRIPTION}
</job_description>

And here is the candidate's CV:

<cv>
{CV}
</cv>

Please read through the job description carefully and identify the key qualifications, skills, experience,
and characteristics required and preferred for the role.

Then review the candidate's CV in detail. Evaluate how well the candidate's qualifications, skills and experience
align with and demonstrate the key requirements from the job description.

<justification> Write a short paragraph here evaluating and justifying how good of a match the candidate is for the
role based on how well their CV shows they meet the job requirements. Mention the candidate's key strengths and
weaknesses or gaps relative to the job description. </justification>

<score> Based on your analysis, rate how good of a match the candidate is for this role on a 1-5 scale,
where 1 = very poor match, 2 = below average match, 3 = average match, 4 = good match, and 5 = great match. </score>"""

cv_eval_chain = LLMChain.from_string(
    llm=llm,
    template=prompt
)

jd_text = get_jd_text(uploaded_jd)
cv_text = get_cv_text(uploaded_cv)

if evaluate_cv and jd_text and cv_text:

    st.write("Evaluating CV...")
    response = cv_eval_chain.run({'JOB_DESCRIPTION': jd_text,
                                  'CV': cv_text})
    st.write(response)
