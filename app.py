import os
from fastapi import FastAPI, File, UploadFile, HTTPException
import base64
import io
from PIL import Image 
import pdf2image
import google.generativeai as genai

app = FastAPI()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(job_desc, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content([job_desc, pdf_content[0], prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        ## Convert the PDF to image
        images = pdf2image.convert_from_bytes(uploaded_file.file.read())

        first_page = images[0]

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return pdf_parts
    else:
        raise HTTPException(status_code=404, detail="No file uploaded")

@app.post("/performance_metrics/")
async def evaluate_resume(job_description: str, resume_pdf: UploadFile = File(...)):
    pdf_content = input_pdf_setup(resume_pdf)
    input_prompt_rejected = """
    Your task is to compare project details and resume and give me some metrics, job description will contain skills required section for the project
    and other is a PDF that will contain resume of a developer with skills section. Your task is to compare skills required for the project and 
    developers skills and provide me a score out of 100 percentage with a final score and feedback, be as  creative as possible.
    """
    response = get_gemini_response(job_description, pdf_content, input_prompt_rejected)
    return {"feedback": response}
