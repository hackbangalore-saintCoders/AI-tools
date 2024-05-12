import os
from fastapi import FastAPI, File, UploadFile, HTTPException
import base64
import io
from PIL import Image 
import pdf2image
import google.generativeai as genai

app = FastAPI()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_text, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content([input_text, pdf_content[0], prompt])
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

@app.post("/evaluate_resume/")
async def evaluate_resume(job_description: str, resume_pdf: UploadFile = File(...)):
    pdf_content = input_pdf_setup(resume_pdf)
    input_prompt_rejected = """
    As an experienced recruiter, you have been entrusted with the responsibility of thoroughly assessing resumes against specific job descriptions. However, in this scenario, the candidate's qualifications do not align with the requirements for the role. Craft a professional email providing constructive feedback to the candidate, delineating the specific areas in which their profile falls short and offering guidance for improvement. Your aim is to communicate respectfully while clearly outlining the deficiencies in their application.
    """
    response = get_gemini_response(job_description, pdf_content, input_prompt_rejected)
    return {"feedback": response}
