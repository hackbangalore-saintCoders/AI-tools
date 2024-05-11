from typing import List
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
from langchain_community.vectorstores.faiss import FAISS
from fpdf import FPDF
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI, File, UploadFile
from langchain_community.vectorstores import FAISS

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def convert_to_pdf(input_files):
    pdfs = []
    for file in input_files:
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension == '.pdf':
            # If the file is already a PDF, append its path directly
            pdfs.append(file.file)
        # Assuming the file is a text file, create a PDF with the same content
        else:
            pdf = FPDF()
            pdf.add_page()
            with file.file as f:
                content = f.read()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, content.decode('utf-8'))
            output_file_path = file.filename.replace(os.path.splitext(file.filename)[1], '.pdf')
            pdf.output(output_file_path)
            pdfs.append(output_file_path)
    return pdfs

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Your task is to generate a detailed report for atleast 10,000 words.
    I will be dropping multiple pdf's contaning codebase. The repost must be very detailed, it should contain introduction, dependencies and how to install it,
    the codebase summary, domain knowledge, quality of code and other metrics, Introduction, Background and Literature Review,
    Problem Statement, Objectives, Methodology, System Design or Architecture, Implementation, Results and Analysis,
    Discussion, Conclusion, Future Work, References. You are allowed to add more topics and 
    broad headings as you feel convenient but the report must be very detailed adn elaborate.
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.9)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

app = FastAPI()

@app.post("/report/")
async def generate_report(files: List[UploadFile] = File(...)):
    pdf_docs = convert_to_pdf(files)
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    user_question = "Generate a detailed report."
    result = user_input(user_question)
    return {"report": result}
