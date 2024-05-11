import streamlit as st
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


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def convert_to_pdf(input_files):
    pdfs = []
    for file in input_files:
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == '.pdf':
            # If the file is already a PDF, append its path directly
            pdfs.append(file.name)
        # Assuming the file is a text file, create a PDF with the same content
        else: 
            pdf = FPDF()
            pdf.add_page()
            with open(file.name, 'r', encoding='utf-8') as f:
                content = f.read()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, content)
            output_file_path = file.name.replace(os.path.splitext(file.name)[1], '.pdf')
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
    Your task is to compare two PDF's, one PDF contains project details, it will contain skills required section for the project
    and other PDF will contain resume of a developer with skills section. Your task is to compare skills required for the project and 
    developers skills and provide me a score out of 10, be as  creative as possible.
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
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Test bot")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Convert dropped files to PDF
                pdf_docs = convert_to_pdf(pdf_docs)
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
