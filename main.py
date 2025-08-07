# For debugging: 
# import os
# from dotenv import load_dotenv

# Load .env file
# load_dotenv()

# Get your API keys
# openai_api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
# pinecone_env = os.getenv("PINECONE_ENV")

# check if keys are loaded successfully
# print("OpenAI Key:", openai_api_key[:5] + "..." if openai_api_key else "Missing")
# print("Pinecone Env:", pinecone_env)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from document_handler import process_document, query_document

app = FastAPI()

# Allow CORS for all origins (you can restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    result = process_document(file.filename, content)
    return {
        "message": "Document processed and summarized.",
        "summary": result["summary"]
    }

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    # Use retriever or QA chain here
    return {"answer": "Example answer"}
