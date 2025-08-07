import os
import pdfplumber
import tempfile
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document

load_dotenv()

# Load keys
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

# Setup Pinecone client
pc = PineconeClient(api_key=pinecone_api_key)

# Index name
INDEX_NAME = "doc-index"

# Create embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Connect to vectorstore
vectorstore = LangchainPinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embedding_model
)

print("✅ Pinecone + OpenAI embedding setup complete.")

# Use pdfplumber to load documents
def process_document(filename, content):
    # Save the file temporarily
    temp_file_path = os.path.join(tempfile.gettempdir(), filename)

    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(content)

    # Use pdfplumber to extract text from the PDF
    with pdfplumber.open(temp_file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    # If no text was extracted, return an error
    if not text:
        raise ValueError("No text extracted from PDF")

    # Split text into documents for LangChain
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    docs = [Document(page_content=chunk) for chunk in chunks]

    # Summarize using LangChain's summarize chain
    llm = ChatOpenAI(temperature=0.3, model="gpt-4")  
    chain = load_summarize_chain(llm, chain_type="stuff")  # options: map_reduce, refine, stuff

    summary = chain.run(docs)

    return {
        "chunks": docs,
        "summary": summary
    }

def query_document(question):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = LangchainPinecone.from_existing_index(INDEX_NAME, embeddings)
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=docsearch.as_retriever())
    return qa.run(question)
