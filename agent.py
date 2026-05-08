from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from ddgs import DDGS
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import os
import shutil

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Global vector DB
vector_db = None

class Question(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global vector_db
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            return {"message": "No readable text in PDF"}

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()
        vector_db = FAISS.from_documents(chunks, embeddings)

        return {"message": f"PDF uploaded successfully! ({len(chunks)} chunks)"}

    except Exception as e:
        return {"message": f"Error: {str(e)}"}

@app.post("/ask")
def ask(body: Question):
    try:
        pdf_context = ""
        web_context = ""

        # Step 1 - Search PDF if uploaded
        if vector_db is not None:
            docs = vector_db.similarity_search(body.question, k=3)
            pdf_context = "\n".join([doc.page_content for doc in docs])

        # Step 2 - Search the web
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(body.question, max_results=6):
                if r.get('body') and len(r['body']) > 50:
                    results.append(f"Source: {r['href']}\n{r['body']}")
        web_context = "\n\n".join(results)

        # Step 3 - Build combined context
        combined = ""
        if pdf_context:
            combined += f"--- FROM YOUR PDF ---\n{pdf_context}\n\n"
        if web_context:
            combined += f"--- FROM THE WEB ---\n{web_context}"

        if not combined:
            return {"answer": "No results found. Try a different question."}

        # Step 4 - Ask Groq
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful and accurate AI assistant.
You will be given context from two sources - a PDF document and web search results.
Your job is to:
1. Combine information from both sources intelligently
2. Prioritize PDF content for document specific questions
3. Use web results for current or general information
4. Never make up information not present in the sources
5. Keep your answer clear and to the point"""
                },
                {
                    "role": "user",
                    "content": f"{combined}\n\nQuestion: {body.question}\n\nAnswer based on the sources above."
                }
            ],
            temperature=0.3
        )

        return {"answer": response.choices[0].message.content}

    except Exception as e:
        return {"answer": f"Error: {str(e)}"}
    