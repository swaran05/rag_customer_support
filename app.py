import os
import sqlite3

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise Exception("GROQ_API_KEY not found in .env")

# -------------------------------
# Create FastAPI App
# -------------------------------
app = FastAPI(
    title="Context-Aware Customer Support RAG Bot",
    description="AI-powered Customer Support Chatbot using RAG",
    version="1.0"
)

# -------------------------------
# Load Embedding Model
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# Load Chroma Vector Store
# -------------------------------
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# Load Groq LLM
# -------------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# -------------------------------
# SQLite Connection
# -------------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

# -------------------------------
# Request Model
# -------------------------------
class ChatRequest(BaseModel):
    user_id: int
    user_query: str

# -------------------------------
# Home Endpoint
# -------------------------------
@app.get("/")
def home():
    return {
        "message": "Context-Aware Customer Support RAG Bot is running successfully!"
    }

# -------------------------------
# Chat Endpoint
# -------------------------------
@app.post("/chat")
def chat(request: ChatRequest):

    cursor.execute(
        "SELECT name, membership_tier FROM users WHERE user_id=?",
        (request.user_id,)
    )

    user = cursor.fetchone()

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found. Please enter a valid user_id."
        )

    name, membership = user

    docs = retriever.invoke(request.user_query)

    if len(docs) == 0:
        return {
            "answer": "I do not have enough information in the provided knowledge base to answer this."
        }

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an AI customer support assistant.

You are speaking with:

Name: {name}
Membership Tier: {membership}

Answer the user's question using only the context provided below.

If the answer is not available in the context, say:

"I do not have enough information in the provided knowledge base to answer this."

Context:

{context}

User Question:

{request.user_query}

Answer:
"""

    try:
        response = llm.invoke(prompt)

        return {
            "name": name,
            "membership_tier": membership,
            "answer": response.content
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )