import os
import sqlite3

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    print("Error: GROQ_API_KEY not found. Please set it in the .env file.")
    exit()

# -----------------------------
# Load Embedding Model
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Load Chroma Vector Database
# -----------------------------
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# Load Groq Model
# -----------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# -----------------------------
# Connect SQLite
# -----------------------------
conn = sqlite3.connect("users.db")
cursor = conn.cursor()

print("=" * 60)
print("Context-Aware Customer Support RAG Bot")
print("=" * 60)

while True:

    user_input = input("\nEnter User ID (or type exit): ")

    if user_input.lower() == "exit":
        break

    if not user_input.isdigit():
        print("Please enter a valid numeric User ID.")
        continue

    user_id = int(user_input)

    cursor.execute(
        "SELECT name, membership_tier FROM users WHERE user_id=?",
        (user_id,)
    )

    user = cursor.fetchone()

    if user is None:
        print("User not found. Please enter a valid user_id.")
        continue

    name, membership = user

    query = input("Enter your question: ")

    docs = retriever.invoke(query)

    if len(docs) == 0:
        print("I do not have enough information in the provided knowledge base to answer this.")
        continue

    context = "\n\n".join([doc.page_content for doc in docs])

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
{query}

Answer:
"""

    try:
        response = llm.invoke(prompt)

        print("\n" + "=" * 60)
        print(f"Hello {name}!")
        print(f"Membership Tier: {membership}\n")
        print(response.content)
        print("=" * 60)

    except Exception as e:
        print("Error communicating with Groq API.")
        print(e)

conn.close()