# Context-Aware Customer Support RAG Bot

## Project Overview

The Context-Aware Customer Support RAG Bot is an AI-powered customer support chatbot that uses the Retrieval-Augmented Generation (RAG) architecture. Instead of relying only on the Large Language Model's general knowledge, the chatbot retrieves relevant information from a local knowledge base before generating a response.

The chatbot also personalizes its responses by retrieving customer information from an SQLite database.

This project demonstrates the integration of:

- Retrieval-Augmented Generation (RAG)
- LangChain
- Groq Large Language Model
- SQLite Database
- Chroma Vector Database
- HuggingFace Embeddings

---

# Objective

The objective of this project is to build a customer support chatbot that:

- Retrieves customer information from a relational database.
- Retrieves relevant company policy information using semantic search.
- Generates accurate responses using a Large Language Model.
- Avoids hallucinations by restricting the model to retrieved knowledge.

---

# Features

- Customer-specific responses
- Retrieval-Augmented Generation (RAG)
- Local vector database
- SQLite user database
- Semantic search
- Personalized answers
- Hallucination-safe prompt
- Error handling
- Local execution
- Groq LLM Integration

---

# Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Main programming language |
| LangChain | RAG orchestration framework |
| Groq API | Large Language Model inference |
| SQLite | User information storage |
| Chroma | Local vector database |
| HuggingFace | Text embedding generation |
| Sentence Transformers | Embedding model |
| python-dotenv | Environment variable management |
| FastAPI (optional) | REST API deployment |
| Git | Version control |

---

# Project Structure

```
rag_customer_support/
│
├── app.py
├── ingest.py
├── create_db.py
├── company_faq.txt
├── users.db
├── chroma_db/
├── requirements.txt
├── README.md
├── .env
├── .env.example
└── venv/
```

---

# Project Architecture

```
                   Customer

                      │

                      ▼

             Enter User ID

                      │

                      ▼

              SQLite Database

      (Name + Membership Tier)

                      │

                      ▼

           Enter Customer Question

                      │

                      ▼

          Chroma Vector Database

     (Semantic Similarity Search)

                      │

                      ▼

       Relevant FAQ Chunks Retrieved

                      │

                      ▼

          Prompt Construction

                      │

                      ▼

         Groq Llama Language Model

                      │

                      ▼

          Personalized Response

                      │

                      ▼

                 Customer
```

---

# RAG Workflow

## Step 1

Customer enters

- User ID
- Question

Example

```
User ID : 101

Question :
What is the refund policy?
```

---

## Step 2

SQLite Database

The chatbot searches

```
users.db
```

to retrieve

```
Name

Membership Tier
```

Example

```
101

↓

Riya Sharma

↓

Gold
```

---

## Step 3

Semantic Search

The user question is converted into embeddings.

Example

```
"What is refund policy?"

↓

Embedding Vector

↓

Chroma Search

↓

Most Relevant FAQ
```

---

## Step 4

Prompt Construction

The chatbot constructs a prompt.

Example

```
Name : Riya Sharma

Membership : Gold

Context :
Refund Policy

Question :
What is the refund policy?
```

---

## Step 5

Groq LLM

The prompt is sent to

```
Llama Model
```

The LLM generates an answer using only the retrieved context.

---

## Step 6

Response

Example

```
Hello Riya Sharma

Membership : Gold

Customers may request a refund within
30 calendar days...
```

---

# Database Design

## Users Table

| Column | Type |
|---------|------|
| user_id | INTEGER |
| name | TEXT |
| membership_tier | TEXT |

Sample Records

| user_id | Name | Membership |
|---------|------|------------|
|101|Riya Sharma|Gold|
|102|Aman Verma|Silver|
|103|Neha Iyer|Platinum|

---

# Knowledge Base

The chatbot uses

```
company_faq.txt
```

Topics included

- Refund Policy
- Delivery Timelines
- Membership Benefits
- Account Cancellation
- Contact Hours
- Premium Support

---

# Libraries Used

## sqlite3

Purpose

Provides connectivity between Python and SQLite database.

Used For

- Creating database
- Reading user information
- Executing SQL queries

Example

```
SELECT name, membership_tier
FROM users
WHERE user_id=101
```

---

## LangChain

Purpose

Framework used to build Retrieval-Augmented Generation applications.

Responsibilities

- Document Loading
- Text Splitting
- Retrieval
- Prompt Management
- LLM Integration

---

## Chroma

Purpose

Stores vector embeddings locally.

Why Used

Traditional databases search using keywords.

Chroma searches using semantic similarity.

Example

```
Refund

Refund Policy

Money Back

Return

↓

Similar Meaning

↓

Retrieved
```

---

## HuggingFace Embeddings

Purpose

Converts text into numerical vectors.

Used Model

```
sentence-transformers/all-MiniLM-L6-v2
```

Why Used

- Free
- Lightweight
- Accurate
- Local execution

---

## Sentence Transformers

Purpose

Generate sentence-level embeddings.

Instead of matching words,

it matches meaning.

Example

```
Refund

Money Back

Return

↓

Similar Meaning
```

---

## Groq API

Purpose

Provides access to Large Language Models.

Used Model

```
Llama
```

Responsibilities

- Reads prompt
- Understands retrieved context
- Generates response

---

## python-dotenv

Purpose

Loads environment variables.

Instead of writing

```
API_KEY="..."
```

inside source code,

the key is securely stored inside

```
.env
```

---

# Installation

Clone repository

```
git clone <repository_url>

cd rag_customer_support
```

Create virtual environment

```
python -m venv venv
```

Activate virtual environment

Windows

```
venv\Scripts\activate
```

Linux / Mac

```
source venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Setup

Create

```
.env
```

Add

```
GROQ_API_KEY=your_api_key
```

---

# Create Database

```
python create_db.py
```

Creates

```
users.db
```

---

# Generate Vector Database

```
python ingest.py
```

Creates

```
chroma_db/
```

---

# Run Application

```
python app.py
```

---

# Sample Test Cases

### Test Case 1

```
User ID

101

Question

What is the refund policy?
```

---

### Test Case 2

```
User ID

103

Question

Do I get premium customer support?
```

---

### Test Case 3

```
User ID

999

Question

What are my benefits?
```

Expected

```
User not found.
```

---

### Test Case 4

```
User ID

102

Question

Can I cancel my account?
```

---

# Error Handling

The application handles

- Invalid User ID
- Missing API Key
- Missing Context
- Groq API Errors
- Database Errors

---

# Future Improvements

- FastAPI Web API
- Streamlit UI
- Conversation History
- Authentication
- Admin Dashboard
- PDF Knowledge Base
- Multiple Knowledge Sources
- Docker Deployment
- Cloud Database
- Redis Caching

---

# Author

**Swaran Chandra Reddy**

B.Tech – Computer Science Engineering

KLH University

Project developed as part of the Kovon AI Engineering Technical Assessment.