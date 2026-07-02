from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# -------------------------------
# Step 1: Load the FAQ document
# -------------------------------
print("Loading FAQ document...")

loader = TextLoader("company_faq.txt", encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} document(s).")

# -------------------------------
# Step 2: Split into chunks
# -------------------------------
print("Splitting document into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

# -------------------------------
# Step 3: Create Embeddings
# -------------------------------
print("Loading HuggingFace embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# Step 4: Store in Chroma
# -------------------------------
print("Creating Chroma vector database...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("\nVector database created successfully!")
print("Vector database saved in folder: chroma_db")