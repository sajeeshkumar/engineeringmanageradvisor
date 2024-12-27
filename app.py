import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
import os
import requests
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel

# Load environment variables
load_dotenv()

class QueryRequest(BaseModel):
    prompt: str

app = FastAPI()

# File paths
FAISS_INDEX_PATH = "./embeddings/index.faiss"
CHUNKS_FILE_PATH = "chunks.pkl"  # Path to the saved chunks
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please configure it in the .env file.")

# Global variables for FAISS index and chunks
index = None
CHUNKS = None

# Functions for loading FAISS index and chunks
def load_faiss_index(index_path):
    """Load the FAISS index from disk."""
    return faiss.read_index(index_path)

def load_chunks(chunks_file):
    """Load the preprocessed chunks from a file."""
    with open(chunks_file, 'rb') as f:
        chunks = pickle.load(f)
    return chunks

@app.on_event("startup")
async def startup():
    """Load preprocessed data during FastAPI startup."""
    global index, CHUNKS
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index file not found at {FAISS_INDEX_PATH}")
    if not os.path.exists(CHUNKS_FILE_PATH):
        raise FileNotFoundError(f"Chunks file not found at {CHUNKS_FILE_PATH}")
    
    index = load_faiss_index(FAISS_INDEX_PATH)
    CHUNKS = load_chunks(CHUNKS_FILE_PATH)
    print("Preprocessed data loaded successfully.")

def search_index(query, top_k=5):
    """Retrieve the most relevant chunks from the FAISS index."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    relevant_chunks = [CHUNKS[i] for i in indices[0]]
    return " ".join(relevant_chunks)

def query_gemini_api(prompt, context, api_key):
    """Query Gemini API with the context and prompt."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [{"text": f"Context: {context}\n\nQuestion: {prompt}"}]
        }]
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        response_data = response.json()
        return response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    else:
        return f"Error: {response.status_code} - {response.text}"

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Handle user queries."""
    prompt = request.prompt
    context = search_index(prompt)
    response = query_gemini_api(prompt, context, API_KEY)
    return {"response": response}

def main():
    """Main function to start the FastAPI server."""
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
