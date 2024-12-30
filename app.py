import faiss
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
import os
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

class QueryRequest(BaseModel):
    prompt: str

app = FastAPI()

origins = [
    "https://jarvis-io9g.onrender.com",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods like GET, POST, PUT, DELETE
    allow_headers=["*"],  # Allows all headers
)

# File paths
FAISS_INDEX_PATH = "./embeddings/index.faiss"
CHUNKS_FILE_PATH = "chunks.pkl"  # Path to the saved chunks
METADATA_FILE_PATH = "metadata.pkl"  # Path to the metadata file
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please configure it in the .env file.")

# Global variables for FAISS index, chunks, and metadata
index = None
CHUNKS = None
METADATA = None

# Functions for loading FAISS index, chunks, and metadata
def load_faiss_index(index_path):
    """Load the FAISS index from disk."""
    return faiss.read_index(index_path)

def load_chunks(chunks_file):
    """Load the preprocessed chunks from a file."""
    with open(chunks_file, 'rb') as f:
        chunks = pickle.load(f)
    return chunks

def load_metadata(metadata_file):
    """Load metadata from a file."""
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    return metadata

@app.on_event("startup")
async def startup():
    """Load preprocessed data during FastAPI startup."""
    global index, CHUNKS, METADATA
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index file not found at {FAISS_INDEX_PATH}")
    if not os.path.exists(CHUNKS_FILE_PATH):
        raise FileNotFoundError(f"Chunks file not found at {CHUNKS_FILE_PATH}")
    if not os.path.exists(METADATA_FILE_PATH):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_FILE_PATH}")
    
    index = load_faiss_index(FAISS_INDEX_PATH)
    CHUNKS = load_chunks(CHUNKS_FILE_PATH)
    METADATA = load_metadata(METADATA_FILE_PATH)
    print("Preprocessed data loaded successfully.")

def search_index(query, top_k=5):
    """Retrieve the most relevant chunks from the FAISS index."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []
    for i, chunk_index in enumerate(indices[0]):
        if chunk_index < len(CHUNKS):  # Ensure the index is valid
            results.append({
                "file_name": CHUNKS[chunk_index]["file_name"],
                "chunk_index": chunk_index,
                "chunk": CHUNKS[chunk_index]["text"],
                "distance": float(distances[0][i])  # Convert numpy.float32 to Python float
            })
    return results

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
        try:
            response_data = response.json()
            candidates = response_data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return "No response text available."
        except Exception as e:
            return f"Error parsing Gemini API response: {e}"
    else:
        return f"Error: {response.status_code} - {response.text}"

@app.options("/query")
async def options():
    return {}

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Handle user queries."""
    prompt = request.prompt
    search_results = search_index(prompt)
    
    # Combine context with file names for better results
    context = "\n".join(
        [f"File: {result['file_name']} (Chunk {result['chunk_index']}): {result['chunk']}" for result in search_results]
    )
    response = query_gemini_api(prompt, context, API_KEY)
    for result in search_results:
        # Convert numpy.int64 fields to int
        if isinstance(result.get("chunk_index"), np.int64):
            result["chunk_index"] = int(result["chunk_index"])
    
    return {
        "response": response,
        "search_results": search_results,
    }

@app.post("/passthrough")
async def passthrough_api(request: QueryRequest):
    """Passthrough API to forward the request to the external endpoint."""
    EXTERNAL_API_URL = "https://solutionarchitectagentimages-latest.onrender.com/query"
    try:
        # Forward the request to the external API
        external_response = requests.post(
            EXTERNAL_API_URL,
            json=request.dict(),
            headers={"Content-Type": "application/json"},
        )

        # Check if the external API call was successful
        if external_response.status_code == 200:
            return {"response": external_response.json()}
        else:
            return {
                "error": f"External API returned {external_response.status_code}: {external_response.text}"
            }
    except Exception as e:
        return {"error": f"An error occurred while forwarding the request: {str(e)}"}