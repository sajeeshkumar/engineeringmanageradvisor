import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file, handling errors."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""  # Return an empty string for problematic files
    except Exception as e:
        print(f"Unexpected error reading {pdf_path}: {e}")
        return ""  # Handle other unexpected errors

def split_text_into_chunks(text, chunk_size=500):
    """Split text into smaller chunks for embedding."""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def create_faiss_index_from_pdfs(pdf_paths, index_path, chunk_size=500):
    """Create a FAISS index from multiple PDF files."""
    all_chunks = []
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(text, chunk_size)
        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append({"file_name": os.path.basename(pdf_path), "text": chunk})

    # Generate embeddings for all chunks
    embeddings = embedding_model.encode([chunk["text"] for chunk in all_chunks])

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
    index.add(np.array(embeddings))

    # Save the FAISS index and chunks to disk
    faiss.write_index(index, index_path)
    with open("chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("Preprocessing completed and data saved.")

# Specify paths
pdf_folder = "./books"  # Folder containing your PDFs
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
faiss_index_path = "./embeddings/index.faiss"

# Preprocess PDFs and create FAISS index
create_faiss_index_from_pdfs(pdf_files, faiss_index_path)
