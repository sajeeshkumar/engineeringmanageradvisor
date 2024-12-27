# README

This project implements a two-step process for creating and deploying a Retrieval-Augmented Generation (RAG) pipeline using FAISS for document retrieval and the Gemini API for response generation.

## Step 1: Preprocessing PDFs and Creating a FAISS Index
The preprocessing script extracts text from multiple PDF files, splits the text into chunks, and generates embeddings for those chunks using the `SentenceTransformer` model. These embeddings are stored in a FAISS index for efficient retrieval.

### Preprocessor Script (`indexcreator.py`)

#### **How It Works**
1. **Extract Text**: Reads text from all PDF files in a specified folder.
2. **Split into Chunks**: Splits the extracted text into smaller chunks for embedding.
3. **Generate Embeddings**: Uses `SentenceTransformer` to create vector embeddings for the text chunks.
4. **Save FAISS Index**: Stores the embeddings in a FAISS index for later retrieval.

#### **Usage**
1. Place all your PDF files in a folder (e.g., `./books`).
2. Run the `indexcreator.py` script to process the PDFs and create the FAISS index.

```bash
python indexcreator.py
```

#### **Configuration**
- Set the path to your PDFs folder in the script (e.g., `pdf_folder = "./books"`).
- Set the output path for the FAISS index (e.g., `faiss_index_path = "./embeddings/index.faiss"`).

The script generates two outputs:
- FAISS index file: `./embeddings/index.faiss`
- Text chunks file: `./embeddings/chunks.json`

---

## Step 2: Running the FastAPI App
The FastAPI app uses the preprocessed FAISS index and text chunks to serve user queries. It retrieves relevant chunks based on a user query, sends the context and query to the Gemini API, and returns the generated response.

### App Script (`app.py`)

#### **How It Works**
1. **Load FAISS Index and Chunks**: The app loads the preprocessed FAISS index and text chunks into memory.
2. **Search the Index**: On receiving a query, it retrieves the most relevant chunks from the index.
3. **Call Gemini API**: The query and retrieved chunks are sent to the Gemini API to generate a response.

#### **Usage**
1. Make sure the FAISS index and chunks file are in the specified locations (e.g., `./embeddings/index.faiss` and `./embeddings/chunks.json`).
2. Start the FastAPI app:

```bash
python app.py
```

3. Send queries to the `/query` endpoint using a tool like `curl`, Postman, or a web client.

#### **Example Query**
```bash
curl -X POST "http://127.0.0.1:8000/query" \
-H "Content-Type: application/json" \
-d '{"prompt": "Explain how conflict can be managed in remote teams"}'
```

#### **Environment Variables**
- `API_KEY`: Your Gemini API key. Set it in the `app.py` script or as an environment variable.

---

## Project Structure
```
.
├── books/                     # Folder containing PDF files
├── embeddings/
│   ├── index.faiss            # Generated FAISS index
│   ├── chunks.json            # Generated text chunks
├── preprocessor.py            # Script for preprocessing PDFs
├── app.py                     # FastAPI app for querying the RAG pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Deployment
This project can be deployed to a platform like **Render**. 