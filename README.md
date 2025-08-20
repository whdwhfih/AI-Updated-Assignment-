# AI-Updated-Assignment-

Below is the Project structure
NSquare Xpert
│── app.py # Flask backend (RAG pipeline)
│── templates/
│ └── index.html # UI for uploading PDFs and chatting
│── uploads

# Project Overview

1) This project implements a web application that allows users to:

2) Upload a PDF document (supports 500+ pages)

3) Ask questions about its content in a chat-like interface

4) Get accurate, context-aware answers generated using an LLM (Flan-T5 by HuggingFace, but can be replaced with any LLM such as GPT)

# The system uses a Retrieval-Augmented Generation (RAG) pipeline:

a) Extract text from PDFs (PyMuPDF)

b) Split into overlapping chunks

c) Convert chunks to embeddings (sentence-transformers)

d) Store embeddings in FAISS for fast similarity search

e) Retrieve top chunks, pass them with user question to LLM for answer generation


🔹 Code Explanation
1. Imports & Setup
import os, uuid, fitz, faiss
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


os, uuid: file handling & unique filenames for uploaded PDFs.

fitz (PyMuPDF): extracts text from PDFs.

faiss: efficient similarity search (vector DB).

flask: web server framework.

CORS: allows frontend (different domain) to connect to Flask API.

SentenceTransformer: to get embeddings.

transformers: loads Flan-T5 model for answering questions.

2. Flask App Initialization
app = Flask(__name__)
CORS(app)
os.makedirs("uploads", exist_ok=True)


Creates Flask app.

Enables CORS so frontend can call backend API.

Ensures an uploads/ folder exists to store uploaded PDFs.

3. Embedding Model + FAISS Setup
embedder = SentenceTransformer("all-MiniLM-L6-v2")
dim = embedder.get_sentence_embedding_dimension()
index = faiss.IndexFlatIP(dim)
docs = []


Loads a sentence embedding model (all-MiniLM-L6-v2 → 384-dim vectors).

Creates FAISS index (IndexFlatIP) for cosine similarity search.

Keeps a list docs to map embeddings back to text chunks.

4. LLM Setup (Flan-T5)
tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


Loads Flan-T5 tokenizer & model for answer generation.

5. Text Chunking Function
def chunk_text(text, size=1000, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


Splits PDF text into chunks of ~1000 characters with 200-character overlap.

Overlap ensures context continuity across chunks.

6. Routes
a) Home
@app.route("/")
def home():
    return render_template("index.html")


Serves a frontend index.html page.

b) Upload PDF
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    path = f"uploads/{uuid.uuid4()}.pdf"
    file.save(path)

    text = "".join(page.get_text() for page in fitz.open(path))
    chunks = chunk_text(text)

    embs = embedder.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embs)
    index.add(embs)
    docs.extend(chunks)

    return jsonify({"status": "ok", "chunks_added": len(chunks)})


Saves uploaded PDF with a unique filename.

Extracts text using PyMuPDF.

Splits text into chunks.

Encodes each chunk into embeddings, normalizes them, and adds to FAISS index.

Stores chunks in docs.

Returns how many chunks were stored.

c) Ask a Question
@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]
    top_k = int(request.form.get("top_k", 5))

    q_emb = embedder.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, top_k)
    ctx = "\n\n".join(docs[i] for i in I[0])

    prompt = f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_length=200)
    ans = tok.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"answer": ans})


Takes a user question.

Converts it into an embedding.

Searches FAISS for top_k most relevant chunks.

Concatenates them into ctx (context).

Builds a prompt with context + question.

Passes to Flan-T5 model → generates answer.

Returns answer as JSON.

7. App Runner
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


Runs Flask on port 8000.

0.0.0.0 → accessible from local network.

Debug mode enabled for dev.

✅ Summary:
This code is a mini RAG (Retrieval-Augmented Generation) system.

FAISS stores embeddings of PDF text.

Flan-T5 generates answers based on retrieved chunks.

Flask API handles PDF upload + Q&A.
