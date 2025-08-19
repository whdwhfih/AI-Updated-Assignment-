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
