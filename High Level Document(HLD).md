**#Project Name: GenAI-RAG-Q-n-A-chatbot App**

#1. Overview

GenAI-RAG-Q-n-A-chatbot App is an AI-powered Retrieval-Augmented Generation (RAG) system that allows users to ask questions based on different input sources (text, PDFs, DOCX, links) and receive AI-generated answers. It integrates FAISS for vector search, Hugging Face LLMs for text generation, and Streamlit for UI.

#2. System Architecture

a) Input Sources

Link-based text extraction

PDFs, TXT, DOCX file parsing

Direct text input

b) Data Preprocessing

Extracts text from documents

Splits text into manageable chunks (CharacterTextSplitter)

Converts text into numerical representations (embeddings)

c) FAISS Vector Store

Stores text embeddings in a high-speed FAISS index

Allows fast similarity search for query retrieval

d) LLM-Based Answer Generation

Uses Meta Llama-3-8B-Instruct from Hugging Face

Answers user queries based on retrieved text chunks

e) Frontend - Streamlit UI

Allows users to upload files, enter text, and ask questions

Displays retrieved documents and AI-generated responses

3. Key Components

FAISS (Vector Search Engine) → Stores and retrieves document chunks.

Hugging Face LLM → Generates AI responses.

LangChain → Manages document processing and retrieval.

Streamlit → Web-based UI for interaction.

PyPDF2 / python-docx → Parses PDF and DOCX files.

Low-Level Design (LLD)

1. Input Handling

a) Supported Input Types

Text Input

PDFs (processed via PyPDF2)

DOCX (processed via python-docx)

TXT files

Web links (processed via WebBaseLoader)

b) Processing Logic

Extracts text from input source

Splits long texts into smaller 1000-character chunks

Converts chunks into embeddings (using sentence-transformers)

2. FAISS Vector Store

a) Index Creation & Storage

Uses IndexFlatL2 (L2-distance-based FAISS index)

Stores text embeddings + document metadata

b) Querying the Index

Converts user’s question into an embedding

Finds most relevant text chunks via FAISS search

Sends retrieved text chunks to LLM

3. Question Answering Workflow

User inputs a question

FAISS retrieves relevant chunks

LLM (Meta Llama-3-8B-Instruct) generates a response

Answer displayed in Streamlit UI

4. Streamlit UI

Dropdown for input type selection

Upload button for PDFs, DOCX, and TXT

Text input for direct queries

Output area for AI-generated answers

5. Error Handling

Handles missing API keys

Invalid document format detection

FAISS index errors (empty vector store warnings)

Input text validation (ensuring non-empty input)

Future Enhancements

Multi-file input support (process multiple PDFs/DOCX at once)

Advanced document parsing (support scanned PDFs via OCR)

API integration (deploy as a REST API for wider usage)

This document provides a structured breakdown of the project from architecture to detailed component behavior. Let me know if you need further refinements!

