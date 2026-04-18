---
title: DocRAG
emoji: ⚡
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
app_file: app.py
pinned: false
---

# DocRAG Intelligence

DocRAG Intelligence is a powerful Retrieval-Augmented Generation (RAG) system running a modern, button-less GUI perfectly adapted for documentation. It securely builds, stores (in-memory FAISS), and executes dynamic contextual searches against massive online documentation.

## Running Locally

1. Create a `venv` and activate it.
2. Install the `requirements.txt`.
3. Fill in your `GEMINI_API_KEY` in your `.env`.
4. Run `python app.py`

## Hugging Face Spaces Deployment
This application leverages `sentence-transformers` and `google-genai` embedded within a Python Flask API serving HTML strings from the root controller.

Due to the architectural constraints around scaling and maintaining stateful FAISS databases in volatile instances, the HF Space version binds using Docker/Gunicorn on Port 7860.
