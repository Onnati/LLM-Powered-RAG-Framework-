# LLM-Powered Retrieval-Augmented Generation(RAG) Framework using ObjectBox Vector Search

In this end-to-end project, I developed a **Retrieval-Augmented Generation (RAG)** application using **ObjectBox Vector Database** and **LangChain**. The solution enhances the capabilities of large language models by retrieving relevant information from a custom knowledge base, enabling the AI to generate more accurate, context-aware, and up-to-date responses. By leveraging ObjectBox's on-device vector database, the application ensures efficient semantic search and retrieval while keeping all data local, preserving privacy and eliminating the need to send sensitive information to external servers.

<img width="1698" height="977" alt="image" src="https://github.com/user-attachments/assets/a74440d8-d5f8-465f-9cd3-a6c32ce89f8f" />



## Description

This project implements a RAG system that uses:

- **ObjectBox** as the on-device vector database
- **Google Gemini 2.5 Flash** as the LLM (`ChatGoogleGenerativeAI`)
- **HuggingFace BGE embeddings** (`BAAI/bge-small-en-v1.5`) for document and query vectorization
- **Streamlit** for the web UI

Users upload their own **PDF** or **TXT** files, embed them into ObjectBox, and ask questions about the uploaded content.

### How it works

1. Upload one or more PDF or TXT documents through the Streamlit file uploader.
2. Click **Embedd Documents** to load, split, embed, and store the content in ObjectBox.
3. Enter a question — the app retrieves the most relevant chunks and sends them to Gemini with your prompt.
4. View the answer and the retrieved source chunks in the **Document Similarity Search** expander.

### Implementation steps

1. Load uploaded files with `PyPDFLoader` (PDF) or plain text parsing (TXT).
2. Split documents into chunks of `1000` characters with `200` overlap using `RecursiveCharacterTextSplitter`.
3. Generate embeddings with `HuggingFaceBgeEmbeddings` (`384` dimensions, cosine similarity).
4. Store vectors in ObjectBox via `ObjectBox.from_documents`.
5. Configure Gemini with `ChatGoogleGenerativeAI` (`gemini-2.5-flash`).
6. Build a `create_stuff_documents_chain` and `create_retrieval_chain` to connect the retriever, prompt, and LLM.

## Libraries Used

- langchain==0.1.20
- langchain-community==0.0.38
- langchain-core==0.1.52
- langchain-google-genai==1.0.4
- langchain-objectbox
- python-dotenv==1.0.1
- pypdf==4.2.0
- streamlit
- sentence-transformers

## Installation

1. **Prerequisites**
   - Git
   - Python 3.10+
   - Command line familiarity
   ```

2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install streamlit sentence-transformers
   ```

4. **Configure your Google Gemini API key**

   Copy the example env file and add your key from [Google AI Studio](https://aistudio.google.com/apikey):

   ```bash
   cp .env.example .env
   ```

   Edit `.env`:
   ```env
   GOOGLE_API_KEY=your-google-api-key-here
   ```

6. **Run the app**
   ```bash
   cd app
   streamlit run app.py
   ```

## Usage

1. Upload one or more **PDF** or **TXT** files.
2. Click **Embedd Documents** and wait for processing to finish.
3. Enter a question about your uploaded documents.
4. Read the answer and inspect retrieved chunks under **Document Similarity Search**.

> **Note:** You must embed documents before asking questions. Re-clicking **Embedd Documents** replaces the previous index with your newly uploaded files.

## Project Structure

```
.
├── app/
│   ├── app.py          # Streamlit RAG application
│   ├── config.py       # Environment variable loading
│   └── utils.py        # Gemini LLM and embedding setup
├── objectbox/          # ObjectBox vector database (generated at runtime)
├── .env.example        # API key template
├── requirements.txt
└── README.md
``
