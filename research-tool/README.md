

# ScholarAI Research Tool 🤖📊

## Overview

ScholarAI is an AI-powered **Research Tool** that helps users extract meaningful insights from multiple online sources using **semantic search** and **large language models (LLMs)**.
Users provide URLs, ask questions, and get **context-aware answers with sources**.

---

## Features

* Load content from multiple URLs
* Automatically clean and process text
* Split long documents into smaller chunks
* Generate embeddings for semantic understanding
* Store embeddings in a FAISS vector database
* Answer user questions using an LLM
* Display answers with source references

---

## How It Works

```
URLs → Fetch Content → Split Text → Create Embeddings → Store in FAISS
                                                           ↓
                                            User Question → Semantic Search
                                                           ↓
                                              LLM generates final answer
                                                           ↓
                                                Answer + Sources
```

---

## Tech Stack

* **Python**
* **Streamlit** – User Interface
* **LangChain** – LLM orchestration
* **FAISS** – Vector search database
* **Google Gemini** – Language model
* **SentenceTransformers** – Local fallback embeddings
* **BeautifulSoup & Requests** – Web scraping

---

## Project Structure

```
project/
│
├── main.py              # Streamlit application
├── requirements.txt     # Project dependencies
├── faiss_index/         # Vector database (auto-generated)
├── .env                 # API keys (not pushed to GitHub)
└── README.md
```

---

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Environment

**Windows**

```powershell
.\venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run main.py
```

Open your browser at:

```
http://localhost:8501
```

---

## Usage

1. Paste one or more URLs into the input fields
2. Click **Process URLs**
3. Ask a research question
4. Get an AI-generated answer with sources

---

## Environment Variables

Create a `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```

---


---

## Future Improvements

* PDF upload support
* Chat history
* Multi-language support


---

## Author

Built with ❤️ using Python, Streamlit, LangChain, and FAISS

