import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import json

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

load_dotenv()


class LocalEmbeddings:
    """Light wrapper around sentence-transformers to provide the embed_documents/embed_query API."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts):
        self._ensure_model()
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, q):
        self._ensure_model()
        return self._model.encode([q], show_progress_bar=False)[0].tolist()

    def __call__(self, text):
        """Allow direct calling as embedding function."""
        return self.embed_query(text)


# Verify API key exists and configure the Google SDK
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found in .env file!")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="ScholarAI: Research Tool")

st.title("ScholarAI: Research Tool 🧠")
st.sidebar.title("Source URLs / Documents")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
# We'll call Google Gemini via the google.generativeai SDK when answering queries


def load_urls(urls):
    docs = []
    for url in urls:
        if url:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                docs.append(Document(page_content=text,
                            metadata={"source": url}))
            except Exception as e:
                st.warning(f"Failed to load {url}: {str(e)}")
    return docs


if process_url_clicked:
    if not any(urls):
        st.error("Please enter at least one URL")
    else:
        try:
            main_placeholder.text("Data Loading...Started...✅✅✅")
            data = load_urls([url for url in urls if url])

            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)

            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001")
                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                main_placeholder.text(
                    "Embedding Vector Started Building...✅✅✅")
            except Exception as e:
                # Fallback to local embeddings if Google embedding quota exhausted
                err_msg = str(e).lower()
                if "quota" in err_msg or "resource_exhausted" in err_msg or "429" in err_msg:
                    main_placeholder.warning(
                        "Embedding quota exceeded on Google — using local embeddings fallback.")
                    try:
                        local_emb = LocalEmbeddings()
                    except Exception:
                        raise RuntimeError(
                            "Local fallback requires 'sentence-transformers' package. Install it with: pip install sentence-transformers")

                    vectorstore_openai = FAISS.from_documents(docs, local_emb)
                    main_placeholder.text(
                        "Embedding Vector (local) Started Building...✅✅✅")
                else:
                    raise
            time.sleep(2)

            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)

            main_placeholder.text("✅ URLs processed successfully!")
        except Exception as e:
            main_placeholder.error(f"Error: {str(e)}")

st.subheader("Ask a Question")
query = st.text_input("Question: ")

if query:
    # Load FAISS index: prefer saved local index, otherwise try pickle
    if os.path.isdir("faiss_index"):
        try:
            # read metadata
            import json

            with open(os.path.join("faiss_index", "meta.json"), "r", encoding="utf-8") as mf:
                meta = json.load(mf)
        except Exception:
            meta = {"embedding": "google"}

        try:
            if meta.get("embedding") == "local":
                emb = LocalEmbeddings(model_name=meta.get(
                    "model", "all-MiniLM-L6-v2"))
            else:
                emb = GoogleGenerativeAIEmbeddings(
                    model=meta.get("model", "models/embedding-001"))

            vectorstore = FAISS.load_local("faiss_index", emb)
        except Exception as e:
            st.error(f"Failed to load FAISS index: {e}")
            vectorstore = None

    elif os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
        except Exception as e:
            st.error(
                "Failed to load legacy pickle index — please reprocess URLs to rebuild the index.")
            vectorstore = None

    else:
        vectorstore = None

    if vectorstore:
        try:
            # Retrieve top documents from FAISS
            docs = vectorstore.similarity_search(query, k=4)

            # Build context from retrieved docs
            context_parts = []
            sources = []
            for d in docs:
                # `d` may be a LangChain Document or simple object
                text = getattr(d, "page_content", None) or getattr(
                    d, "content", None) or str(d)
                src = d.metadata.get("source") if getattr(
                    d, "metadata", None) else None
                context_parts.append(text)
                if src:
                    sources.append(src)

            context = "\n\n----\n\n".join(context_parts)

            # Compose prompt for the LLM
            system_prompt = (
                "You are an assistant that answers user questions using the provided context. "
                "Answer concisely and include a Sources section listing origins when appropriate."
            )
            full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

            # Call Google Gemini using the correct API
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            response = model.generate_content(full_prompt)
            answer_text = response.text if response else ""

            # `answer_text` is the generated response

            st.header("Answer")
            st.write(answer_text)
            if sources:
                st.subheader("Sources:")
                for s in sources:
                    st.write(s)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please process URLs first before asking questions")
