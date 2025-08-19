#!/usr/bin/env python3
"""
QA Bot Web App (RAG with LangChain + IBM watsonx + Chroma + Gradio)

This script implements the full project described in your PDF:

1) document_loader()  -> PyPDFLoader                      [screenshot: pdf_loader.png]
2) text_splitter()    -> RecursiveCharacterTextSplitter   [screenshot: code_splitter.png]
3) watsonx_embedding()-> WatsonxEmbeddings                [screenshot: embedding.png]
4) vector_database()  -> Chroma.from_documents            [screenshot: vectordb.png]
5) retriever(file)    -> builds a Chroma retriever        [screenshot: retriever.png]
6) retriever_qa(file, query) + Gradio UI                  [screenshot: QA_bot.png]

Before running, set these environment variables for IBM watsonx.ai:
  export WATSONX_APIKEY="YOUR_API_KEY"
  export WATSONX_URL="https://us-south.ml.cloud.ibm.com"           # or your region endpoint
  export WATSONX_PROJECT_ID="YOUR_WATSONX_PROJECT_ID"
  # (optional overrides)
  export WATSONX_LLM_MODEL_ID="ibm/granite-13b-chat-v2"
  export WATSONX_EMBED_MODEL_ID="ibm/slate.30m.english.rtrvr"      # adjust to a valid embed model ID in your account
"""
import os
import pathlib
from functools import lru_cache
from typing import List, Union

# --- LangChain imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# IBM watsonx.ai
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM

# Chroma vector store (handle both modern & legacy import paths)
try:
    from langchain_chroma import Chroma  # preferred
except Exception:
    from langchain_community.vectorstores import Chroma  # fallback

# UI
import gradio as gr


# =====================
# Configuration Helpers
# =====================

def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing environment variable: {name}. "
            f"Please 'export {name}=...' before running."
        )
    return val


def get_embed_model_id() -> str:
    return os.getenv("WATSONX_EMBED_MODEL_ID", "ibm/slate.30m.english.rtrvr")


def get_llm_model_id() -> str:
    return os.getenv("WATSONX_LLM_MODEL_ID", "ibm/granite-13b-chat-v2")


# ===========================
# 1) Load PDF Documents
# ===========================
def document_loader(file_path: Union[str, os.PathLike]):
    """
    Load a PDF into LangChain Documents using PyPDFLoader.

    SCREENSHOT: pdf_loader.png
    """
    file_path = str(file_path)
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


# ==========================================
# 2) Split Long Documents into Text Chunks
# ==========================================
def text_splitter(docs: List, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into manageable chunks using RecursiveCharacterTextSplitter.

    SCREENSHOT: code_splitter.png
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


# ===================================
# 3) IBM watsonx Embeddings (Vectors)
# ===================================
def watsonx_embedding() -> WatsonxEmbeddings:
    """
    Instantiate WatsonxEmbeddings from langchain_ibm.

    SCREENSHOT: embedding.png
    """
    apikey = _require_env("WATSONX_APIKEY")
    url = _require_env("WATSONX_URL")
    project_id = _require_env("WATSONX_PROJECT_ID")
    model_id = get_embed_model_id()

    return WatsonxEmbeddings(
        model_id=model_id,
        url=url,
        apikey=apikey,
        project_id=project_id,
    )


# ============================================
# 4) Build / Populate Chroma Vector Database
# ============================================
def vector_database(chunks: List, persist_directory: Union[str, os.PathLike, None] = None):
    """
    Embed chunks using watsonx_embedding() and store in Chroma.

    SCREENSHOT: vectordb.png
    """
    embedding_model = watsonx_embedding()
    persist_directory = str(persist_directory) if persist_directory else None

    if persist_directory:
        pathlib.Path(persist_directory).mkdir(parents=True, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )

    if persist_directory:
        vectordb.persist()

    return vectordb


# =======================
# LLM (watsonx.ai) helper
# =======================
@lru_cache(maxsize=1)
def get_llm() -> WatsonxLLM:
    """
    Create a WatsonxLLM client.

    You can adjust decoding params to your preference.
    """
    apikey = _require_env("WATSONX_APIKEY")
    url = _require_env("WATSONX_URL")
    project_id = _require_env("WATSONX_PROJECT_ID")
    model_id = get_llm_model_id()

    # Reasonable default params for Q&A
    params = dict(
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        decoding_method="greedy",
        repetition_penalty=1.05,
    )

    return WatsonxLLM(
        model_id=model_id,
        url=url,
        apikey=apikey,
        project_id=project_id,
        params=params,
    )


# =====================
# 5) Build a Retriever
# =====================
def retriever(file_path: Union[str, os.PathLike], persist_dir: Union[str, os.PathLike, None] = None):
    """
    Load, split, embed, and convert to a Chroma retriever.

    SCREENSHOT: retriever.png
    """
    docs = document_loader(file_path)
    chunks = text_splitter(docs)
    vectordb = vector_database(chunks, persist_directory=persist_dir)
    retr = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retr


# ======================================================
# 6) Retrieval-QA over the PDF + Gradio front-end
# ======================================================
def retriever_qa(file_path: Union[str, os.PathLike], query: str) -> str:
    """
    Use RetrievalQA chain with watsonx LLM + Chroma retriever to answer a question.

    This is the function you will wire into the Gradio UI.

    SCREENSHOT: QA_bot.png (with UI)
    """
    if not file_path:
        return "Please upload a PDF first."

    llm = get_llm()
    r = retriever(file_path)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=r,
        chain_type="stuff",
        return_source_documents=False,
        verbose=False,
    )

    try:
        answer = qa.run(query)
    except Exception as e:
        answer = f"Error while answering your question: {e}"

    return answer


# ==============
# 7) Gradio App
# ==============
def build_ui():
    with gr.Blocks(title="QA Bot Web App (LangChain + watsonx + Chroma)") as demo:
        gr.Markdown(
            "# 📄🔎 QA Bot Web App\n"
            "Upload a PDF and ask questions. Built with LangChain, IBM watsonx.ai, and Chroma."
        )

        with gr.Row():
            pdf_file = gr.File(
                label="Upload a PDF",
                file_types=[".pdf"],
                type="filepath",  # we want the temp file path string
            )

        query = gr.Textbox(
            label="Your question",
            lines=2,
            value="What this paper is talking about?",
            placeholder="Ask anything about your PDF...",
        )

        ask_btn = gr.Button("Ask")
        answer = gr.Textbox(label="Answer", lines=8)

        def _on_click(file_path, q):
            return retriever_qa(file_path, q)

        ask_btn.click(_on_click, inputs=[pdf_file, query], outputs=answer)

        gr.Markdown(
            "Tip: For grading screenshots, capture the code sections named in the comments "
            "(e.g., **pdf_loader.png**, **code_splitter.png**, **embedding.png**, **vectordb.png**, "
            "**retriever.png**, and the UI as **QA_bot.png**)."
        )

    return demo


if __name__ == "__main__":
    # Launch the Gradio app
    app = build_ui()
    app.launch()
