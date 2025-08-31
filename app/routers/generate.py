from fastapi import APIRouter, HTTPException, Query
from typing import List
import os
from app.config import settings
from app.utils.retrieve_utils import retrieve_from_vectordb
from app.utils.prompt_utils import build_system_prompt

# from langchain.llms import Ollama
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

router = APIRouter()

@router.get("/generate_answer/")
async def generate_answer(
    question: str,
    fetch_k: int = Query(settings.fetch_k, ge=1),
    top_k: int = Query(settings.top_k, ge=1),
    similarity_threshold: float = Query(settings.similarity_threshold, ge=0.0, le=1.0),
    re_rank: bool = Query(settings.re_rank),
):
    """
    Generate answer for user question using RAG + Ollama.
    Steps:
    1. Retrieve top-k documents from FAISS vectorstore.
    2. If no docs → return default message.
    3. If docs → build system prompt with context.
    4. Call Ollama locally via LangChain to generate answer.
    """
    try:
        retrieved_docs = retrieve_from_vectordb(question, fetch_k, top_k, similarity_threshold, re_rank)

        # Build system prompt using retrieved documents
        system_prompt_text = build_system_prompt(question, retrieved_docs)
        print("#SYSTEM_PROMPT_TEXT", system_prompt_text)

        # Call Ollama directly
        llm = OllamaLLM(
            model=settings.ollama_model,
            base_url=f"{settings.ollama_host}:{settings.ollama_port}",
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
        )

        # Directly run your prepared prompt
        answer = ""
        answer = llm.invoke(system_prompt_text)

        return {"question": question, "answer": answer, "retrieved_docs_count": len(retrieved_docs), "retrieved_docs": retrieved_docs}


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
