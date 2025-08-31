from fastapi import APIRouter, HTTPException, Query
from typing import List
import os
from app.config import settings
from app.utils.retrieve_utils import retrieve_from_vectordb

router = APIRouter()

@router.get("/retrieve/")
async def retrieve_similar(
    question: str,
    fetch_k: int = Query(settings.fetch_k, ge=1),
    top_k: int = Query(settings.top_k, ge=1),
    similarity_threshold: float = Query(settings.similarity_threshold, ge=0.0, le=1.0),
    re_rank: bool = Query(settings.re_rank),
):
    """
    Retrieve top-k most similar records from FAISS vector store.
    Args:
        question: User input question
        fetch_k: Number of top similar records to fetch from vector similarity search
        top_k: Number of top similar records to return
        similarity_threshold: Minimum cosine similarity threshold (0-1)
    Returns:
        List of matching records with record_id, group_id, question, answer, and similarity score
    """
    try:
        
        retrieved_docs = retrieve_from_vectordb(question, fetch_k, top_k, similarity_threshold, re_rank)
        return {"query": question, "results": retrieved_docs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
