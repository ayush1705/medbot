from app.config import settings
from app.utils.embedding_utils import get_embeddings
from typing import List
import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import CrossEncoder

VECTORSTORE_DIR = settings.vectorstore_path
INDEX_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")
METADATA_PATH = os.path.join(VECTORSTORE_DIR, "metadata.csv")
RERANKER_MODEL_NAME = settings.reranker_model_name

reranker_model = CrossEncoder(RERANKER_MODEL_NAME)

def rerank_with_cross_encoder(query: str, candidates: List[dict]) -> List[dict]:
    """
    Rerank retrieved candidates using a cross-encoder model.
    Args:
        query: Input user question
        candidates: List of dicts with "question" and "similarity" keys
    Returns:
        List of candidates reranked with new score
    """
    if not candidates:
        return []

    pairs = [(query, c["question"]) for c in candidates]

    scores = reranker_model.predict(pairs)

    # Attach reranker score to each candidate
    for i, cand in enumerate(candidates):
        cand["rerank_score"] = float(scores[i].item())

    # Sort by rerank score descending
    candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return candidates

def retrieve_from_vectordb(question: str, fetch_k: int, top_k: int, similarity_threshold: float, re_rank: bool):
    metadata_df = pd.read_csv(METADATA_PATH, quotechar='"')

    # Compute embedding for the input question
    query_embedding = np.array(get_embeddings([question]), dtype="float32")

    # Search FAISS
    index = faiss.read_index(INDEX_PATH)
    distances, indices = index.search(query_embedding, fetch_k)
    distances = distances[0]
    indices = indices[0]

    results = []
    for i, idx in enumerate(indices):
        if idx == -1:
            continue
        similarity = 1 - distances[i]  # Convert L2 distance to pseudo similarity
        if similarity < similarity_threshold:
            continue

        record = metadata_df.iloc[idx].to_dict()
        results.append(
            {
                "record_id": record.get("record_id"),
                "group_id": record.get("group_id"),
                "similarity": float(similarity),
                "question": record.get("question"),
                "answer": record.get("answer"),
                # Optional: include original question and answer if stored in metadata
            }
        )

    if re_rank and results:
        results = rerank_with_cross_encoder(question, results)

    results = results[:top_k]
    return results
