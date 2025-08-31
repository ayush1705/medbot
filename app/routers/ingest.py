from fastapi import APIRouter, HTTPException
import os
import csv
import pandas as pd
import faiss
import numpy as np
from app.utils.embedding_utils import get_embeddings
from app.config import settings

router = APIRouter()

DATA_PATH = "data/train.csv"
VECTORSTORE_DIR = "vectorstore/faiss_index"

@router.post("/ingest/")
async def ingest_data():
    """
    Ingests training dataset:
    - Loads train.csv
    - Converts questions into embeddings
    - Builds FAISS index
    - Saves index + metadata (record_id, group_id)
    """
    try:
        # Load train.csv
        df = pd.read_csv(DATA_PATH, quotechar='"')

        # Generate embeddings
        questions = df["question"].tolist()
        embeddings = get_embeddings(questions)  # shape (N, D)

        # Convert to numpy float32
        embeddings_np = np.array(embeddings, dtype="float32")

        # Create FAISS index
        dim = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings_np)

        # Ensure vectorstore dir exists
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)

        # Save FAISS index
        faiss.write_index(index, os.path.join(VECTORSTORE_DIR, "index.faiss"))

        # Save metadata (record_id + group_id)
        metadata_path = os.path.join(VECTORSTORE_DIR, "metadata.csv")
        df[["record_id", "group_id", "question", "answer"]].to_csv(metadata_path, index=False, quotechar='"', quoting=csv.QUOTE_ALL)

        return {"status": "success", "message": f"FAISS index created with {len(df)} records."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
