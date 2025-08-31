from app.config import settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import torch
import numpy as np

# Initialize models based on config
if settings.embedding_model == "bge-large-en":
    model = SentenceTransformer("BAAI/bge-large-en")
elif settings.embedding_model == "BioMedBERT":
    # Assuming BioMedBERT for biomedical embeddings
    model = SentenceTransformer("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
elif settings.embedding_model == "PubMedBERT":
    # Assuming PubMedBERT for biomedical embeddings
    model = SentenceTransformer("neuml/pubmedbert-base-embeddings")
elif settings.embedding_model == "SapBERT":
    # Assuming SapBERT for biomedical embeddings
    model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")



def get_embeddings(texts):
    """
    Returns embeddings for a list of texts.
    Chooses appropriate Embedding model based on config.
    Converts list of texts into list of embeddings.
    Normalizes embeddings to cater cosine similarity evaluation and distance calculation.
    """
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings
