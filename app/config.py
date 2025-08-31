from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Vector store & data paths
    data_path: str
    vectorstore_path: str

    # Embedding model (can be swapped: bge-large-en, BioBERT, etc.)
    embedding_model: str

    # Ollama / LLM model settings
    ollama_model: str
    llm_temperature: float
    llm_max_tokens: int
    ollama_host: str
    ollama_port: int

    # Retrieval thresholds
    fetch_k: int
    top_k: int
    similarity_threshold: float

    # Re-Ranking
    reranker_model_name: str
    re_rank: bool

    class Config:
        env_file = ".env"
        extra = "ignore"


# Instantiate settings so it can be imported directly
settings = Settings()
