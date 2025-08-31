from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import logging

# Import routers
from app.routers import upload, ingest, retrieve, eval_rag, generate

# Initialize FastAPI app
app = FastAPI(
    title="Medical RAG API",
    version="1.0.0",
    docs_url="/docs"  # default Swagger UI
)

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Include routers
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(ingest.router, prefix="/ingest", tags=["Ingest"])
app.include_router(retrieve.router, prefix="/retrieve", tags=["Retrieve"])
app.include_router(generate.router, prefix="/generate", tags=["Generate Answer"])
app.include_router(eval_rag.router, prefix="/eval_rag", tags=["RAG Evaluation"])
# app.include_router(eval_generate.router, prefix="/eval_generate", tags=["Generation Evaluation"])

# Redirect root "/" to Swagger docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")
