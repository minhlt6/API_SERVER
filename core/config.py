import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except Exception:
    pass

GROQ_API_KEYS = os.getenv('GROQ_API_KEYS', os.getenv('GROQ_API_KEY', '')).strip()
GEMINI_API_KEYS = os.getenv('GEMINI_API_KEYS', '').strip()

# Name models
LLM_MODEL = os.getenv('LLM_MODEL', 'llama-3.1-70b-versatile')
FAST_LLM_MODEL = os.getenv('FAST_LLM_MODEL', 'llama-3.1-8b-instant')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'BAAI/bge-m3')
CROSS_ENCODER_MODEL = os.getenv('CROSS_ENCODER_MODEL', 'BAAI/bge-reranker-v2-m3')

# Chunking and retrieval settings
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '800'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '150'))
TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '10'))
FINAL_TOP_K = int(os.getenv('FINAL_TOP_K', '3'))

DATA_DIR = os.getenv('DATA_DIR', 'data')
VECTOR_DIR = os.getenv('VECTOR_DIR', 'vectorstore')
UPLOAD_DIR = os.getenv('UPLOAD_DIR', 'uploads')
MAX_UPLOAD_SIZE_MB = int(os.getenv('MAX_UPLOAD_SIZE_MB', '20'))
QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION', 'rag_docs')
DOCUMENTS_DATABASE_URL = os.getenv('DOCUMENTS_DATABASE_URL', 'sqlite:///./rag_metadata.db')

# External service configs
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')

# - Context and output limits
MAX_CONTEXT_CHARS = int(os.getenv('MAX_CONTEXT_CHARS', '12000'))
MAX_OUT_CHARS = int(os.getenv('MAX_OUT_CHARS', '3000'))
MAX_HISTORY_MESSAGES = int(os.getenv('MAX_HISTORY_MESSAGES', '20'))