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
CROSS_ENCODER_MODEL = os.getenv('CROSS_ENCODER_MODEL', 'BAAI/bge-reranker-base')

# Chunking and retrieval settings
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '800'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '150'))
TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '25'))
FINAL_TOP_K = int(os.getenv('FINAL_TOP_K', '5'))

DATA_DIR = os.getenv('DATA_DIR', 'data')
VECTOR_DIR = os.getenv('VECTOR_DIR', 'vectorstore')

# External service configs
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')

# - Context and output limits
MAX_CONTEXT_CHARS = int(os.getenv('MAX_CONTEXT_CHARS', '12000'))
MAX_OUT_CHARS = int(os.getenv('MAX_OUT_CHARS', '3000'))
MAX_HISTORY_MESSAGES = int(os.getenv('MAX_HISTORY_MESSAGES', '20'))