import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except Exception:
    pass

# Read configuration from environment (safe for production). Defaults provided for convenience.
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
LLM_MODEL = os.getenv('LLM_MODEL', 'llama-3.1-8b-instant')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
CROSS_ENCODER_MODEL = os.getenv('CROSS_ENCODER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
DATA_DIR = os.getenv('DATA_DIR', 'data')
VECTOR_DIR = os.getenv('VECTOR_DIR', 'vectorstore')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1500'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '300'))
TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '10'))
FINAL_TOP_K = int(os.getenv('FINAL_TOP_K', '5'))
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///chat_history.db')