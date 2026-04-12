import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except Exception:
    pass


def _is_hf_persistent_storage_available() -> bool:
    data_dir = Path('/data')
    return data_dir.exists() and os.access(data_dir, os.W_OK)


_USE_HF_PERSISTENT_STORAGE = _is_hf_persistent_storage_available()


def _default_documents_db_url() -> str:
    if _USE_HF_PERSISTENT_STORAGE:
        return 'sqlite:////data/rag_metadata.db'
    return 'sqlite:///./rag_metadata.db'


def _bounded_int_from_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        parsed = default

    return max(minimum, min(maximum, parsed))

GROQ_API_KEYS = os.getenv('GROQ_API_KEYS', os.getenv('GROQ_API_KEY', '')).strip()
GEMINI_API_KEYS = os.getenv('GEMINI_API_KEYS', '').strip()

# Name models
LLM_MODEL = os.getenv('LLM_MODEL', 'llama-3.1-70b-versatile')
FAST_LLM_MODEL = os.getenv('FAST_LLM_MODEL', 'llama-3.1-8b-instant')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'bkai-foundation-models/vietnamese-bi-encoder')
CROSS_ENCODER_MODEL = os.getenv('CROSS_ENCODER_MODEL', 'itdainb/PhoRanker')

# Chunking and retrieval settings
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '800'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '150'))
TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '10'))
FINAL_TOP_K = int(os.getenv('FINAL_TOP_K', '5'))

QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION', 'rag_docs')
DOCUMENTS_DATABASE_URL = os.getenv('DOCUMENTS_DATABASE_URL', _default_documents_db_url())

# External service configs
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
SUPABASE_URL = (os.getenv('SUPABASE_URL') or '').rstrip('/')
SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY', '').strip()
SUPABASE_STORAGE_BUCKET = os.getenv('SUPABASE_STORAGE_BUCKET', 'file').strip()
SUPABASE_SYNC_INTERVAL_SECONDS = _bounded_int_from_env('SUPABASE_SYNC_INTERVAL_SECONDS', 120, 60, 180)
SUPABASE_STARTUP_SYNC_WAIT_SECONDS = _bounded_int_from_env('SUPABASE_STARTUP_SYNC_WAIT_SECONDS', 5, 0, 120)
SUPABASE_ADMIN_SYNC_TOKEN = os.getenv('SUPABASE_ADMIN_SYNC_TOKEN', '').strip()
SUPABASE_SYNC_SNAPSHOT_FILE = os.getenv('SUPABASE_SYNC_SNAPSHOT_FILE', 'supabase_sync_snapshot.json').strip()
SUPABASE_SYNC_ENABLED = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and SUPABASE_STORAGE_BUCKET)
SUPABASE_SYNC_ALLOWED_IPS = [ip.strip() for ip in os.getenv('SUPABASE_SYNC_ALLOWED_IPS', '').split(',') if ip.strip()]
SUPABASE_SYNC_ALLOW_PRIVATE_NETWORK = os.getenv('SUPABASE_SYNC_ALLOW_PRIVATE_NETWORK', 'true').strip().lower() in {'1', 'true', 'yes', 'on'}
COLLECTION_ROUTER_TOP_N = _bounded_int_from_env('COLLECTION_ROUTER_TOP_N', 3, 1, 20)

# - Context and output limits
MAX_CONTEXT_CHARS = int(os.getenv('MAX_CONTEXT_CHARS', '12000'))
MAX_OUT_CHARS = int(os.getenv('MAX_OUT_CHARS', '3000'))
MAX_HISTORY_MESSAGES = int(os.getenv('MAX_HISTORY_MESSAGES', '20'))