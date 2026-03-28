from langchain_huggingface import HuggingFaceEmbeddings
from .config import EMBED_MODEL

# Model này sẽ chạy trên CPU của Hugging Face
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

llm = None