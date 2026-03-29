from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder 
from .config import EMBED_MODEL, CROSS_ENCODER_MODEL

# Khởi tạo Embedding model - Chạy trên CPU của Hugging Face
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device='cpu')

llm = None