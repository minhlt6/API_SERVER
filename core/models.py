from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from .config import LLM_MODEL, EMBED_MODEL, CROSS_ENCODER_MODEL, GROQ_API_KEY
from langchain_groq import ChatGroq

print(" Đang khởi tạo các models...")
llm  = ChatGroq(
    model=LLM_MODEL,
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
)


print(f" Đang tải Cross-Encoder: {CROSS_ENCODER_MODEL}")
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 128}
)

print(" Hoàn tất khởi tạo models!")