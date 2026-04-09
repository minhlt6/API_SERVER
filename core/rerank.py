from typing import List
from .models import cross_encoder

MAX_RERANK_CHARS = 1200

def advanced_rerank(question: str, docs: List, top_k: int = 5) -> List:
    if not docs:
        return []
    print(f"Đang rerank {len(docs)} documents với Cross-Encoder...")
    pairs = [(question, (doc.page_content or "")[:MAX_RERANK_CHARS]) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    print(f" Top 3 scores: {[f'{s:.3f}' for s, _ in ranked[:3]]}")
    return [doc for score, doc in ranked[:top_k]]

