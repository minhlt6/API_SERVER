from typing import List
import logging
from .models import cross_encoder

MAX_RERANK_CHARS = 1200
logger = logging.getLogger(__name__)

def advanced_rerank(question: str, docs: List, top_k: int = 5) -> List:
    if not docs:
        return []
    logger.info("Đang rerank %s tài liệu với Cross-Encoder...", len(docs))
    pairs = [(question, (doc.page_content or "")[:MAX_RERANK_CHARS]) for doc in docs]
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    logger.info("Top 3 điểm: %s", [f"{s:.3f}" for s, _ in ranked[:3]])
    return [doc for score, doc in ranked[:top_k]]

