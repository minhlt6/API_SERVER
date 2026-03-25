from typing import List
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """Kết hợp BM25 và Vector Search."""
    def __init__(self, vectorstore, documents):
        self.vectorstore = vectorstore
        self.documents = documents
        print(" Đang khởi tạo BM25...")
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(" BM25 sẵn sàng!")

    def search(self, query: str, k: int = 10, alpha: float = 0.6) -> List:
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        try:
            vector_results = self.vectorstore.similarity_search_with_score(
                query, k=len(self.documents)
            )
        except:
            return self.documents[:k]
        vector_scores = {}
        for doc, distance in vector_results:
            similarity = 1 / (1 + distance)
            vector_scores[doc.page_content] = similarity
        combined = []
        for i, doc in enumerate(self.documents):
            bm25_score = bm25_scores[i]
            vector_score = vector_scores.get(doc.page_content, 0)
            final_score = alpha * vector_score + (1 - alpha) * bm25_score
            combined.append((final_score, doc))
        combined.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in combined[:k]]

