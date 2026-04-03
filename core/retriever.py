from typing import List
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """Kết hợp BM25 và Vector Search."""
    def __init__(self, vectorstore, documents):
        self.vectorstore = vectorstore
        self.documents = documents
        print(" Đang khởi tạo BM25...")
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs,k1=1.5, b=0.5)
        print(" BM25 sẵn sàng!")

    def search(self, query: str, k: int = 10, alpha: float = 0.6) -> List:
        # Lấy top k từ BM25
        tokenized_query = query.lower().split()
        bm25_top_indices = self.bm25.get_top_n(tokenized_query, self.documents, n=k*2)
        bm25_ranked = {doc.page_content: rank for rank, doc in enumerate(bm25_top_indices, 1)}

        # Lấy top k từ Vector
        try:
            vector_results = self.vectorstore.similarity_search(query, k=k*2)
            vector_ranked = {doc.page_content: rank for rank, doc in enumerate(vector_results, 1)}
        except Exception:
            return bm25_top_indices[:k]

        # Kết hợp bằng RRF
        rrf_scores = {}
        c = 60 # Hằng số RRF chuẩn
        
        for doc in self.documents:
            content = doc.page_content
            score = 0.0
            if content in bm25_ranked:
                score += 1.0 / (c + bm25_ranked[content])
            if content in vector_ranked:
                score += 1.0 / (c + vector_ranked[content])
                
            if score > 0:
                rrf_scores[doc] = score

        # Sắp xếp theo điểm RRF và trả về top K
        combined = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return [doc for doc, score in combined[:k]]

