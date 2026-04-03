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
        bm25_ranked = {doc.page_content.strip(): rank for rank, doc in enumerate(bm25_top_indices, 1)}

        # Lấy top k từ Vector
        try:
            vector_results = self.vectorstore.similarity_search(query, k=k*2)
            # Tạo dictionary lưu thứ hạng (rank) của Vector
            vector_ranked = {doc.page_content: rank for rank, doc in enumerate(vector_results, 1)}
        except Exception as e:
            print(f"Lỗi Vector Search: {e}")
            return [doc for doc in bm25_top_indices[:k]]

        all_retrieved = {doc.page_content.strip(): doc for doc in bm25_top_indices + vector_results}
        rrf_results = [] 
        c = 60 
        
        for content, doc in all_retrieved.items():
            score = 0.0
            if content in bm25_ranked:
                score += 1.0 / (c + bm25_ranked[content])
            if content in vector_ranked:
                score += 1.0 / (c + vector_ranked[content])
                
            if score > 0:
                rrf_results.append((score, doc))

        rrf_results.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in rrf_results[:k]]

