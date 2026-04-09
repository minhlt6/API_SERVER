from typing import List
import hashlib
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """Kết hợp BM25 và Vector Search."""
    def __init__(self, vectorstore, documents):
        self.vectorstore = vectorstore
        self.documents = documents
        print(" Đang khởi tạo BM25...")
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.5)
        self.rrf_c = 60
        print(" BM25 sẵn sàng!")

    @staticmethod
    def _doc_key(doc) -> str:
        metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
        source = str(metadata.get("source_relpath") or metadata.get("source_file") or metadata.get("source") or "")
        page = str(metadata.get("page_number") or metadata.get("page") or "")
        content = (doc.page_content or "").strip()
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest() if content else "empty"
        return f"{source}|{page}|{digest}"

    def search(self, query: str, k: int = 10, alpha: float = 0.6, year_scope: str | None = None) -> List:
        del year_scope
        if not self.documents or k <= 0:
            return []

        alpha = max(0.0, min(1.0, float(alpha)))
        bm25_weight = 1.0 - alpha
        vector_weight = alpha

        # Lấy top k từ BM25
        tokenized_query = query.lower().split()
        candidate_k = min(max(k * 4, k), len(self.documents))
        bm25_top_docs = self.bm25.get_top_n(tokenized_query, self.documents, n=candidate_k)

        bm25_ranked = {}
        all_retrieved = {}
        for rank, doc in enumerate(bm25_top_docs, 1):
            key = self._doc_key(doc)
            bm25_ranked[key] = rank
            all_retrieved[key] = doc

        # Lấy top k từ Vector
        try:
            vector_results = self.vectorstore.similarity_search(query, k=candidate_k)
        except Exception as e:
            print(f"Lỗi Vector Search: {e}")
            return [doc for doc in bm25_top_docs[:k]]

        vector_ranked = {}
        for rank, doc in enumerate(vector_results, 1):
            key = self._doc_key(doc)
            vector_ranked[key] = rank
            all_retrieved[key] = doc

        rrf_results = []
        
        for content, doc in all_retrieved.items():
            score = 0.0
            if content in bm25_ranked:
                score += bm25_weight / (self.rrf_c + bm25_ranked[content])
            if content in vector_ranked:
                score += vector_weight / (self.rrf_c + vector_ranked[content])

            if score > 0:
                rrf_results.append((score, doc))

        rrf_results.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in rrf_results[:k]]

