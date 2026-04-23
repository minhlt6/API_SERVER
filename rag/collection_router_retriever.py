import hashlib
import logging
from typing import List

from langchain_core.documents import Document as LangChainDocument
from rank_bm25 import BM25Okapi


try:
    from pyvi import ViTokenizer
except Exception:
    ViTokenizer = None

from .collection_utils import collection_matches_cohort
from database.document_db import SessionLocal, list_active_collection_names

logger = logging.getLogger(__name__)


def _vi_tokenize(text: str) -> List[str]:
    normalized = (text or "").lower().strip()
    if not normalized:
        return []
    if ViTokenizer is None:
        return normalized.split()
    return ViTokenizer.tokenize(normalized).split()


class CollectionRouterRetriever:
    def __init__(
        self,
        qdrant_client,          
        embeddings_model,
        top_n_collections: int = 3,
    ) -> None:
        self.qdrant_client = qdrant_client
        self.embeddings_model = embeddings_model
        self.top_n_collections = max(1, int(top_n_collections or 3))
        # Cache giờ đây lưu một dict: { 'bm25': obj, 'corpus_docs': list, 'count': int }
        self._bm25_cache = {}  

    @staticmethod
    def _doc_key(doc) -> str:
        metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
        source = str(
            metadata.get("object_path")
            or metadata.get("source_relpath")
            or metadata.get("source_file")
            or metadata.get("source")
            or ""
        )
        page = str(metadata.get("page_number") or metadata.get("page") or "")
        content = (doc.page_content or "").strip()
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest() if content else "empty"
        return f"{source}|{page}|{digest}"

    def _get_active_collections(self, limit: int) -> List[str]:
        db = SessionLocal()
        try:
            return list_active_collection_names(db, limit=limit)
        finally:
            db.close()

    def _select_target_collections(self, cohort_key: str | None) -> List[str]:
        fetch_limit = max(self.top_n_collections * 4, 12)
        active_collections = self._get_active_collections(limit=fetch_limit)
        if not active_collections:
            return []

        normalized_cohort = (cohort_key or "").strip()
        if normalized_cohort:
            return [
                collection_name
                for collection_name in active_collections
                if collection_matches_cohort(collection_name, normalized_cohort)
            ]

        return active_collections[: self.top_n_collections]

    def _ensure_bm25_loaded(self, collection_name: str) -> tuple[BM25Okapi, List[LangChainDocument]] | None:
        """Lazy load and cache BM25 index and corpus for a collection (với cơ chế tự động làm mới Cache)"""
        
        # 1. Lấy tổng số chunks hiện tại trong Qdrant (Rất nhanh, tốn < 10ms)
        try:
            collection_info = self.qdrant_client.get_collection(collection_name)
            current_count = collection_info.points_count
        except Exception:
            logger.exception("Failed to get collection info for %s", collection_name)
            return None

        # 2. Kiểm tra Cache: Nếu chưa có hoặc số lượng thay đổi -> Xóa cache build lại
        cached_data = self._bm25_cache.get(collection_name)
        if cached_data and cached_data.get('count') == current_count:
            # Tái sử dụng (Phải trả về cả bm25 VÀ corpus_docs để map điểm)
            return cached_data['bm25'], cached_data['corpus_docs'] 
            
        logger.info(f"Phát hiện dữ liệu mới hoặc chưa có cache cho {collection_name} (Count: {current_count}). Đang build lại BM25...")
        
        try:
            points_list = []
            offset = None
            # Phân trang để lấy TOÀN BỘ documents từ collection
            while True:
                response = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=10000,
                    offset=offset,
                    with_payload=True,     
                    with_vectors=False     
                )
                batch_points, next_offset = response
                points_list.extend([p for p in batch_points if p is not None])
                
                offset = next_offset
                if offset is None:  
                    break
            
            if not points_list:
                logger.warning("No documents found in collection=%s for BM25 indexing", collection_name)
                return None
            
            # Trích xuất content và build documents
            corpus_docs = []
            for point in points_list:
                payload = point.payload if isinstance(point.payload, dict) else {}
                content = str(payload.get("content") or "").strip()
                if content:
                    metadata = {
                        "source": payload.get("path") or payload.get("object_path") or payload.get("stored_name") or "",
                        "source_file": payload.get("filename") or payload.get("stored_name") or "",
                        "source_relpath": payload.get("object_path") or payload.get("path") or "",
                        "object_path": payload.get("object_path") or "",
                        "folder_key": payload.get("folder_key") or "",
                        "collection_name": collection_name,
                        "academic_year": payload.get("academic_year") or "",
                        "chunk_index": payload.get("chunk_index"),
                        "page_number": payload.get("page_number"),
                    }
                    doc = LangChainDocument(page_content=content, metadata=metadata)
                    corpus_docs.append(doc)
            
            if not corpus_docs:
                logger.warning("No valid content found in collection=%s for BM25 indexing", collection_name)
                return None
            
            tokenized_docs = [_vi_tokenize(doc.page_content) for doc in corpus_docs]
            bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.5)
            
            # 3. Lưu lại Cache kèm the con số count và corpus_docs để đối chiếu lần sau
            self._bm25_cache[collection_name] = {
                'bm25': bm25,
                'corpus_docs': corpus_docs,
                'count': current_count
            }
            logger.info("BM25 index built and cached for collection=%s (docs=%d)", collection_name, len(corpus_docs))
            
            return bm25, corpus_docs
            
        except Exception:
            logger.exception("Failed to build BM25 index for collection=%s", collection_name)
            return None

    def _search_target_collections(self, query: str, collections: List[str], limit: int, alpha: float = 0.6) -> List:
        """Hybrid search: BM25 + Vector + RRF"""
        if not collections:
            return []

        try:
            query_vector = self.embeddings_model.embed_query(query)
        except Exception:
            logger.exception("Failed to embed query for collection routing")
            return []

        # Step 1: Vector search 
        all_docs_dict = {}  
        vector_ranked = {}  
        
        vector_rank = 0
        for collection_name in collections:
            try:
                points = self.qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                )
            except Exception:
                logger.exception("Qdrant search failed for collection=%s", collection_name)
                continue

            for point in points:
                payload = point.payload if isinstance(point.payload, dict) else {}
                content = str(payload.get("content") or "").strip()
                if not content:
                    continue

                metadata = {
                    "source": payload.get("path") or payload.get("object_path") or payload.get("stored_name") or "",
                    "source_file": payload.get("filename") or payload.get("stored_name") or "",
                    "source_relpath": payload.get("object_path") or payload.get("path") or "",
                    "object_path": payload.get("object_path") or "",
                    "folder_key": payload.get("folder_key") or "",
                    "collection_name": collection_name,
                    "academic_year": payload.get("academic_year") or "",
                    "chunk_index": payload.get("chunk_index"),
                    "page_number": payload.get("page_number"),
                }
                doc = LangChainDocument(page_content=content, metadata=metadata)
                doc_key = self._doc_key(doc)
                
                all_docs_dict[doc_key] = doc
                if doc_key not in vector_ranked:
                    vector_rank += 1
                    vector_ranked[doc_key] = vector_rank

        # Step 2: BM25 search 
        bm25_ranked = {}  
        try:
            tokenized_query = _vi_tokenize(query)
            
            if not tokenized_query:
                logger.warning("Query is empty after tokenization, skipping BM25 search")
            else:
                for collection_name in collections:
                    bm25_data = self._ensure_bm25_loaded(collection_name)
                    if bm25_data is None:
                        continue

                    bm25, corpus_docs = bm25_data

                    scores = bm25.get_scores(tokenized_query)
                    scored_docs = sorted(zip(corpus_docs, scores), key=lambda x: x[1], reverse=True)

                    bm25_rank = 0
                    for doc, score in scored_docs:
                        if score <= 0:  
                            break
                            
                        doc_key = self._doc_key(doc)
                        
                        if doc_key not in all_docs_dict:
                            all_docs_dict[doc_key] = doc
                            
                        if doc_key not in bm25_ranked:
                            bm25_rank += 1
                            bm25_ranked[doc_key] = bm25_rank
                            
                        if bm25_rank >= limit:
                            break
                            
        except Exception:
            logger.exception("BM25 search failed, falling back to vector-only")

        # Step 3: RRF combination
        alpha = max(0.0, min(1.0, float(alpha)))
        bm25_weight = 1.0 - alpha
        vector_weight = alpha
        rrf_c = 60
        
        rrf_scores = {}
        for doc_key, doc in all_docs_dict.items():
            score = 0.0
            
            if doc_key in vector_ranked:
                score += vector_weight / (rrf_c + vector_ranked[doc_key])
            
            if doc_key in bm25_ranked:
                score += bm25_weight / (rrf_c + bm25_ranked[doc_key])
            
            if score > 0:
                rrf_scores[doc_key] = score
        
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [all_docs_dict[doc_key] for doc_key, _ in sorted_results[:limit]]

    def search(self, query: str, k: int = 10, alpha: float = 0.6, cohort_key: str | None = None) -> List:
        if k <= 0:
            return []

        candidate_k = max(k * 4, k)
        cohort_scoped = bool((cohort_key or "").strip())
        target_collections = self._select_target_collections(cohort_key)

        if cohort_scoped and not target_collections:
            return []

        routed_docs = self._search_target_collections(
            query=query,
            collections=target_collections,
            limit=candidate_k,
            alpha=alpha,
        )
        
        if not routed_docs:
            logger.warning("No documents found for query=%s, cohort=%s", query[:50], cohort_key)
            return []

        deduplicated = []
        seen = set()
        for doc in routed_docs:
            key = self._doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(doc)
            if len(deduplicated) >= k:
                break
                
        return deduplicated