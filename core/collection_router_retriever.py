import hashlib
import logging
from typing import List

from langchain_core.documents import Document as LangChainDocument
from rank_bm25 import BM25Okapi

from .collection_utils import collection_matches_cohort
from .document_db import SessionLocal, list_active_collection_names

logger = logging.getLogger(__name__)


class CollectionRouterRetriever:
    def __init__(
        self,
        base_retriever,
        qdrant_client,
        embeddings_model,
        top_n_collections: int = 3,
    ) -> None:
        self.base_retriever = base_retriever
        self.qdrant_client = qdrant_client
        self.embeddings_model = embeddings_model
        self.top_n_collections = max(1, int(top_n_collections or 3))
        self._bm25_cache = {}  # {collection_name -> BM25Okapi instance}

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

    def _ensure_bm25_loaded(self, collection_name: str) -> BM25Okapi | None:
        """Lazy load and cache BM25 index for a collection.
        
        First time: fetch all docs from Qdrant, build BM25, cache it (~0.3s)
        Subsequent times: reuse from cache (~0.001s)
        """
        # Check if already cached
        if collection_name in self._bm25_cache:
            return self._bm25_cache[collection_name]
        
        try:
            # Fetch ALL documents from collection (no query vector, get full corpus)
            all_points = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=10000,  # Batch size
            )
            
            points_list = all_points[0] if isinstance(all_points, tuple) else all_points
            
            if not points_list:
                logger.warning("No documents found in collection=%s for BM25 indexing", collection_name)
                return None
            
            # Extract documents and tokenize for BM25
            docs_for_bm25 = []
            for point in points_list:
                payload = point.payload if isinstance(point.payload, dict) else {}
                content = str(payload.get("content") or "").strip()
                if content:
                    docs_for_bm25.append(content)
            
            if not docs_for_bm25:
                logger.warning("No valid content found in collection=%s for BM25 indexing", collection_name)
                return None
            
            # Build BM25 index
            tokenized_docs = [doc.lower().split() for doc in docs_for_bm25]
            bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.5)
            
            # Cache it
            self._bm25_cache[collection_name] = bm25
            logger.info("BM25 index built and cached for collection=%s (docs=%d)", collection_name, len(docs_for_bm25))
            
            return bm25
            
        except Exception:
            logger.exception("Failed to build BM25 index for collection=%s", collection_name)
            return None

    def _search_target_collections(self, query: str, collections: List[str], limit: int, alpha: float = 0.6) -> List:
        """Hybrid search: BM25 + Vector + RRF (Option 2 with cached BM25)"""
        if not collections:
            return []

        try:
            query_vector = self.embeddings_model.embed_query(query)
        except Exception:
            logger.exception("Failed to embed query for collection routing")
            return []

        # Step 1: Vector search (từ Qdrant)
        all_docs_dict = {}  # {doc_key -> LangChainDocument}
        vector_ranked = {}  # {doc_key -> rank}
        
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

        # Step 2: BM25 search (lexical) - using CACHED index
        bm25_ranked = {}  # {doc_key -> rank}
        if all_docs_dict:
            try:
                tokenized_query = query.lower().split()
                
                # For each collection, use cached BM25 index
                for collection_name in collections:
                    # Load cached BM25 (or build if first time)
                    bm25 = self._ensure_bm25_loaded(collection_name)
                    if bm25 is None:
                        continue
                    
                    # Get BM25 scores for vector results
                    docs_from_collection = [
                        doc for doc in all_docs_dict.values()
                        if doc.metadata.get("collection_name") == collection_name
                    ]
                    
                    if not docs_from_collection:
                        continue
                    
                    # Get BM25 ranks
                    bm25_results = bm25.get_top_n(tokenized_query, docs_from_collection, n=len(docs_from_collection))
                    
                    bm25_rank = 0
                    for doc in bm25_results:
                        doc_key = self._doc_key(doc)
                        if doc_key not in bm25_ranked:
                            bm25_rank += 1
                            bm25_ranked[doc_key] = bm25_rank
                            
            except Exception:
                logger.exception("BM25 search failed, falling back to vector-only")

        # Step 3: RRF combination (Reciprocal Rank Fusion)
        alpha = max(0.0, min(1.0, float(alpha)))
        bm25_weight = 1.0 - alpha
        vector_weight = alpha
        rrf_c = 60
        
        rrf_scores = {}
        for doc_key, doc in all_docs_dict.items():
            score = 0.0
            
            # Vector score
            if doc_key in vector_ranked:
                score += vector_weight / (rrf_c + vector_ranked[doc_key])
            
            # BM25 score
            if doc_key in bm25_ranked:
                score += bm25_weight / (rrf_c + bm25_ranked[doc_key])
            
            if score > 0:
                rrf_scores[doc_key] = score
        
        # Sort by RRF score
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

        if cohort_scoped:
            deduplicated = []
            seen = set()
            for doc in routed_docs:
                key = self._doc_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                deduplicated.append(doc)
                if len(deduplicated) >= candidate_k:
                    break
            return deduplicated[:k]

        fallback_docs = []
        if self.base_retriever is not None:
            try:
                fallback_docs = self.base_retriever.search(
                    query,
                    k=candidate_k,
                    alpha=alpha,
                    cohort_key=cohort_key,
                )
            except TypeError:
                fallback_docs = self.base_retriever.search(
                    query,
                    k=candidate_k,
                    alpha=alpha,
                )
            except Exception:
                logger.exception("Base retriever fallback failed")

        deduplicated = []
        seen = set()

        for doc in routed_docs + list(fallback_docs or []):
            key = self._doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(doc)
            if len(deduplicated) >= candidate_k:
                break

        return deduplicated[:k]
