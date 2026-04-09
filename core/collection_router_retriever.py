import hashlib
import logging
from typing import List

from langchain_core.documents import Document as LangChainDocument

from .collection_utils import collection_matches_year
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

    def _select_target_collections(self, year_scope: str | None) -> List[str]:
        fetch_limit = max(self.top_n_collections * 4, 12)
        active_collections = self._get_active_collections(limit=fetch_limit)
        if not active_collections:
            return []

        normalized_year_scope = (year_scope or "").strip()
        if normalized_year_scope:
            return [
                collection_name
                for collection_name in active_collections
                if collection_matches_year(collection_name, normalized_year_scope)
            ]

        return active_collections[: self.top_n_collections]

    def _search_target_collections(self, query: str, collections: List[str], limit: int) -> List:
        if not collections:
            return []

        try:
            query_vector = self.embeddings_model.embed_query(query)
        except Exception:
            logger.exception("Failed to embed query for collection routing")
            return []

        scored_docs = []
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
                scored_docs.append(
                    (
                        float(getattr(point, "score", 0.0) or 0.0),
                        LangChainDocument(page_content=content, metadata=metadata),
                    )
                )

        scored_docs.sort(key=lambda row: row[0], reverse=True)
        return [doc for _, doc in scored_docs]

    def search(self, query: str, k: int = 10, alpha: float = 0.6, year_scope: str | None = None) -> List:
        if k <= 0:
            return []

        candidate_k = max(k * 4, k)
        year_scoped = bool((year_scope or "").strip())
        target_collections = self._select_target_collections(year_scope)

        if year_scoped and not target_collections:
            return []

        routed_docs = self._search_target_collections(
            query=query,
            collections=target_collections,
            limit=candidate_k,
        )

        if year_scoped:
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

        try:
            fallback_docs = self.base_retriever.search(
                query,
                k=candidate_k,
                alpha=alpha,
                year_scope=year_scope,
            )
        except TypeError:
            fallback_docs = self.base_retriever.search(
                query,
                k=candidate_k,
                alpha=alpha,
            )

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
