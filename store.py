"""
store.py
--------
Abstracts Pinecone (production) and FAISS (local dev) vector stores
behind a unified interface. Swap backends via the VECTOR_STORE env var.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.vectorstores.base import VectorStore

logger = logging.getLogger(__name__)


class Backend(str, Enum):
    PINECONE = "pinecone"
    FAISS = "faiss"


class VectorStoreManager:
    """
    Unified interface for Pinecone and FAISS vector stores.

    Usage
    -----
    manager = VectorStoreManager(embeddings=my_embedder, backend="pinecone")
    manager.ingest(documents)
    retriever = manager.as_retriever(k=5)
    """

    def __init__(
        self,
        embeddings: Embeddings,
        backend: str = "faiss",
        index_name: str = "clinical-notes",
        faiss_save_path: str = "./faiss_index",
    ):
        self.embeddings = embeddings
        self.backend = Backend(backend.lower())
        self.index_name = index_name
        self.faiss_save_path = faiss_save_path
        self._store: Optional[VectorStore] = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, documents: List[Document], batch_size: int = 100) -> None:
        """
        Embed and upsert documents into the vector store.
        Creates the index if it does not exist.
        """
        if not documents:
            logger.warning("No documents provided for ingestion.")
            return

        logger.info(
            "Ingesting %d documents into %s/%s",
            len(documents),
            self.backend.value,
            self.index_name,
        )

        if self.backend == Backend.PINECONE:
            self._ingest_pinecone(documents, batch_size)
        else:
            self._ingest_faiss(documents)

        logger.info("Ingestion complete.")

    def _ingest_pinecone(self, documents: List[Document], batch_size: int) -> None:
        import pinecone

        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENVIRONMENT"],
        )

        # Create index if needed
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=1536,  # text-embedding-3-small
                metric="cosine",
            )
            logger.info("Created Pinecone index: %s", self.index_name)

        # Upsert in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            if self._store is None:
                self._store = Pinecone.from_documents(
                    batch, self.embeddings, index_name=self.index_name
                )
            else:
                self._store.add_documents(batch)
            logger.debug("Upserted batch %d/%d", i // batch_size + 1, -(-len(documents) // batch_size))

    def _ingest_faiss(self, documents: List[Document]) -> None:
        if self._store is None:
            self._store = FAISS.from_documents(documents, self.embeddings)
        else:
            self._store.add_documents(documents)
        self._store.save_local(self.faiss_save_path)
        logger.info("FAISS index saved to %s", self.faiss_save_path)

    # ------------------------------------------------------------------
    # Loading existing index
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load an existing index (FAISS from disk, Pinecone from cloud)."""
        if self.backend == Backend.FAISS:
            self._store = FAISS.load_local(self.faiss_save_path, self.embeddings)
            logger.info("Loaded FAISS index from %s", self.faiss_save_path)
        elif self.backend == Backend.PINECONE:
            import pinecone

            pinecone.init(
                api_key=os.environ["PINECONE_API_KEY"],
                environment=os.environ["PINECONE_ENVIRONMENT"],
            )
            self._store = Pinecone.from_existing_index(
                index_name=self.index_name, embedding=self.embeddings
            )
            logger.info("Connected to Pinecone index: %s", self.index_name)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def as_retriever(self, k: int = 5, score_threshold: float = 0.7):
        """Return a LangChain retriever with similarity score filtering."""
        self._assert_loaded()
        return self._store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        self._assert_loaded()
        return self._store.similarity_search(query, k=k)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assert_loaded(self) -> None:
        if self._store is None:
            raise RuntimeError(
                "Vector store not loaded. Call .ingest() or .load() first."
            )
