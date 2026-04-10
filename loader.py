"""
loader.py
---------
Loads clinical documents from PDF, plain text, and CSV (EHR export) formats.
Applies PII scrubbing before returning chunks ready for embedding.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Generator, List

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .pii_scrubber import ClinicalPIIScrubber

logger = logging.getLogger(__name__)


class ClinicalDocumentLoader:
    """
    Loads, cleans, de-identifies, and chunks clinical documents.

    Parameters
    ----------
    chunk_size : int
        Target token count per chunk (default 512).
    chunk_overlap : int
        Overlap between consecutive chunks (default 64).
    scrub_pii : bool
        Whether to run PII de-identification (default True).
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        scrub_pii: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.scrub_pii = scrub_pii

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        self.scrubber = ClinicalPIIScrubber() if scrub_pii else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_directory(self, directory: str | Path) -> List[Document]:
        """
        Recursively load all supported files from a directory.
        Returns a flat list of de-identified, chunked LangChain Documents.
        """
        directory = Path(directory)
        documents: List[Document] = []

        for filepath in sorted(directory.rglob("*")):
            if filepath.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            try:
                docs = list(self._load_file(filepath))
                documents.extend(docs)
                logger.info("Loaded %d chunks from %s", len(docs), filepath.name)
            except Exception as exc:
                logger.error("Failed to load %s: %s", filepath, exc)

        logger.info("Total chunks loaded: %d", len(documents))
        return documents

    def load_file(self, filepath: str | Path) -> List[Document]:
        return list(self._load_file(Path(filepath)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, filepath: Path) -> Generator[Document, None, None]:
        ext = filepath.suffix.lower()
        if ext == ".pdf":
            yield from self._load_pdf(filepath)
        elif ext == ".txt":
            yield from self._load_text(filepath)
        elif ext == ".csv":
            yield from self._load_csv(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _load_pdf(self, filepath: Path) -> Generator[Document, None, None]:
        loader = PyPDFLoader(str(filepath))
        pages = loader.load()
        for doc in self.splitter.split_documents(pages):
            yield self._maybe_scrub(doc, source=filepath.name)

    def _load_text(self, filepath: Path) -> Generator[Document, None, None]:
        loader = TextLoader(str(filepath), encoding="utf-8")
        docs = loader.load()
        for doc in self.splitter.split_documents(docs):
            yield self._maybe_scrub(doc, source=filepath.name)

    def _load_csv(self, filepath: Path) -> Generator[Document, None, None]:
        """
        Expects CSV with at minimum a 'note_text' column.
        Optional: 'patient_id', 'encounter_date', 'note_type'
        """
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("note_text", "")
                if not text.strip():
                    continue
                metadata = {
                    "source": filepath.name,
                    "patient_id": row.get("patient_id", "UNKNOWN"),
                    "encounter_date": row.get("encounter_date", ""),
                    "note_type": row.get("note_type", ""),
                }
                chunks = self.splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={**metadata, "chunk_index": i},
                    )
                    yield self._maybe_scrub(doc, source=filepath.name)

    def _maybe_scrub(self, doc: Document, source: str) -> Document:
        if not self.scrub_pii or self.scrubber is None:
            return doc
        result = self.scrubber.scrub(doc.page_content)
        doc.page_content = result.scrubbed_text
        doc.metadata["pii_entities_removed"] = result.scrubbed_count
        doc.metadata["source"] = source
        return doc
