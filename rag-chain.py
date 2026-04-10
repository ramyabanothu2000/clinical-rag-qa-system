"""
rag_chain.py
------------
Main RAG chain built with LangChain Expression Language (LCEL).
Supports GPT-4 and LLaMA 2 (via Ollama) as LLM backends.
Includes conversational memory and source-attributed responses.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .prompts import CLINICAL_QA_SYSTEM_PROMPT, CONDENSE_QUESTION_PROMPT

logger = logging.getLogger(__name__)


class ClinicalRAGChain:
    """
    Conversational RAG chain for clinical Q&A.

    Parameters
    ----------
    retriever : BaseRetriever
        A LangChain-compatible retriever (Pinecone, FAISS, ensemble, etc.)
    llm_model : str
        OpenAI model name (default: gpt-4o).
    temperature : float
        LLM sampling temperature (default: 0.0 for deterministic clinical answers).
    memory_window : int
        Number of past turns to keep in memory (default: 5).
    streaming : bool
        Stream LLM tokens to stdout (default: False).
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_model: str = "gpt-4o",
        temperature: float = 0.0,
        memory_window: int = 5,
        streaming: bool = False,
    ):
        self.retriever = retriever
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else []

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            streaming=streaming,
            callbacks=callbacks,
        )

        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        self.chain = self._build_chain()

    # ------------------------------------------------------------------
    # Chain construction
    # ------------------------------------------------------------------

    def _build_chain(self) -> ConversationalRetrievalChain:
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": ChatPromptTemplate.from_messages([
                    ("system", CLINICAL_QA_SYSTEM_PROMPT),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}"),
                    ("human", "Context:\n{context}"),
                ])
            },
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def query(self, question: str) -> Dict[str, Any]:
        """
        Run a clinical question through the RAG chain.

        Returns
        -------
        dict with keys:
            answer        : str  — LLM response
            source_docs   : list — retrieved document chunks
            num_sources   : int  — number of sources used
        """
        logger.info("Query: %s", question[:120])

        result = self.chain({"question": question})

        sources = result.get("source_documents", [])
        formatted_sources = [
            {
                "content": doc.page_content[:300],
                "source": doc.metadata.get("source", "unknown"),
                "note_type": doc.metadata.get("note_type", ""),
                "encounter_date": doc.metadata.get("encounter_date", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
            }
            for doc in sources
        ]

        return {
            "answer": result["answer"],
            "source_docs": formatted_sources,
            "num_sources": len(sources),
        }

    def reset_memory(self) -> None:
        """Clear conversation history."""
        self.memory.clear()
        logger.info("Conversation memory cleared.")
