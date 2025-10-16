import math

from collections import Counter
from typing import Callable, Any, List, Dict, Tuple, Optional

from loguru import logger

from mnemoweaver.models import Memory, RetrievedMemory
from mnemoweaver.storage import InMemoryBasicDocumentStorage
from mnemoweaver.protocols import MemoryIndex


class BM25Index(MemoryIndex):
    """BM25 ranking algorithm implementation with separate document storage."""

    name: str = "bm25"
    
    def __init__(
        self,
        storage: InMemoryBasicDocumentStorage,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ):
        self.k1 = k1
        self.b = b
        self._storage = storage
        self._doc_freqs: Dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._idf: Dict[str, float] = {}
        self._index_built: bool = False
        self._tokenizer = tokenizer if tokenizer else self._storage._tokenizer

        
    def _rebuild_index(self) -> None:
        """Rebuild the index statistics from current storage."""
        self._doc_freqs = {}
        self._index_built = False
        self._build_index()

    def _calculate_idf(self) -> None:
        """Calculate Inverse Document Frequency for all terms."""
        N = len(self._storage)
        self._idf = {}
        for term, freq in self._doc_freqs.items():
            idf_score = math.log(((N - freq + 0.5) / (freq + 0.5)) + 1)
            self._idf[term] = idf_score

    def _build_index(self) -> None:
        """Build the BM25 index from stored documents."""
        if not self._storage:
            self._avg_doc_len = 0.0
            self._idf = {}
            self._index_built = True
            return

        # Calculate document frequencies
        self._doc_freqs = {}
        for id, document in self._storage.documents.items():
            seen_tokens = set()
            for token in document.tokens:
                if token not in seen_tokens:
                    self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                    seen_tokens.add(token)

        # Calculate average document length
        self._avg_doc_len = self._storage.get_total_length() / len(self._storage)

        # Calculate IDF scores
        self._calculate_idf()
        self._index_built = True

    def _compute_bm25_score(self, query_tokens: List[str], doc_id: str) -> float:
        """Compute BM25 score for a query against a specific document.
        
        Args:
            query_tokens: Tokenized query terms
            doc_id: ID of document in storage
            
        Returns:
            BM25 score for the document
        """
        score = 0.0
        doc_tokens = self._storage.get_document_tokens(doc_id)
        doc_term_counts = Counter(doc_tokens)
        doc_length = self._storage.get_document(doc_id).length

        for token in query_tokens:
            if token not in self._idf:
                continue

            idf = self._idf[token]
            term_freq = doc_term_counts.get(token, 0)

            numerator = idf * term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * (doc_length / self._avg_doc_len)
            )
            score += numerator / (denominator + 1e-9)

        return score

    # MemoryIndex protocol implementation

    async def add_memory(self, memory: Dict[str, Any], rebuild_index: bool) -> None:
        """Add a document to storage and update index stats.
        
        Args:
            memory: Dictionary containing at least a 'content' key
        """
        self._storage.add_document(memory)
        logger.info(f"Memory added to the storage: {memory}")
        if rebuild_index:
            logger.info(f"Rebuilding index...")
            self._rebuild_index()

    async def add_memories(self, memories: List[Dict[str, Any]]) -> None:
        """Add multiple documents to storage and update index.
        
        Args:
            memories: List of memory dictionaries with 'content' key
        """
        for memory in memories:
            await self.add_memory(memory, rebuild_index=False)
            
        self._rebuild_index()
        logger.info(f"Index rebuilt after adding memories to the storage")

    def retrieve(
        self,
        query: str,
        k: int = 1,
        score_normalization_factor: float = 0.1,
    ) -> List[RetrievedMemory]:
        """Search for documents using BM25 ranking.

        Args:
            query: Query string to search for
            k: Number of top results to return
            score_normalization_factor: Factor for score normalization

        Returns:
            List of RetrievedMemory objects sorted by relevance (highest score first)

        Raises:
            ValueError: If k is not positive
        """
        if not self._storage:
            return []

        query_text = query

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        logger.info(f"Retrieving {k} memories for query: {query}")
        
        if not self._index_built:
            self._build_index()

        if self._avg_doc_len == 0:
            return []

        # Get tokenizer from storage
        query_tokens = self._tokenizer(query_text)
        if not query_tokens:
            return []

        # Compute BM25 scores for all memories and sort them
        scored_memories = []
        for id in self._storage.get_document_ids():
            raw_score = self._compute_bm25_score(query_tokens, id)
            if raw_score > 1e-9:
                memory = self._storage.get_document(id).memory
                scored_memories.append(RetrievedMemory(memory=memory, score=raw_score))

        # Sort by score descending (highest scores first)
        scored_memories.sort(key=lambda item: item.score, reverse=True)

        # Take top k and normalize scores if needed
        top_k_memories = scored_memories[:k]

        # Apply score normalization and return as RetrievedMemory objects
        normalized_results = []
        for retrieved_memory in top_k_memories:
            normalized_score = math.exp(-score_normalization_factor * retrieved_memory.score)
            normalized_results.append(
                RetrievedMemory(memory=retrieved_memory.memory, score=normalized_score)
            )

        return normalized_results

    def __len__(self) -> int:
        """Return number of documents in storage."""
        return len(self._storage)

    def __repr__(self) -> str:
        """String representation of the BM25 index."""
        return f"BM25Index(count={len(self)} (Storage id: {id(self._storage)}), k1={self.k1}, b={self.b}, index_built={self._index_built})"
    