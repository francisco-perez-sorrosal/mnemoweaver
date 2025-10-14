import math

from collections import Counter
from typing import Callable, Any, List, Dict, Tuple, Optional

from mcp.server.fastmcp import Context

from mnemoweaver.storage import InMemoryBasicDocumentStorage


class BM25Index:
    """BM25 ranking algorithm implementation with separate document storage."""
    
    def __init__(
        self,
        storage: Optional[InMemoryBasicDocumentStorage] = None,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ):
        self.k1 = k1
        self.b = b
        self._storage = storage if storage else InMemoryBasicDocumentStorage(tokenizer)
        self._doc_freqs: Dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._idf: Dict[str, float] = {}
        self._index_built: bool = False

    async def add_document(self, document: Dict[str, Any], rebuild_index: bool, context: Context) -> None:
        """Add a document to storage and update index stats.
        
        Args:
            document: Dictionary containing at least a 'content' key
        """
        self._storage.add_document(document)
        await context.log('info', message=f"Document added to the storage: {document}")
        if rebuild_index:
            await context.log('info', message=f"Rebuilding index...")
            self._rebuild_index()

    async def add_documents(self, documents: List[Dict[str, Any]], context: Context) -> None:
        """Add multiple documents to storage and update index.
        
        Args:
            documents: List of document dictionaries
        """
        for document in documents:
            await self.add_document(document, rebuild_index=False, context=context)
            
        await context.log('info', message=f"Index rebuilt after adding documents to the storage")
        self._rebuild_index()

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
        for i in range(len(self._storage)):
            doc_tokens = self._storage.get_document_tokens(i)
            seen_tokens = set()
            for token in doc_tokens:
                if token not in seen_tokens:
                    self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                    seen_tokens.add(token)

        # Calculate average document length
        total_length = sum(self._storage.get_document_length(i) for i in range(len(self._storage)))
        self._avg_doc_len = total_length / len(self._storage)

        # Calculate IDF scores
        self._calculate_idf()
        self._index_built = True

    def _compute_bm25_score(
        self, query_tokens: List[str], doc_index: int
    ) -> float:
        """Compute BM25 score for a query against a specific document.
        
        Args:
            query_tokens: Tokenized query terms
            doc_index: Index of document in storage
            
        Returns:
            BM25 score for the document
        """
        score = 0.0
        doc_tokens = self._storage.get_document_tokens(doc_index)
        doc_term_counts = Counter(doc_tokens)
        doc_length = self._storage.get_document_length(doc_index)

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

    def search(
        self,
        query: Any,
        k: int = 1,
        score_normalization_factor: float = 0.1,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for documents using BM25 ranking.
        
        Args:
            query: Query string to search for
            k: Number of top results to return
            score_normalization_factor: Factor for score normalization
            
        Returns:
            List of (document, score) tuples sorted by relevance
            
        Raises:
            TypeError: If query is not a string
            ValueError: If k is not positive
        """
        if not self._storage:
            return []

        if isinstance(query, str):
            query_text = query
        else:
            raise TypeError("Query must be a string for BM25Index.")

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        if not self._index_built:
            self._build_index()

        if self._avg_doc_len == 0:
            return []

        # Get tokenizer from storage
        query_tokens = self._storage._tokenizer(query_text)
        if not query_tokens:
            return []

        raw_scores = []
        for i in range(len(self._storage)):
            raw_score = self._compute_bm25_score(query_tokens, i)
            if raw_score > 1e-9:
                doc = self._storage.get_document(i)
                raw_scores.append((raw_score, doc))

        raw_scores.sort(key=lambda item: item[0], reverse=True)

        normalized_results = []
        for raw_score, doc in raw_scores[:k]:
            normalized_score = math.exp(-score_normalization_factor * raw_score)
            normalized_results.append((doc, normalized_score))

        normalized_results.sort(key=lambda item: item[1])

        return normalized_results

    def __len__(self) -> int:
        """Return number of documents in storage."""
        return len(self._storage)

    def __repr__(self) -> str:
        """String representation of the BM25 index."""
        return f"BM25Index(count={len(self)}, k1={self.k1}, b={self.b}, index_built={self._index_built})"
    