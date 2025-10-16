from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from mnemoweaver.mem_compressor import MemCompressor, SentenceTransformerMemCompressor
from mnemoweaver.models import RetrievedMemory
from mnemoweaver.storage import InMemoryBasicDocumentStorage, InMemoryVectorizedDocumentStorage
from mnemoweaver.utils import cosine_distance, euclidean_distance
from mnemoweaver.protocols import MemoryIndex


class DistanceMetric(str, Enum):
    """Supported distance metrics for vector similarity."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class VectorIndex(MemoryIndex):
    """Vector index implementation using a memory compressor and a distance metric."""
    name: str = "vector_index"
    
    def __init__(
        self,
        document_storage: InMemoryBasicDocumentStorage,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
    ):
        self._document_storage = document_storage
        self._vector_storage = InMemoryVectorizedDocumentStorage()
        self._mem_compressor = SentenceTransformerMemCompressor()
        try:
            self._distance_metric = DistanceMetric(distance_metric)
        except ValueError:
            valid_metrics = [metric.value for metric in DistanceMetric]
            raise ValueError(f"distance_metric must be one of {valid_metrics}, got '{distance_metric}'")

    async def add_memory(self, memory: Dict[str, Any]) -> str:
        if "content" not in memory:
            raise ValueError(
                "Memory dictionary must contain a 'content' key."
            )

        content = memory["content"]
        if not isinstance(content, str):
            raise TypeError("Memory 'content' must be a string.")

        # Add memory to both document and vector storages
        mem_id = self._document_storage.add_document(memory)
        memory_with_id = memory.copy()
        memory_with_id["id"] = mem_id
        embedding_batch = self._mem_compressor.compress(memory_with_id, input_type="text")
        self._vector_storage.add_documents(embedding_batch)
        return mem_id

    async def add_memories(self, memories: List[Dict[str, Any]]) -> List[str]:
        mem_ids = []
        for memory in memories:
            mem_id = await self.add_memory(memory)
            mem_ids.append(mem_id)
        return mem_ids
    
    def retrieve(self, query: str, k: int = 1) -> List[RetrievedMemory]:
        logger.info(f"Retrieving {k} memories for query: {query}")
        query_embedding_batch = self._mem_compressor.compress({"content": query, "id": ""}, input_type="query")
        query_vector = query_embedding_batch.embeddings[0].values

        # Validate query vector dimension matches stored vectors
        if len(self._vector_storage.vectorized_documents) > 0:
            expected_dim = self._vector_storage.vectorized_documents[0].dimensions
            query_dim = len(query_vector)
            if query_dim != expected_dim:
                raise ValueError(f"Query vector dimension mismatch. Expected {expected_dim}, got {query_dim}")

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        match self._distance_metric:
            case DistanceMetric.COSINE:
                dist_func = cosine_distance
            case DistanceMetric.EUCLIDEAN:
                dist_func = euclidean_distance
            case _:
                raise ValueError(f"Invalid distance metric: {self._distance_metric}")

        retrieved_memories = []
        for stored_vector in self._vector_storage.vectorized_documents:
            distance = dist_func(query_vector, stored_vector.values) # type: ignore
            retrieved_memories.append(RetrievedMemory(memory=self._document_storage.get_document(stored_vector.document_id).memory, score=distance))
            
        retrieved_memories.sort(key=lambda item: item.score)
        
        return retrieved_memories[:k]
    
    def __str__(self) -> str:
        return f"VectorIndex(count={len(self._vector_storage.vectorized_documents)} (Doc Storage id: {id(self._document_storage)}), distance_metric={self._distance_metric})"
