# Embedding Generation
#https://milvus.io/blog/hands-on-rag-with-qwen3-embedding-and-reranking-models-using-milvus.md
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from loguru import logger
from numpy import dtype, ndarray
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import Tensor

class EmbeddingVector(BaseModel):
    """A single embedding vector with validation."""
    
    values: ndarray = Field(description="The embedding vector values")
    document_id: str = Field(description="The id of the document that the embedding belongs to")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('values')
    @classmethod
    def validate_values(cls, v):
        import numpy as np
        # Check if empty using size instead of truthiness
        if isinstance(v, np.ndarray):
            if v.size == 0:
                raise ValueError("Embedding vector cannot be empty")
            # For numpy arrays, check dtype instead of iterating
            if not np.issubdtype(v.dtype, np.number):
                raise ValueError("All embedding values must be numbers")
        else:
            # For lists or other sequences
            if len(v) == 0:
                raise ValueError("Embedding vector cannot be empty")
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("All embedding values must be numbers")
        return v
    
    @property
    def dimensions(self) -> int:
        """Get the dimension of the embedding vector."""
        return len(self.values)
    
    def __str__(self) -> str:
        return f"EmbeddingVector(dim={self.dimensions}, norm={sum(x**2 for x in self.values)**0.5:.4f})"


class EmbeddingBatch(BaseModel):
    """A batch of embedding vectors."""
    
    embeddings: List[EmbeddingVector] = Field(description="List of embedding vectors")
    
    @field_validator('embeddings')
    @classmethod
    def validate_embeddings(cls, v):
        if not v:
            raise ValueError("Embedding batch cannot be empty")
        return v
    
    @property
    def dimensions(self) -> int:
        """Get the dimension of the embeddings (assumes all have same dimension)."""
        return len(self.embeddings[0].values)
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __str__(self) -> str:
        return f"EmbeddingBatch(count={len(self)}, dimensions={self.dimensions})"


class MemCompressor(ABC):
    """Abstract base class for memory compression using embeddings."""
    
    def __init__(self, model: str = "voyage-3-large", emb_dim: int = 1024):
        self.model = model
        self.emb_dim = emb_dim

    @abstractmethod
    def _embed(self, chunks: Dict[str, Any] | List[Dict[str, Any]], input_type: str = "text") -> EmbeddingBatch:
        """Generate embeddings for text chunks.
        
        Args:
            chunks: Single chunk or list of chunks to embed
            input_type: The type of input to embed.
        Returns:
            EmbeddingBatch containing the generated embeddings
        """
        pass

    def compress(self, memories: Dict[str, Any] | List[Dict[str, Any]], input_type: str = "text") -> EmbeddingBatch:
        """Generate embeddings for text chunks and return as EmbeddingBatch.
        
        Args:
            memories: Single memory or list of memories to embed
            input_type: The type of input to embed.
                
        Returns:
            EmbeddingBatch containing the generated embeddings
        """
        is_list = isinstance(memories, list)
        input_chunks = memories if is_list else [memories]
        embedding_batch = self._embed(input_chunks, input_type)
        
        return embedding_batch

from sentence_transformers import SentenceTransformer

class SentenceTransformerMemCompressor(MemCompressor):
    """Concrete implementation using SentenceTransformer embeddings."""
    
    def __init__(self, model: str = "Qwen/Qwen3-Embedding-0.6B", emb_dim: int = 1024, input_type: str = "query"):
        super().__init__(model=model, emb_dim=emb_dim)
        self.input_type = input_type
        self.client = SentenceTransformer(model_name_or_path=model)
    
    def _embed(self, chunks: Dict[str, Any] | List[Dict[str, Any]], input_type: str = "text") -> EmbeddingBatch:       
        is_list = isinstance(chunks, list)
        input_chunks = chunks if is_list else [chunks]
        chunk_texts = [chunk.get("content", "") for chunk in input_chunks]
        document_ids = [chunk.get("id", "") for chunk in input_chunks]
        result : Union[List[Tensor], ndarray, Tensor] = self.client.encode(chunk_texts, prompt_name=self.input_type)
        logger.info(f"Embedding result: {result.shape}")
        return EmbeddingBatch(embeddings=[EmbeddingVector(values=vector, document_id=document_id) for vector, document_id in zip(result, document_ids)])
