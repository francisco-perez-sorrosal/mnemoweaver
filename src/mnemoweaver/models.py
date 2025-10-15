from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field


class Memory(BaseModel):
    """Model representing a memory with basic identification and content."""
    
    id: str = Field(default="unknown", description="Unique memory identifier")
    content: str = Field(
        description="The content of the memory"
    )
    
    def __str__(self) -> str:
        """String representation of the memory."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Memory(id={self.id}, content='{content_preview}')"


class RetrievedMemory(BaseModel):
    """Model representing a retrieved memory with its relevance score."""
    
    memory: Memory = Field(description="The memory containing the information")
    score: float = Field(description="Relevance score for this memory (higher is more)", ge=0.0)
    
    def __str__(self) -> str:
        return f"RetrievedMemory(id={self.memory.id}, score={self.score:.4f}, content='{self.memory.content[:50]}{'...' if len(self.memory.content) > 50 else ''}')"


class RetrieveRequest(BaseModel):
    """Model for memory retrieval request parameters."""
    
    query: str = Field(
        description="Query string to search for memories",
        min_length=1
    )
    k: int = Field(
        default=1,
        description="Number of top memories to retrieve",
        ge=1
    )
    k_rrf: int = Field(
        default=60,
        description="Parameter for Reciprocal Rank Fusion scoring",
        ge=0
    )
    
    def __str__(self) -> str:
        """String representation of the retrieve request."""
        return f"RetrieveRequest(query='{self.query}', k={self.k}, k_rrf={self.k_rrf})"


class RetrieveResponse(BaseModel):
    """Model for memory retrieval response."""
    
    memories: List[RetrievedMemory] = Field(
        description="List of retrieved memories with their scores"
    )
    query: str = Field(
        description="The original query that was processed"
    )
    total_found: int = Field(
        description="Total number of memories found"
    )
    
    @property
    def top_memory(self) -> Optional[RetrievedMemory]:
        """Get the highest scoring memory."""
        return self.memories[0] if self.memories else None
    
    def __str__(self) -> str:
        """String representation of the retrieve response."""
        return f"RetrieveResponse(query='{self.query}', found={self.total_found} memories)"
