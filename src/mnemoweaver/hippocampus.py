import string

from loguru import logger
from typing import Callable
from mcp.server.fastmcp import Context
from typing_extensions import List, Optional, Dict, Any

from mnemoweaver.models import RetrievedMemory
from mnemoweaver.protocols import MemoryIndex


def calc_rrf_score(k_rrf: int, ranks: List[float]) -> float:
    """Calculate Reciprocal Rank Fusion (RRF) score for combining multiple retrieval results.
    
    Reciprocal Rank Fusion is a method for combining search results from multiple retrieval
    systems by giving higher scores to documents that appear at better ranks (lower numbers)
    across different systems. This helps identify consensus among heterogeneous retrieval
    approaches.
    
    Args:
        k_rrf: A constant parameter that controls the fusion behavior. Higher values
               reduce the impact of rank differences. Typical values range from 30-100,
               with 60 being a common default that provides good balance.
        ranks: List of ranks (positions) where the same document appeared in different
               retrieval systems. Lower rank numbers indicate better relevance.
               Infinite ranks (float("inf")) are ignored.
    
    Returns:
        RRF score as a float. Higher scores indicate better consensus across systems.
        The score is calculated as: sum(1.0 / (k_rrf + r) for each valid rank r)
    
    Example:
        >>> # Document appears at rank 1, 3, and 5 in different systems
        >>> ranks = [1, 3, 5]
        >>> score = calc_rrf_score(60, ranks)
        >>> # score = 1/(60+1) + 1/(60+3) + 1/(60+5) ≈ 0.0477
    """
    return sum(1.0 / (k_rrf + r) for r in ranks if r != float("inf"))


class Hippocampus:
    """Hippocampus-like memory retrieval system that combines multiple memory indexes.
    
    The Hippocampus acts as a meta-retrieval system that queries multiple specialized
    memory indexes (e.g., BM25, semantic search, keyword matching) and combines their
    results using Reciprocal Rank Fusion (RRF). This approach mimics how the biological
    hippocampus integrates information from different brain regions.
    
    Key Features:
    - Multi-index querying: Queries all registered memory indexes simultaneously
    - RRF scoring: Combines results using Reciprocal Rank Fusion for consensus ranking
    - Optional reranking: Supports post-processing with custom reranker functions
    - Memory aggregation: Identifies the same memory across different indexes
    
    The RRF algorithm helps find memories that are consistently relevant across
    different retrieval approaches, reducing the impact of individual system biases.
    """
    
    def __init__(
        self,
        *memory_indexes: MemoryIndex,
        reranker_fn: Optional[
            Callable[[List[RetrievedMemory], str, int], List[str]]
        ] = None,
    ):
        """Initialize the Hippocampus with memory indexes and optional reranker.
        
        Args:
            *memory_indexes: One or more MemoryIndex instances to query (e.g., BM25Index)
            reranker_fn: Optional function to rerank final results. Should accept
                        (memories_list, query, k) and return list of memory IDs
        """
        # The hippocampus is a collection of one or more memory indexes
        if len(memory_indexes) == 0:
            raise ValueError("At least one memory index must be provided")
        
        self._memory_indexes: Dict[str, MemoryIndex] = {index.name: index for index in memory_indexes}
        self._reranker_fn = reranker_fn
        
    def _validate_retrieve_input(self, query: str, k: int, k_rrf: int):
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if k_rrf < 0:
            raise ValueError("k_rrf must be non-negative.")


    # MemoryIndex protocol implementation
    
    async def add_memory(self, memory: Dict[str, Any], ctx: Optional[Context] = None):
        for index in self._memory_indexes.values():
            await ctx.log('info', message=f"Adding memory to index: {index.name}") if ctx else None
            index.add_memory(memory)

    async def add_memories(self, memories: List[Dict[str, Any]], ctx: Optional[Context] = None):
        for index in self._memory_indexes.values():
            await ctx.log('info', message=f"Adding memories to index: {index}") if ctx else None
            index.add_memories(memories)

    async def retrieve(self, query: str, k: int = 1, k_rrf: int = 60, ctx: Optional[Context] = None) -> List[RetrievedMemory]:
        """Retrieve memories using multi-index consensus via Reciprocal Rank Fusion.
        
        This method implements the core Hippocampus functionality:
        1. Queries all registered memory indexes with the given query
        2. Tracks where each memory appears in each index's ranking
        3. Calculates RRF scores based on consensus across indexes
        4. Returns the top-k memories with highest consensus scores
        
        The RRF approach is particularly effective when different memory indexes
        have complementary strengths (e.g., BM25 for keyword matching, semantic
        search for meaning, temporal indexes for recency).
        
        Args:
            query: Search query string
            k: Number of top memories to return
            k_rrf: RRF constant parameter (higher = less sensitive to rank differences)
            ctx: Optional logging context for MCPs
            
        Returns:
            List of RetrievedMemory objects sorted by RRF consensus score
        """
        self._validate_retrieve_input(query, k, k_rrf)

        all_memories = {memory_index.name: memory_index.retrieve(query, k=k * 5) for memory_index in self._memory_indexes.values()}
        await ctx.log('info', message=f"Retrieved memories from {len(all_memories)} indexes. Proceeding to rank and score...") if ctx else None
        logger.info(f"Retrieved memories from {len(all_memories)} indexes")

        memory_ranks = {}
        index_count = 0
        for index_name, index_memories in all_memories.items():            
            await ctx.log('info', message=f"Ranking memory {index_name} of {len(all_memories)} indexes") if ctx else None
            for rank, retrieved_memory in enumerate(index_memories):
                memory_id = retrieved_memory.memory.id
                if memory_id not in memory_ranks:
                    memory_ranks[memory_id] = {
                        "memory_obj": retrieved_memory,
                        "ranks": [float("inf")] * len(self._memory_indexes),
                    }
                memory_ranks[memory_id]["ranks"][index_count] = rank + 1
            index_count += 1
        logger.info(f"Calculated ranks for {len(memory_ranks)} memories")

        scored_memories: List[RetrievedMemory] = [
            RetrievedMemory(memory=memory_data["memory_obj"].memory, score=calc_rrf_score(k_rrf, memory_data["ranks"])) 
            for memory_data in memory_ranks.values()
        ]

        logger.info(f"Calculated RRF scores for {len(scored_memories)} memories")

        # Filter out memories with a score of 0 and sort by score, selecting the top k
        filtered_memories = [
            memory for memory in scored_memories if memory.score > 0
        ]
        logger.info(f"Filtered out {len(scored_memories) - len(filtered_memories)} memories with a score of 0")
        filtered_memories.sort(key=lambda x: x.score, reverse=True)
        

        selected_memories = filtered_memories[:k]

        # Rerank the memories only if a reranker function is provided!!! Otherwise, return the selected memories
        if self._reranker_fn is not None:           
            memory_lookup = {selected_memory.memory.id: selected_memory.memory for selected_memory in selected_memories}
            reranked_ids = self._reranker_fn(selected_memories, query, k)

            reranked_memories = []
            original_scores = {retrieved_memory.memory.id: retrieved_memory.score for retrieved_memory in selected_memories}

            for memory_id in reranked_ids:
                if memory_id in memory_lookup:
                    memory = memory_lookup[memory_id]
                    score = original_scores.get(memory_id, 0.0)
                    reranked_memories.append(RetrievedMemory(memory=memory, score=score))

            selected_memories = reranked_memories
            logger.info(f"Reranked {len(selected_memories)} memories")

        await ctx.log('info', message=f"Returning {len(selected_memories)} memories") if ctx else None
        return selected_memories
