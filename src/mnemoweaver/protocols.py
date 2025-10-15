from typing import Any, Dict, List, Protocol, Tuple

from mnemoweaver.models import RetrievedMemory


# Remember that in Python's structural typing (Protocol), the implementing class 
# does not need to have exactly the same parameters as the protocol method. 
# Python uses structural subtyping for protocols, which means:
#
# What Matters for Protocol Compliance
#    - Method name must match ✅
#    - Method signature must be compatible (can accept the protocol's parameters)
#    - Return type should be compatible

class MemoryIndex(Protocol):
    name: str
    
    def add_memory(self, memory: Dict[str, Any]) -> None: ...

    def add_memories(self, memories: List[Dict[str, Any]]) -> None: ...

    def retrieve(self, query: str, k: int = 1) -> List[RetrievedMemory]: ...
