import random
import string
from typing import Callable, Any, List, Dict, Optional

from loguru import logger

from mnemoweaver.models import Memory
from mnemoweaver.tokenization import BasicTokenizer

from pydantic import BaseModel, computed_field

class InMemoryDocument(BaseModel):
    memory: Memory
    tokens: List[str]
    
    @computed_field
    @property
    def length(self) -> int:
        """Calculate the length as the number of tokens."""
        return len(self.tokens)

class InMemoryBasicDocumentStorage:
    """Handles basic document storage and tokenization."""
    
    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None):
        """Initialize document storage with optional tokenizer.
        
        Args:
            tokenizer: Optional tokenizer function. Defaults to BasicTokenizer.
        """
        self.documents: Dict[str, InMemoryDocument] = {}
        self._tokenizer = tokenizer if tokenizer else BasicTokenizer()
    
    def add_document(self, document: Dict[str, Any]) -> None:
        """Add a single document to storage.
        
        Args:
            document: Dictionary containing at least a 'content' key with string value
            
        Raises:
            ValueError: If document doesn't contain 'content' key
            TypeError: If content is not a string
        """
        if "content" not in document:
            raise ValueError("Document dict must contain a 'content' str key.")
        
        content = document.get("content", "")
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")
        
        default_doc_id = "".join(random.choices(string.ascii_letters + string.digits, k=4))
        if "id" not in document:
            logger.warning(f"No implicit ID found in memory, generating one")
            id = default_doc_id
        else:
            id = str(document.get("id")) if document.get("id") else default_doc_id
        doc_tokens = self._tokenizer(content)

        self.documents[id] = InMemoryDocument(memory=Memory(id=str(id), content=content), tokens=doc_tokens)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add multiple documents to storage.
        
        Args:
            documents: List of document dictionaries
        """
        for i, doc in enumerate(documents):
            logger.info(f"Adding document {i} of {len(documents)}")
            self.add_document(doc)
    
    def get_document(self, id: str) -> InMemoryDocument:
        """Get document by index.
        
        Args:
            id: Document id
            
        Returns:
            Document dictionary
        """
        return self.documents[id]
    
    def get_document_ids(self) -> List[str]:
        """Get all document ids.
        
        Returns:
            List of document ids
        """
        return list(self.documents.keys())
    
    def get_document_tokens(self, id: str) -> List[str]:
        """Get tokenized document by index.
        
        Args:
            index: Document index
            
        Returns:
            List of tokens for the document
        """
        return self.documents[id].tokens
    
    def get_document_length(self, id: str) -> int:
        """Get document length (token count) by index.
        
        Args:
            index: Document index
            
        Returns:
            Number of tokens in the document
        """
        return self.documents[id].length
    
    def get_total_length(self) -> int:
        return sum(self.get_document_length(id) for id in self.documents.keys())
    
    def __len__(self) -> int:
        """Return number of stored documents."""
        return len(self.documents)
    
    def __repr__(self) -> str:
        """String representation of the storage."""
        return f"InMemoryStorage(count={len(self)})"
