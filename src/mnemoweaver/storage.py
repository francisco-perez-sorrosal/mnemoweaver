import random
import string
from typing import Callable, Any, List, Dict, Optional

from loguru import logger

from mnemoweaver.mem_compressor import EmbeddingBatch, EmbeddingVector
from mnemoweaver.models import Memory
from mnemoweaver.tokenization import BasicTokenizer

from pydantic import BaseModel, computed_field

from mnemoweaver.utils import generate_document_id

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
    
    def add_document(self, document: Dict[str, Any]) -> str:
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
        
        default_doc_id = generate_document_id(content)

        if "id" not in document:
            logger.warning(f"No implicit ID found in dictionary; generating one...")
            id = default_doc_id
        else:
            id = str(document.get("id")) if document.get("id") else default_doc_id
            logger.warning(f"Implicit ID found in dictionary; using it...")
        doc_tokens = self._tokenizer(content)

        document_exists = self.documents.get(id) is not None
        if document_exists:
            old_document = self.documents[id]
            logger.warning(f"""Document with ID {id} already exists\n
                           Old document:\n\n{old_document.memory.content[:100]}...\n\n
                           New document:\n\n{content[:100]}...\n\n""")
        else:
            new_document = InMemoryDocument(memory=Memory(id=str(id), content=content), tokens=doc_tokens)
            self.documents[id] = new_document
            logger.info(f"New document with ID {id} added")

        return id
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add multiple documents to storage.
        
        Args:
            documents: List of document dictionaries
        """
        ids = []
        for i, doc in enumerate(documents):
            logger.info(f"Adding document {i} of {len(documents)}")
            id = self.add_document(doc)
            ids.append(id)
        return ids
    
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

class InMemoryVectorizedDocumentStorage:
    """Handles vectorized document storage and tokenization."""
    
    def __init__(self):
        self.vectorized_documents: List[EmbeddingVector] = []

    def add_document(self, document: EmbeddingVector):
        """Add a single document to storage.
        
        Args:
            document: Embedding vector
        """
        self.vectorized_documents.append(document)        
        
    def add_documents(self, documents: EmbeddingBatch):
        """Add multiple documents to storage.
        
        Args:
            documents: List of embedding vectors
        """
        self.vectorized_documents.extend(documents.embeddings)
