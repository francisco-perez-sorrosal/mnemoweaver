from typing import Callable, Any, List, Dict, Optional

from loguru import logger

from mnemoweaver.tokenization import BasicTokenizer


class InMemoryBasicDocumentStorage:
    """Handles basic document storage and tokenization."""
    
    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None):
        """Initialize document storage with optional tokenizer.
        
        Args:
            tokenizer: Optional tokenizer function. Defaults to BasicTokenizer.
        """
        self.documents: List[Dict[str, Any]] = []
        self._corpus_tokens: List[List[str]] = []
        self._doc_len: List[int] = []
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
        
        doc_tokens = self._tokenizer(content)
        
        self.documents.append(document)
        self._corpus_tokens.append(doc_tokens)
        self._doc_len.append(len(doc_tokens))
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add multiple documents to storage.
        
        Args:
            documents: List of document dictionaries
        """
        for i, doc in enumerate(documents):
            logger.info(f"Adding document {i} of {len(documents)}")
            self.add_document(doc)
    
    def get_document(self, index: int) -> Dict[str, Any]:
        """Get document by index.
        
        Args:
            index: Document index
            
        Returns:
            Document dictionary
        """
        return self.documents[index]
    
    def get_document_tokens(self, index: int) -> List[str]:
        """Get tokenized document by index.
        
        Args:
            index: Document index
            
        Returns:
            List of tokens for the document
        """
        return self._corpus_tokens[index]
    
    def get_document_length(self, index: int) -> int:
        """Get document length (token count) by index.
        
        Args:
            index: Document index
            
        Returns:
            Number of tokens in the document
        """
        return self._doc_len[index]
    
    def __len__(self) -> int:
        """Return number of stored documents."""
        return len(self.documents)
    
    def __repr__(self) -> str:
        """String representation of the storage."""
        return f"DocumentStorage(count={len(self)})"
