
import re
from typing import List


class BasicTokenizer:
    """A simple text tokenizer that splits text into words."""
    
    def __init__(self, pattern: str = r"\W+"):
        """Initialize the tokenizer with a regex pattern for splitting.
        
        Args:
            pattern: Regex pattern to split text on. Defaults to non-word characters.
        """
        self.pattern = pattern
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text into a list of words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokenized words
        """
        text = text.lower()
        tokens = re.split(self.pattern, text)
        return [token for token in tokens if token]
    
    def __call__(self, text: str) -> List[str]:
        """Allow the tokenizer to be called directly like a function.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokenized words
            
        Example:
            >>> tokenizer = BasicTokenizer()
            >>> tokens = tokenizer("Hello, world! This is a test.")
            >>> print(tokens)
            ['hello', 'world', 'this', 'is', 'a', 'test']
        """
        return self.tokenize(text)
