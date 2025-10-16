from ast import List
import hashlib
import math
from numpy import ndarray

def generate_document_id(text: str) -> str:
    """Generate a unique document id based on the text."""
    normalized = text.lower().strip()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def euclidean_distance(vec1: ndarray, vec2: ndarray) -> float:
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same dimension")
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(vec1, vec2)))

def dot_product(vec1: ndarray, vec2: ndarray) -> float:
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same dimension")
    return sum(p * q for p, q in zip(vec1, vec2))

def magnitude(vec: ndarray) -> float:
    return math.sqrt(sum(x * x for x in vec))

def cosine_distance(vec1: ndarray, vec2: ndarray) -> float:
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same dimension")

    mag1 = magnitude(vec1)
    mag2 = magnitude(vec2)

    if mag1 == 0 and mag2 == 0:
        return 0.0
    elif mag1 == 0 or mag2 == 0:
        return 1.0

    dot_prod = dot_product(vec1, vec2)
    cosine_similarity = dot_prod / (mag1 * mag2)
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

    return 1.0 - cosine_similarity
