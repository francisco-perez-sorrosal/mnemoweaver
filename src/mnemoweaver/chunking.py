import re

def chunk_by_section(document_text):
    pattern = r"\n## "
    return re.split(pattern, document_text)
