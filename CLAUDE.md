# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MnemoWeaver is an MCP (Model Context Protocol) server that provides text indexing and retrieval functionality for a knowledge base. It implements a BM25-based search system with a multi-index retrieval architecture called "Hippocampus" that uses Reciprocal Rank Fusion (RRF) to combine results from multiple search indexes.

## Development Environment

This project uses **Pixi** for environment management. Always use `pixi run` to execute commands in the proper Python environment.

### Common Commands

```bash
# Run the MCP server locally
pixi run mcps

# Run the server in debug/test mode
pixi run test

# Package the server as an MCP bundle
pixi run pack

# Update MCP bundle dependencies (requires uv)
pixi run update-mcpb-deps
pixi run mcp-bundle
```

## Project Architecture

### Core Components

1. **Memory Storage** (`storage.py`)
   - `InMemoryBasicDocumentStorage`: Handles document storage and tokenization
   - Stores documents with their tokenized representations
   - Uses `BasicTokenizer` by default for text tokenization

2. **Search Index** (`bm25.py`)
   - `BM25Index`: Implements BM25 ranking algorithm
   - Configurable parameters: k1 (default: 1.5), b (default: 0.75)
   - Automatically rebuilds index when documents are added
   - Implements the `MemoryIndex` protocol

3. **Multi-Index Retrieval** (`hippocampus.py`)
   - `Hippocampus`: Meta-retrieval system combining multiple indexes
   - Uses Reciprocal Rank Fusion (RRF) to merge results from different indexes
   - Supports optional reranking via custom reranker functions
   - RRF parameters: k_rrf (default: 60)

4. **MCP Server** (`main.py`)
   - FastMCP-based server with two main tools:
     - `memorize`: Adds content to the knowledge base (chunks by markdown sections)
     - `retrieve`: Searches and retrieves relevant memories
   - Supports multiple transports: stdio (local), streamable-http (remote)
   - Environment variables: TRANSPORT, HOST, PORT

5. **Data Models** (`models.py`)
   - `Memory`: Basic memory with id and content
   - `RetrievedMemory`: Memory with relevance score
   - `RetrieveRequest/Response`: Request/response models

6. **Protocols** (`protocols.py`)
   - `MemoryIndex`: Protocol defining the interface for memory indexes
   - Uses Python's structural typing (Protocol) for flexible implementations

7. **Utilities**
   - `chunking.py`: Splits documents by markdown sections (## headers)
   - `tokenization.py`: Basic text tokenizer using regex word splitting

### Data Flow

```
User Input (memorize)
  → chunk_by_section()
  → InMemoryBasicDocumentStorage.add_documents()
  → BM25Index.add_memories()
  → Index rebuilding

User Query (retrieve)
  → Hippocampus.retrieve()
  → Query all registered indexes
  → Reciprocal Rank Fusion scoring
  → Optional reranking
  → Return top-k results
```

### Key Design Patterns

- **Protocol-based design**: `MemoryIndex` protocol allows multiple index implementations
- **Separation of concerns**: Storage, indexing, and retrieval are separate layers
- **Structural subtyping**: Protocol implementations don't need exact parameter matching
- **Multi-index fusion**: Hippocampus combines multiple retrieval strategies

## Testing and Development

Currently, there are no formal test files. When testing changes:
- Use `pixi run test` to run the server in debug mode
- Check logs in stderr (loguru is configured to output there)
- The server supports both stdio and streamable-http transports

## Code Style

- Python 3.13 required
- Black formatter: line length 88, target Python 3.13
- isort profile: black
- mypy: strict mode enabled (warnings_as_errors=false)
- Type hints should be used throughout

## MCP Bundle Packaging

The project can be packaged as an MCP bundle (.mcpb):
1. Update dependencies: `pixi run update-mcpb-deps`
2. Install to lib/: `pixi run mcp-bundle`
3. Pack: `pixi run pack`

The manifest.json defines the server configuration for MCP deployment.
