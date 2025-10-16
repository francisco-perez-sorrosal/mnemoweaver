"""Main module for the Text Retrieval MCP server with MCPB compatibility."""

import argparse
import os
import sys
import traceback

from loguru import logger
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent

from mnemoweaver.bm25 import BM25Index
from mnemoweaver.chunking import chunk_by_section
from mnemoweaver.hippocampus import Hippocampus
from mnemoweaver.mem_compressor import SentenceTransformerMemCompressor
from mnemoweaver.storage import InMemoryBasicDocumentStorage
from mnemoweaver.vector_index import DistanceMetric, VectorIndex

# Configure logger for MCPB environment
logger.remove()
logger.add(
    sys.stdout, 
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)   
    
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Mnemoweaver MCP Server")
    parser.add_argument(
        "--transport", 
        choices=["stdio", "streamable-http"], 
        help="Transport method (overrides TRANSPORT environment variable)"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to bind to (overrides HOST environment variable)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        help="Port to bind to (overrides PORT environment variable)"
    )
    return parser.parse_args()

# Parse CLI arguments
args = parse_args()

# Configure transport and statelessness
trspt = args.transport or os.environ.get("TRANSPORT", "stdio")
stateless_http = False

match trspt:
    case "sse":
        raise ValueError("SSE transport is deprecated. Using stdio (locally) or streamable-http (remote) instead.")
    case "streamable-http":
        trspt = "streamable-http"
        stateless_http = True
    case _:
        trspt = "stdio"
        stateless_http = False


# Initialize FastMCP server with error handling
try:
    host = args.host or os.environ.get("HOST", "0.0.0.0")  
    port = args.port or int(os.environ.get("PORT", 10000))  
    mcp = FastMCP("linkedin_mcp_fps", stateless_http=stateless_http, host=host, port=port)
    logger.info(f"FastMCP server initialized with transport: {trspt}, host: {host}, port: {port}")
except Exception as e:
    logger.error(f"Failed to initialize FastMCP server: {e}")
    raise

doc_storage = InMemoryBasicDocumentStorage()
bm25_index = BM25Index(doc_storage)
vector_index = VectorIndex(doc_storage, distance_metric=DistanceMetric.COSINE)

hippocampus = Hippocampus(bm25_index, vector_index, reranker_fn=None)


@mcp.tool()
async def memorize(
    memory: str, context: Context) -> list[base.Message]:
    """
    Adds a memory to the storage.

    Args:
        memory: The memory to add to the storage
        context: The context of the mcp server

    Returns:
        list[base.Message]: Instructions for memory addition process
    """
    await context.log('info', message=f"Adding memory to the knowledge base...")
    chunks = chunk_by_section(memory)
    memories = [{"content": chunk} for chunk in chunks]
    await bm25_index.add_memories(memories)
    await vector_index.add_memories(memories)
    return [base.AssistantMessage(content="Memory added to the knowledge base")]

@mcp.tool()
async def retrieve(
    query: str, context: Context) -> list[base.Message]:
    """
    Retrieves memories from the storage.
    """
    retrieved_memories = await hippocampus.retrieve(query, k=3, k_rrf=60, ctx=context)
    string_memories = "\n".join([f"[{retrieved_memory.memory.id}: Score={retrieved_memory.score}] {retrieved_memory.memory.content}" for retrieved_memory in retrieved_memories])
    return [base.AssistantMessage(content=string_memories)]


if __name__ == "__main__":
    try:
        # Log environment information for debugging
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python path: {sys.path}")
        logger.info(f"Environment variables: TRANSPORT={os.environ.get('TRANSPORT')}, HOST={os.environ.get('HOST')}, PORT={os.environ.get('PORT')}")
        
        # Initialize and run the server with the specified transport
        logger.info(f"Starting LinkedIn MCP server with {trspt} transport ({host}:{port}) and stateless_http={stateless_http}...")
        
        # Additional pre-flight checks
        if trspt == "stdio":
            logger.info("Using stdio transport - suitable for local integration")
        elif trspt == "streamable-http":
            logger.info(f"Using HTTP transport - server will be accessible at http://{host}:{port}/mcp")
        
        # Start the server
        mcp.run(transport=trspt)
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error starting server: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
