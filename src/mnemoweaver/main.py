"""Main module for the Text Retrieval MCP server with MCPB compatibility."""

import os
import sys
import traceback

from loguru import logger
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent

from mnemoweaver.bm25 import BM25Index
from mnemoweaver.chunking import chunk_by_section
from mnemoweaver.storage import InMemoryBasicDocumentStorage

# Configure logger for MCPB environment
logger.remove()
logger.add(
    sys.stderr, 
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)   
    
# Configure transport and statelessness
trspt = "stdio"
stateless_http = False
match os.environ.get("TRANSPORT", trspt):
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
    host = os.environ.get("HOST", "0.0.0.0")  
    port = int(os.environ.get("PORT", 10000))  
    mcp = FastMCP("linkedin_mcp_fps", stateless_http=stateless_http, host=host, port=port)
    logger.info(f"FastMCP server initialized with transport: {trspt}, host: {host}, port: {port}")
except Exception as e:
    logger.error(f"Failed to initialize FastMCP server: {e}")
    raise

storage = InMemoryBasicDocumentStorage()
bm25_index = BM25Index(storage)


@mcp.tool()
async def add_memory(
    memory: str, context: Context) -> list[base.Message]:
    """
    Adds a memory to the storage.

    Args:
        memory: The memory to add to the storage
        context: The context of the mcp server

    Returns:
        list[base.Message]: Instructions for memory addition process
    """
    await context.log('info', message=f"Adding memory to the storage: {memory}")
    chunks = chunk_by_section(memory)
    memories = [{"content": chunk} for chunk in chunks]
    await bm25_index.add_documents(memories, context)
    return [base.AssistantMessage(content="Memory added to the storage")]


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
            logger.info("Using stdio transport - suitable for local Claude Desktop integration")
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
