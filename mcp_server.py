"""
Sample MCP Server for ChatGPT Integration

This server implements the Model Context Protocol (MCP) with search and fetch
capabilities designed to work with ChatGPT's chat and deep research features.
"""

import logging
import os
from typing import Dict, List, Any

from fastmcp import FastMCP
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
VECTOR_STORE_ID = os.environ.get("VECTOR_STORE_ID", "")

# Initialize OpenAI client
openai_client = OpenAI()

server_instructions = """
This MCP server provides management and search capabilities for OpenAI resources.
- Management: Use list_agents, list_vector_stores, and list_store_files to collect JSON data.
- Search: Use the search tool to find relevant documents.
- Retrieval: Use the fetch tool to retrieve complete document content.
- Ingestion: Use upload_file and delete_file to manage your content.
- Metadata: Use get_app_data for application state.
"""


def create_server():
    """Create and configure the MCP server with search and fetch tools."""

    # Initialize the FastMCP server
    mcp = FastMCP(name="Sample MCP Server",
                  instructions=server_instructions)

    @mcp.tool()
    async def search(query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for documents using OpenAI Vector Store search.

        This tool searches through the vector store to find semantically relevant matches.
        Returns a list of search results with basic information. Use the fetch tool to get
        complete document content.

        Args:
            query: Search query string. Natural language queries work best for semantic search.

        Returns:
            Dictionary with 'results' key containing list of matching documents.
            Each result includes id, title, text snippet, and optional URL.
        """
        if not query or not query.strip():
            return {"results": []}

        if not openai_client:
            logger.error("OpenAI client not initialized - API key missing")
            raise ValueError(
                "OpenAI API key is required for vector store search")

        # Search the vector store using OpenAI API
        logger.info(f"Searching {VECTOR_STORE_ID} for query: '{query}'")

        response = openai_client.vector_stores.search(
            vector_store_id=VECTOR_STORE_ID, query=query)

        results = []

        # Process the vector store search results
        if hasattr(response, 'data') and response.data:
            for i, item in enumerate(response.data):
                # Extract file_id, filename, and content
                item_id = getattr(item, 'file_id', f"vs_{i}")
                item_filename = getattr(item, 'filename', f"Document {i+1}")

                # Extract text content from the content array
                content_list = getattr(item, 'content', [])
                text_content = ""
                if content_list and len(content_list) > 0:
                    # Get text from the first content item
                    first_content = content_list[0]
                    if hasattr(first_content, 'text'):
                        text_content = first_content.text
                    elif isinstance(first_content, dict):
                        text_content = first_content.get('text', '')

                if not text_content:
                    text_content = "No content available"

                # Create a snippet from content
                text_snippet = text_content[:200] + "..." if len(
                    text_content) > 200 else text_content

                result = {
                    "id": item_id,
                    "title": item_filename,
                    "text": text_snippet,
                    "url":
                    f"https://platform.openai.com/storage/files/{item_id}"
                }

                results.append(result)

        logger.info(f"Vector store search returned {len(results)} results")
        return {"results": results}

    @mcp.tool()
    async def fetch(id: str) -> Dict[str, Any]:
        """
        Retrieve complete document content by ID for detailed
        analysis and citation. This tool fetches the full document
        content from OpenAI Vector Store. Use this after finding
        relevant documents with the search tool to get complete
        information for analysis and proper citation.

        Args:
            id: File ID from vector store (file-xxx) or local document ID

        Returns:
            Complete document with id, title, full text content,
            optional URL, and metadata

        Raises:
            ValueError: If the specified ID is not found
        """
        if not id:
            raise ValueError("Document ID is required")

        if not openai_client:
            logger.error("OpenAI client not initialized - API key missing")
            raise ValueError(
                "OpenAI API key is required for vector store file retrieval")

        logger.info(f"Fetching content from vector store for file ID: {id}")

        # Fetch file content from vector store
        content_response = openai_client.vector_stores.files.content(
            vector_store_id=VECTOR_STORE_ID, file_id=id)

        # Get file metadata
        file_info = openai_client.vector_stores.files.retrieve(
            vector_store_id=VECTOR_STORE_ID, file_id=id)

        # Extract content from paginated response
        file_content = ""
        if hasattr(content_response, 'data') and content_response.data:
            # Combine all content chunks from FileContentResponse objects
            content_parts = []
            for content_item in content_response.data:
                if hasattr(content_item, 'text'):
                    content_parts.append(content_item.text)
            file_content = "\n".join(content_parts)
        else:
            file_content = "No content available"

        # Use filename as title and create proper URL for citations
        filename = getattr(file_info, 'filename', f"Document {id}")

        result = {
            "id": id,
            "title": filename,
            "text": file_content,
            "url": f"https://platform.openai.com/storage/files/{id}",
            "metadata": None
        }

        # Add metadata if available from file info
        if hasattr(file_info, 'attributes') and file_info.attributes:
            result["metadata"] = file_info.attributes

        logger.info(f"Fetched vector store file: {id}")
        return result

    @mcp.tool()
    async def list_agents() -> Dict[str, Any]:
        """
        List all OpenAI Assistants (Agents) in the account.
        
        Returns:
            JSON object containing a list of assistants with their names, 
            IDs, instructions, and models.
        """
        try:
            assistants = openai_client.beta.assistants.list(limit=50)
            agents = []
            for a in assistants.data:
                agents.append({
                    "id": a.id,
                    "name": a.name,
                    "model": a.model,
                    "instructions": a.instructions[:200] + "..." if a.instructions and len(a.instructions) > 200 else a.instructions,
                    "created_at": a.created_at
                })
            return {"agents": agents, "count": len(agents)}
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def list_vector_stores() -> Dict[str, Any]:
        """
        List all available OpenAI Vector Stores.
        
        Returns:
            JSON object containing vector stores with IDs, names, 
            and file counts.
        """
        try:
            stores_response = openai_client.beta.vector_stores.list(limit=50)
            stores = []
            for s in stores_response.data:
                stores.append({
                    "id": s.id,
                    "name": s.name,
                    "file_counts": s.file_counts.__dict__ if hasattr(s, 'file_counts') else None,
                    "status": s.status,
                    "created_at": s.created_at
                })
            return {"vector_stores": stores, "count": len(stores)}
        except Exception as e:
            logger.error(f"Error listing vector stores: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def list_store_files(vector_store_id: str = None) -> Dict[str, Any]:
        """
        List all files within a specific vector store.
        
        Args:
            vector_store_id: The ID of the vector store. Defaults to the one in ENV.
            
        Returns:
            JSON object containing file details within the store.
        """
        vs_id = vector_store_id or VECTOR_STORE_ID
        if not vs_id:
            return {"error": "No vector store ID provided and VECTOR_STORE_ID not set in environment"}
            
        try:
            files_response = openai_client.beta.vector_stores.files.list(vector_store_id=vs_id, limit=100)
            files = []
            for f in files_response.data:
                files.append({
                    "id": f.id,
                    "status": f.status,
                    "created_at": f.created_at,
                    "vector_store_id": vs_id
                })
            return {"files": files, "vector_store_id": vs_id, "count": len(files)}
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def get_app_data() -> Dict[str, Any]:
        """
        Return configuration and status data about this MCP Application.
        
        Returns:
            JSON object with environment info, active store ID, and server status.
        """
        return {
            "app_name": "MCP Vector Search Server",
            "active_vector_store_id": VECTOR_STORE_ID,
            "transport": "SSE",
            "openai_status": "Initialized" if openai_client else "Missing API Key",
            "available_tools": [
                "search", "fetch", "upload_file", "delete_file", 
                "list_agents", "list_vector_stores", "list_store_files", "get_app_data"
            ]
        }

    @mcp.tool()
    async def upload_file(filename: str, content: str) -> Dict[str, Any]:
        """
        Upload a new text file to the vector store.

        This tool creates a new file in OpenAI with the provided content and
        adds it to the configured vector store for immediate searchability.

        Args:
            filename: Name of the file to create (e.g., 'notes.txt').
            content: The text content of the file.

        Returns:
            Information about the created file and its status.
        """
        if not filename or not content:
            raise ValueError("Filename and content are required")

        if not openai_client:
            raise ValueError("OpenAI API key is required")

        logger.info(f"Uploading new file: {filename}")

        # Create a temporary file to upload
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Upload the file to OpenAI
            with open(tmp_path, 'rb') as f:
                file_batch = openai_client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=VECTOR_STORE_ID,
                    files=[f]
                )

            logger.info(f"File batch status: {file_batch.status}")

            # Get the file ID from the batch (assuming one file)
            # Note: We might need to list files in the vector store to get the exact ID 
            # if upload_and_poll doesn't return individual file IDs directly in a simple way.
            # For simplicity, we return the batch status.
            
            return {
                "status": file_batch.status,
                "file_counts": file_batch.file_counts.__dict__,
                "vector_store_id": VECTOR_STORE_ID
            }

        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @mcp.tool()
    async def delete_file(file_id: str) -> Dict[str, str]:
        """
        Delete a file from the vector store and OpenAI storage.

        Args:
            file_id: The unique ID of the file to delete (e.g., 'file-xxx').

        Returns:
            A confirmation message.
        """
        if not file_id:
            raise ValueError("File ID is required")

        if not openai_client:
            raise ValueError("OpenAI API key is required")

        logger.info(f"Deleting file {file_id} from vector store {VECTOR_STORE_ID}")

        # Remove from vector store
        openai_client.vector_stores.files.delete(
            vector_store_id=VECTOR_STORE_ID,
            file_id=file_id
        )

        # Delete from OpenAI general storage
        openai_client.files.delete(file_id)

        return {"message": f"Successfully deleted file {file_id}"}

    return mcp


def main():
    """Main function to start the MCP server."""
    # Verify OpenAI client is initialized
    if not openai_client:
        logger.error(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        )
        raise ValueError("OpenAI API key is required")

    logger.info(f"Using vector store: {VECTOR_STORE_ID}")

    # Create the MCP server
    server = create_server()

    # Configure and start the server
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting MCP server on 0.0.0.0:{port}")
    logger.info("Server will be accessible via SSE transport")

    try:
        # Use FastMCP's built-in run method with SSE transport
        server.run(transport="sse", host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
