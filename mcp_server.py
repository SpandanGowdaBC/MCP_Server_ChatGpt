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
This MCP server provides comprehensive management and search capabilities for OpenAI resources.
- Management: List, get, and update Assistants (Agents), Vector Stores, Files, and Fine-tuning jobs.
- Search: Use the search tool to find relevant documents in Vector Stores.
- Retrieval: Use the fetch tool to retrieve complete document content.
- Ingestion: Use upload_file and delete_file to manage your content.
- Organization: List available models and projects.
"""


def create_server():
    """Create and configure the MCP server with search and fetch tools."""

    # Initialize the FastMCP server
    mcp = FastMCP(name="Sample MCP Server",
                  instructions=server_instructions)

    @mcp.tool()
    async def search(query: str, vector_store_id: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for documents using OpenAI Vector Store search.

        This tool searches through the vector store to find semantically relevant matches.
        Returns a list of search results with basic information. Use the fetch tool to get
        complete document content.

        Args:
            query: Search query string. Natural language queries work best for semantic search.
            vector_store_id: Optional ID of the vector store to search. Defaults to the one in ENV.
        """
        if not query or not query.strip():
            return {"results": []}

        if not openai_client:
            logger.error("OpenAI client not initialized - API key missing")
            raise ValueError(
                "OpenAI API key is required for vector store search")

        vs_id = vector_store_id or VECTOR_STORE_ID
        if not vs_id:
            return {"error": "No vector store ID provided and VECTOR_STORE_ID not set in environment"}

        # Search the vector store using OpenAI API
        logger.info(f"Searching {vs_id} for query: '{query}'")

        response = openai_client.vector_stores.search(
            vector_store_id=vs_id, query=query)

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
    async def fetch(id: str, vector_store_id: str = None) -> Dict[str, Any]:
        """
        Retrieve complete document content by ID for detailed
        analysis and citation. This tool fetches the full document
        content from OpenAI Vector Store. 

        Args:
            id: File ID from vector store (file-xxx).
            vector_store_id: Optional ID of the vector store. Defaults to the one in ENV.
        """
        if not id:
            raise ValueError("Document ID is required")

        if not openai_client:
            logger.error("OpenAI client not initialized - API key missing")
            raise ValueError(
                "OpenAI API key is required for vector store file retrieval")

        vs_id = vector_store_id or VECTOR_STORE_ID
        if not vs_id:
            return {"error": "No vector store ID provided and VECTOR_STORE_ID not set in environment"}

        logger.info(f"Fetching content from vector store {vs_id} for file ID: {id}")

        # Fetch file content from vector store
        content_response = openai_client.vector_stores.files.content(
            vector_store_id=vs_id, file_id=id)

        # Get file metadata
        file_info = openai_client.vector_stores.files.retrieve(
            vector_store_id=vs_id, file_id=id)

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
    async def get_agent(agent_id: str) -> Dict[str, Any]:
        """
        Retrieve full details of a specific OpenAI Assistant (Agent).
        
        Args:
            agent_id: The ID of the assistant (asst-xxx).
        """
        try:
            a = openai_client.beta.assistants.retrieve(agent_id)
            return {
                "id": a.id,
                "name": a.name,
                "model": a.model,
                "instructions": a.instructions,
                "tools": [t.type if hasattr(t, 'type') else str(t) for t in a.tools],
                "created_at": a.created_at,
                "metadata": a.metadata
            }
        except Exception as e:
            logger.error(f"Error retrieving agent {agent_id}: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def update_agent(agent_id: str, 
                         name: str = None, 
                         instructions: str = None, 
                         model: str = None) -> Dict[str, Any]:
        """
        Update an existing OpenAI Assistant's configuration.
        
        Args:
            agent_id: The ID of the assistant to update.
            name: Optional new name.
            instructions: Optional new instructions.
            model: Optional new model ID (e.g., 'gpt-4o').
        """
        try:
            update_params = {}
            if name: update_params["name"] = name
            if instructions: update_params["instructions"] = instructions
            if model: update_params["model"] = model
            
            a = openai_client.beta.assistants.update(agent_id, **update_params)
            return {
                "id": a.id,
                "status": "updated",
                "name": a.name,
                "model": a.model
            }
        except Exception as e:
            logger.error(f"Error updating agent {agent_id}: {e}")
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
    async def list_files() -> Dict[str, Any]:
        """
        List all files uploaded to the OpenAI account.
        
        Returns:
            JSON object containing a list of all files and their metadata.
        """
        try:
            files_response = openai_client.files.list()
            files = []
            for f in files_response.data:
                files.append({
                    "id": f.id,
                    "filename": f.filename,
                    "purpose": f.purpose,
                    "bytes": f.bytes,
                    "created_at": f.created_at
                })
            return {"files": files, "count": len(files)}
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def get_file_content(file_id: str) -> Dict[str, Any]:
        """
        Retrieve the content of a specific file from OpenAI storage.
        
        Args:
            file_id: The ID of the file (file-xxx).
        """
        try:
            content_response = openai_client.files.content(file_id)
            # For text-based files, we decode the content
            content = content_response.text
            return {"id": file_id, "content": content}
        except Exception as e:
            logger.error(f"Error retrieving content for file {file_id}: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def list_models() -> Dict[str, Any]:
        """
        List all available OpenAI models.
        """
        try:
            models_response = openai_client.models.list()
            models = [{"id": m.id, "created": m.created, "owned_by": m.owned_by} for m in models_response.data]
            return {"models": models, "count": len(models)}
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def list_projects() -> Dict[str, Any]:
        """
        List all OpenAI Projects (requires organization-level API key).
        """
        try:
            # Note: This requires organization admin permissions
            # Using the organization API if available
            if hasattr(openai_client, 'organization'):
                projects = openai_client.organization.projects.list()
                return {"projects": [p.__dict__ for p in projects.data], "count": len(projects.data)}
            else:
                return {"error": "Project management not available with current API key scope"}
        except Exception as e:
            logger.error(f"Error listing projects: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def list_fine_tuning() -> Dict[str, Any]:
        """
        List all fine-tuning jobs in the account.
        """
        try:
            jobs = openai_client.fine_tuning.jobs.list(limit=50)
            result = []
            for j in jobs.data:
                result.append({
                    "id": j.id,
                    "model": j.model,
                    "fine_tuned_model": j.fine_tuned_model,
                    "status": j.status,
                    "created_at": j.created_at
                })
            return {"fine_tuning_jobs": result, "count": len(result)}
        except Exception as e:
            logger.error(f"Error listing fine-tuning jobs: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def get_fine_tuning_job(job_id: str) -> Dict[str, Any]:
        """
        Retrieve details of a specific fine-tuning job.
        """
        try:
            j = openai_client.fine_tuning.jobs.retrieve(job_id)
            return {
                "id": j.id,
                "model": j.model,
                "status": j.status,
                "error": j.error.__dict__ if j.error else None,
                "hyperparameters": j.hyperparameters.__dict__ if hasattr(j, 'hyperparameters') else None,
                "created_at": j.created_at
            }
        except Exception as e:
            logger.error(f"Error retrieving job {job_id}: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def search_resources(query: str) -> Dict[str, Any]:
        """
        Search across all OpenAI resources (Agents, Files, Vector Stores, Models) at once.
        
        Args:
            query: The search term to match against names, IDs, or filenames.
        """
        query = query.lower()
        results = {
            "agents": [],
            "files": [],
            "vector_stores": [],
            "models": []
        }
        
        try:
            # 1. Search Agents
            assistants = openai_client.beta.assistants.list(limit=50)
            for a in assistants.data:
                if query in (a.name or "").lower() or query in a.id.lower():
                    results["agents"].append({"id": a.id, "name": a.name})
            
            # 2. Search Files
            files = openai_client.files.list()
            for f in files.data:
                if query in (f.filename or "").lower() or query in f.id.lower():
                    results["files"].append({"id": f.id, "filename": f.filename})
            
            # 3. Search Vector Stores
            stores = openai_client.beta.vector_stores.list(limit=50)
            for s in stores.data:
                if query in (s.name or "").lower() or query in s.id.lower():
                    results["vector_stores"].append({"id": s.id, "name": s.name})
            
            # 4. Search Models
            models = openai_client.models.list()
            for m in models.data:
                if query in m.id.lower():
                    results["models"].append({"id": m.id})
                    
            return {"query": query, "results": results}
        except Exception as e:
            logger.error(f"Error in universal search: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def get_app_data() -> Dict[str, Any]:
        """
        Return configuration and status data about this MCP Application.
        
        Returns:
            JSON object with environment info, active store ID, and server status.
        """
        return {
            "app_name": "OpenAI Multi-Resource MCP Server",
            "active_vector_store_id": VECTOR_STORE_ID,
            "transport": "SSE",
            "openai_status": "Initialized" if openai_client else "Missing API Key",
            "available_tools": [
                "search", "fetch", "upload_file", "delete_file", 
                "list_agents", "get_agent", "update_agent",
                "list_vector_stores", "list_store_files", 
                "list_files", "get_file_content", "list_models", "list_projects",
                "list_fine_tuning", "get_fine_tuning_job", "search_resources",
                "get_app_data"
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
