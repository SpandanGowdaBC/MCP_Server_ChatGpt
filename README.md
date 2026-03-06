# MCP Vector Search Server

A Model Context Protocol (MCP) server that provides semantic search and document retrieval capabilities for ChatGPT, powered by OpenAI's Vector Store API.

## Features

- **`search` tool**: Perform semantic searches against an OpenAI Vector Store.
- **`fetch` tool**: Retrieve full document content and metadata for analysis and citation.
- **Remote-ready**: Built with Python and FastMCP, optimized for deployment on Render.

## Deployment to Render

This repository is "Blueprint Ready" for Render.

1. **Create a Private Repository**: Push this code to a new, private GitHub repository.
2. **Deploy via Blueprint**:
   - Go to the [Render Dashboard](https://dashboard.render.com).
   - Click **New +** > **Blueprint**.
   - Connect your GitHub repository.
   - Render will use the `render.yaml` file to set up the service.
3. **Configure Environment Variables**:
   In the Render dashboard for your service, navigate to **Environment** and add:
   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `VECTOR_STORE_ID`: The ID of your OpenAI Vector Store.

## Connecting to ChatGPT

1. Open ChatGPT and go to **Settings** > **Personalize** > **Model Context Protocol**.
2. Click **Add New Server** > **Remote (SSE)**.
3. **Name**: `Vector Store Search`
4. **URL**: `https://your-service-name.onrender.com/sse` (replace with your actual Render URL).
5. **Save** and start using your vector store tools!

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set local environment variables
export OPENAI_API_KEY="your-key"
export VECTOR_STORE_ID="your-id"

# Run the server
python mcp_server.py
```
