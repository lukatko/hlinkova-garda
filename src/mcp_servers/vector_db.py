"""
MCP server for vector database queries using ChromaDB and Azure OpenAI embeddings.
Assumes the ChromaDB collection and chunks are already prepared.
"""

import json
import chromadb
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from src.util.utils import get_root_dir
import os
from typing import List, Dict, Any

load_dotenv()
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

# Initialize MCP server
mcp = FastMCP("Vector_Database")

# Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_chunks"
CHUNKS_FILE = "chunks.json"
AZURE_ENDPOINT = "https://aim-azure-ai-foundry.cognitiveservices.azure.com/openai/v1/"
DEPLOYMENT_NAME = "text-embedding-model"

# Global variables for lazy loading
_client = None
_collection = None
_chunks = None
_openai_client = None

def _get_client():
    """Lazy load ChromaDB client"""
    global _client
    if _client is None:
        chroma_path = get_root_dir() / CHROMA_DB_PATH
        if not chroma_path.exists():
            raise FileNotFoundError(f"ChromaDB not found at {chroma_path}")
        _client = chromadb.PersistentClient(path=str(chroma_path))
    return _client

def _get_collection():
    """Lazy load ChromaDB collection"""
    global _collection
    if _collection is None:
        client = _get_client()
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception as e:
            raise FileNotFoundError(f"Collection '{COLLECTION_NAME}' not found in ChromaDB: {e}")
    return _collection

def _get_chunks():
    """Lazy load chunks data"""
    global _chunks
    if _chunks is None:
        chunks_path = get_root_dir() / CHUNKS_FILE
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {chunks_path}")
        with open(chunks_path, encoding="utf-8") as f:
            _chunks = json.load(f)
    return _chunks

def _get_openai_client():
    """Lazy load Azure OpenAI client"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            base_url=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
        )
    return _openai_client

@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for relevant documents using vector similarity.
    
    Args:
        query: The search query to find relevant documents
        top_k: Number of top results to return (default: 5, max: 20)
    
    Returns:
        List of dictionaries containing document matches with metadata
    """
    # Limit top_k to prevent excessive results
    top_k = min(top_k, 20)
    
    try:
        # Get components
        collection = _get_collection()
        openai_client = _get_openai_client()
        
        # Generate embedding using Azure OpenAI
        response = openai_client.embeddings.create(
            input=query,
            model=DEPLOYMENT_NAME
        )
        query_embedding = response.data[0].embedding
        
        # Search in ChromaDB collection
        results_data = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        results = []
        if results_data['documents'] and results_data['documents'][0]:
            for i in range(len(results_data['documents'][0])):
                metadata = results_data['metadatas'][0][i] if results_data['metadatas'] else {}
                distance = results_data['distances'][0][i] if results_data['distances'] else 0.0
                
                results.append({
                    "document": metadata.get("doc", "Unknown"),
                    "page": metadata.get("page", "Unknown"),
                    "text": results_data['documents'][0][i],
                    "similarity_score": float(distance),
                    "rank": i + 1
                })
        
        return results
        
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]

@mcp.tool()
def query_with_context(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Search for relevant documents and return formatted context for LLM processing.
    
    Args:
        question: The question to find relevant context for
        top_k: Number of top results to include in context (default: 5, max: 10)
    
    Returns:
        Dictionary with formatted context and metadata
    """
    # Limit top_k for context
    top_k = min(top_k, 10)
    
    try:
        # Get search results
        results = search_documents(question, top_k)
        
        if results and "error" in results[0]:
            return {"error": results[0]["error"]}
        
        # Format context
        context_parts = []
        documents_used = set()
        
        for result in results:
            doc_info = f"(Document: {result['document']}, Page: {result['page']})"
            context_parts.append(f"{doc_info}: {result['text']}")
            documents_used.add(result['document'])
        
        return {
            "question": question,
            "context": "\n\n".join(context_parts),
            "num_results": len(results),
            "documents_referenced": list(documents_used),
            "search_metadata": {
                "top_k_used": top_k,
                "model_used": DEPLOYMENT_NAME
            }
        }
        
    except Exception as e:
        return {"error": f"Context generation failed: {str(e)}"}

@mcp.resource("vectordb://status")
def get_vector_db_status() -> Dict[str, Any]:
    """
    Get the status and metadata of the vector database.
    
    Returns:
        Dictionary with database status and statistics
    """
    try:
        status = {
            "chroma_db_path": CHROMA_DB_PATH,
            "collection_name": COLLECTION_NAME,
            "chunks_file": CHUNKS_FILE,
            "model_name": DEPLOYMENT_NAME,
            "azure_endpoint": AZURE_ENDPOINT
        }
        
        # Check if files exist
        chroma_path = get_root_dir() / CHROMA_DB_PATH
        chunks_path = get_root_dir() / CHUNKS_FILE
        
        status["chroma_db_exists"] = chroma_path.exists()
        status["chunks_exists"] = chunks_path.exists()
        
        if status["chroma_db_exists"] and status["chunks_exists"]:
            # Get additional stats
            chunks = _get_chunks()
            collection = _get_collection()
            
            status["total_chunks"] = len(chunks)
            
            # Get collection stats
            collection_count = collection.count()
            status["collection_total_vectors"] = collection_count
            
            # Get unique documents
            documents = set()
            for chunk in chunks:
                if "doc" in chunk:
                    documents.add(chunk["doc"])
            status["unique_documents"] = len(documents)
            status["document_list"] = list(documents)
            
        return status
        
    except Exception as e:
        return {"error": f"Status check failed: {str(e)}"}

@mcp.resource("vectordb://documents")
def list_available_documents() -> List[str]:
    """
    List all available documents in the vector database.
    
    Returns:
        List of document names available for search
    """
    try:
        chunks = _get_chunks()
        documents = set()
        
        for chunk in chunks:
            if "doc" in chunk and chunk["doc"]:
                documents.add(chunk["doc"])
        
        return sorted(list(documents))
        
    except Exception as e:
        return [f"Error: {str(e)}"]

if __name__ == "__main__":
    mcp.run(transport="stdio")
