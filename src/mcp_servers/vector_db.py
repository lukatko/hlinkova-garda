"""
MCP server for vector database queries using FAISS and sentence transformers.
Assumes the FAISS index and chunks are already prepared.
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP
from src.util.utils import get_root_dir
import os
from typing import List, Dict, Any

# Initialize MCP server
mcp = FastMCP("Vector_Database")

# Configuration
FAISS_INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# Global variables for lazy loading
_index = None
_chunks = None
_model = None

def _get_index():
    """Lazy load FAISS index"""
    global _index
    if _index is None:
        index_path = get_root_dir() / FAISS_INDEX_FILE
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        _index = faiss.read_index(str(index_path))
    return _index

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

def _get_model():
    """Lazy load sentence transformer model"""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

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
        index = _get_index()
        chunks = _get_chunks()
        model = _get_model()
        
        # Encode query
        query_embedding = model.encode([query]).astype("float32")
        
        # Search in FAISS index
        distances, indices = index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):  # Ensure index is valid
                chunk = chunks[idx]
                results.append({
                    "document": chunk.get("doc", "Unknown"),
                    "page": chunk.get("page", "Unknown"),
                    "text": chunk.get("text", ""),
                    "similarity_score": float(distances[0][i]),
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
                "model_used": MODEL_NAME
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
            "faiss_index_file": FAISS_INDEX_FILE,
            "chunks_file": CHUNKS_FILE,
            "model_name": MODEL_NAME
        }
        
        # Check if files exist
        index_path = get_root_dir() / FAISS_INDEX_FILE
        chunks_path = get_root_dir() / CHUNKS_FILE
        
        status["index_exists"] = index_path.exists()
        status["chunks_exists"] = chunks_path.exists()
        
        if status["index_exists"] and status["chunks_exists"]:
            # Get additional stats
            chunks = _get_chunks()
            index = _get_index()
            
            status["total_chunks"] = len(chunks)
            status["index_dimension"] = index.d if hasattr(index, 'd') else "Unknown"
            status["index_total_vectors"] = index.ntotal if hasattr(index, 'ntotal') else "Unknown"
            
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
