"""
API Integration Endpoints for n8n Workflow
Provides REST endpoints that n8n can call for hybrid RAG features
"""

from fastapi import APIRouter, HTTPException, Body, BackgroundTasks
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import logging
import os

# Import the hybrid features
from hybrid_features import (
    HybridRecordManager,
    EnhancedSessionManager,
    ContextualEmbeddingGenerator,
    MetadataEnrichmentService,
    HybridSearchOrchestrator,
    RecordManagerEntry,
    EnhancedSessionContext
)

logger = logging.getLogger(__name__)

# Create router for hybrid endpoints
hybrid_router = APIRouter(prefix="/hybrid", tags=["hybrid"])

# =====================================================
# REQUEST/RESPONSE MODELS
# =====================================================

class CheckRecordRequest(BaseModel):
    """Request to check document record"""
    doc_id: str
    content_hash: str
    document_title: Optional[str] = None
    data_type: str = "unstructured"

class CheckRecordResponse(BaseModel):
    """Response from record check"""
    exists: bool
    action: str  # "skip", "new", or "update"
    existing_record: Optional[Dict] = None
    message: str

class CreateSessionRequest(BaseModel):
    """Request to create new processing session"""
    source_file: str
    source_type: str
    processing_paths: List[str]
    course_name: str = "General"
    module_number: str = "Unknown"

class UpdateSessionPathRequest(BaseModel):
    """Request to update session path status"""
    session_id: str
    path: str
    status: str
    data: Optional[Dict] = None

class ValidateMergeRequest(BaseModel):
    """Request to validate session merge readiness"""
    session_id: str
    required_paths: List[str]

class ContextualChunkRequest(BaseModel):
    """Request to create contextual chunks"""
    chunks: List[str]
    document: str
    metadata: Optional[Dict] = None
    enable_contextual: bool = True

class EnrichMetadataRequest(BaseModel):
    """Request to enrich document metadata"""
    document_content: str
    document_name: str
    doc_id: str

class HybridSearchRequest(BaseModel):
    """Request for hybrid search"""
    query: str
    search_type: str = "hybrid"  # "vector", "graph", or "hybrid"
    metadata_filter: Optional[Dict] = None
    top_k: int = 10

# =====================================================
# INITIALIZATION (These would be initialized in main.py)
# =====================================================

# These will be initialized when the service starts
record_manager: Optional[HybridRecordManager] = None
session_manager: Optional[EnhancedSessionManager] = None
metadata_service: Optional[MetadataEnrichmentService] = None
contextual_generator: Optional[ContextualEmbeddingGenerator] = None
search_orchestrator: Optional[HybridSearchOrchestrator] = None

# =====================================================
# RECORD MANAGEMENT ENDPOINTS
# =====================================================

@hybrid_router.post("/record/check", response_model=CheckRecordResponse)
async def check_record(request: CheckRecordRequest):
    """
    Check if a document exists and compare hash for change detection.
    This endpoint is called by n8n before processing any document.
    """
    try:
        if not record_manager:
            raise HTTPException(status_code=503, detail="Record manager not initialized")
        
        existing_record, action = await record_manager.check_document(
            request.doc_id,
            request.content_hash
        )
        
        return CheckRecordResponse(
            exists=bool(existing_record),
            action=action,
            existing_record=existing_record.dict() if existing_record else None,
            message=f"Document {action}: {request.doc_id}"
        )
        
    except Exception as e:
        logger.error(f"Record check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hybrid_router.post("/record/create")
async def create_record(request: CheckRecordRequest):
    """Create new document record"""
    try:
        if not record_manager:
            raise HTTPException(status_code=503, detail="Record manager not initialized")
        
        record = await record_manager.create_record(
            doc_id=request.doc_id,
            content_hash=request.content_hash,
            document_title=request.document_title,
            data_type=request.data_type
        )
        
        return {
            "status": "success",
            "record": record.dict(),
            "message": f"Created record for {request.doc_id}"
        }
        
    except Exception as e:
        logger.error(f"Record creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hybrid_router.post("/record/update")
async def update_record(
    doc_id: str = Body(...),
    content_hash: str = Body(...),
    graph_id: Optional[str] = Body(None)
):
    """Update existing document record"""
    try:
        if not record_manager:
            raise HTTPException(status_code=503, detail="Record manager not initialized")
        
        record = await record_manager.update_record(
            doc_id=doc_id,
            content_hash=content_hash,
            graph_id=graph_id
        )
        
        return {
            "status": "success",
            "record": record.dict(),
            "message": f"Updated record for {doc_id}"
        }
        
    except Exception as e:
        logger.error(f"Record update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# SESSION MANAGEMENT ENDPOINTS
# =====================================================

@hybrid_router.post("/session/create")
async def create_session(request: CreateSessionRequest):
    """
    Create new processing session for parallel processing coordination.
    Called at the start of document processing.
    """
    try:
        if not session_manager:
            raise HTTPException(status_code=503, detail="Session manager not initialized")
        
        # Generate session ID
        session_id = f"proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.source_file[:10]}"
        
        session = await session_manager.create_session(
            session_id=session_id,
            source_file=request.source_file,
            processing_paths=request.processing_paths
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "session": session.dict(),
            "message": f"Created session for {request.source_file}"
        }
        
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hybrid_router.post("/session/update")
async def update_session_path(request: UpdateSessionPathRequest):
    """
    Update processing path status within a session.
    Called when each processing path completes.
    """
    try:
        if not session_manager:
            raise HTTPException(status_code=503, detail="Session manager not initialized")
        
        await session_manager.update_path_status(
            session_id=request.session_id,
            path=request.path,
            status=request.status,
            data=request.data
        )
        
        return {
            "status": "success",
            "session_id": request.session_id,
            "path": request.path,
            "path_status": request.status,
            "message": f"Updated path {request.path} to {request.status}"
        }
        
    except Exception as e:
        logger.error(f"Session update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hybrid_router.post("/session/validate")
async def validate_session_merge(request: ValidateMergeRequest):
    """
    Validate that all required paths are complete before merging.
    Called before content merge node in n8n.
    """
    try:
        if not session_manager:
            raise HTTPException(status_code=503, detail="Session manager not initialized")
        
        is_valid, correlation_data = await session_manager.validate_merge(
            session_id=request.session_id,
            required_paths=request.required_paths
        )
        
        return {
            "status": "success" if is_valid else "incomplete",
            "valid": is_valid,
            "correlation_data": correlation_data,
            "session_id": request.session_id,
            "message": "Ready to merge" if is_valid else "Paths incomplete"
        }
        
    except Exception as e:
        logger.error(f"Merge validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hybrid_router.get("/session/{session_id}")
async def get_session_status(session_id: str):
    """Get current session status"""
    try:
        if not session_manager:
            raise HTTPException(status_code=503, detail="Session manager not initialized")
        
        # Get from local cache first
        session = session_manager.local_cache.get(session_id)
        
        if not session:
            # Try database
            # Implementation would query the database
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "status": "success",
            "session": session.dict(),
            "active": session.status == "active"
        }
        
    except Exception as e:
        logger.error(f"Get session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# CONTEXTUAL EMBEDDING ENDPOINTS
# =====================================================

@hybrid_router.post("/embeddings/contextual")
async def create_contextual_chunks(request: ContextualChunkRequest):
    """
    Generate contextual chunks for improved embeddings.
    Called during chunking phase in document processing.
    """
    try:
        if not contextual_generator:
            raise HTTPException(status_code=503, detail="Contextual generator not initialized")
        
        if not request.enable_contextual:
            # Return original chunks without context
            return {
                "status": "success",
                "chunks": [{"text": c, "metadata": request.metadata} for c in request.chunks],
                "contextual": False
            }
        
        # Generate contextual chunks
        contextual_chunks = await contextual_generator.create_contextual_chunks(
            chunks=request.chunks,
            document=request.document,
            metadata=request.metadata
        )
        
        return {
            "status": "success",
            "chunks": contextual_chunks,
            "contextual": True,
            "chunk_count": len(contextual_chunks)
        }
        
    except Exception as e:
        logger.error(f"Contextual chunk generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# METADATA ENRICHMENT ENDPOINTS
# =====================================================

@hybrid_router.post("/metadata/enrich")
async def enrich_document_metadata(request: EnrichMetadataRequest):
    """
    Enrich document with AI-generated metadata.
    Called after content extraction but before vector storage.
    """
    try:
        if not metadata_service:
            raise HTTPException(status_code=503, detail="Metadata service not initialized")
        
        # This would use your existing LLM client
        # For now, returning placeholder
        metadata = await metadata_service.enrich_document(
            document_content=request.document_content,
            document_name=request.document_name,
            llm_client=None  # Would pass actual LLM client
        )
        
        return {
            "status": "success",
            "doc_id": request.doc_id,
            "metadata": metadata,
            "enriched": True
        }
        
    except Exception as e:
        logger.error(f"Metadata enrichment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hybrid_router.get("/metadata/fields")
async def get_metadata_fields():
    """Get configured metadata fields for filtering"""
    try:
        if not metadata_service:
            raise HTTPException(status_code=503, detail="Metadata service not initialized")
        
        fields = await metadata_service.get_metadata_fields()
        
        return {
            "status": "success",
            "fields": [f.dict() for f in fields],
            "count": len(fields)
        }
        
    except Exception as e:
        logger.error(f"Get metadata fields failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# HYBRID SEARCH ENDPOINTS
# =====================================================

@hybrid_router.post("/search")
async def hybrid_search(request: HybridSearchRequest):
    """
    Perform hybrid search across vector store and knowledge graph.
    Can be called by n8n or other services for retrieval.
    """
    try:
        if not search_orchestrator:
            raise HTTPException(status_code=503, detail="Search orchestrator not initialized")
        
        results = await search_orchestrator.hybrid_search(
            query=request.query,
            search_type=request.search_type,
            metadata_filter=request.metadata_filter,
            top_k=request.top_k
        )
        
        return {
            "status": "success",
            "query": request.query,
            "search_type": request.search_type,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# UTILITY ENDPOINTS
# =====================================================

@hybrid_router.get("/status")
async def get_hybrid_status():
    """Get status of hybrid components"""
    return {
        "status": "operational",
        "components": {
            "record_manager": bool(record_manager),
            "session_manager": bool(session_manager),
            "metadata_service": bool(metadata_service),
            "contextual_generator": bool(contextual_generator),
            "search_orchestrator": bool(search_orchestrator)
        },
        "features": {
            "hash_change_detection": True,
            "parallel_processing": True,
            "contextual_embeddings": True,
            "metadata_enrichment": True,
            "hybrid_search": True
        },
        "version": "2.6.0"
    }

@hybrid_router.post("/initialize")
async def initialize_hybrid_components(background_tasks: BackgroundTasks):
    """
    Initialize all hybrid components.
    This would be called once during service startup.
    """
    try:
        # This initialization would be done in main.py
        # Here for demonstration
        
        return {
            "status": "success",
            "message": "Hybrid components initialization started",
            "components": [
                "record_manager",
                "session_manager",
                "metadata_service",
                "contextual_generator",
                "search_orchestrator"
            ]
        }
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# DATABASE SETUP ENDPOINT
# =====================================================

@hybrid_router.get("/database/schema")
async def get_database_schema():
    """Get SQL schema for manual database setup"""
    from hybrid_features import initialize_database
    
    schema = await initialize_database()
    
    return {
        "status": "success",
        "schema": schema,
        "instructions": "Execute this SQL in your Supabase database to create the required tables"
    }
