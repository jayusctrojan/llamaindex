"""
Hybrid RAG Features for AI Empire File Processing System
Adds record management, enhanced session correlation, and contextual embeddings
Version: 2.6.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import asyncpg
import aiohttp
from pydantic import BaseModel
import os

logger = logging.getLogger(__name__)

# =====================================================
# CONFIGURATION
# =====================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Airtable configuration (keeping existing)
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_BASE_URL = "https://api.airtable.com/v0"

# =====================================================
# DATA MODELS
# =====================================================

class RecordManagerEntry(BaseModel):
    """Record in the SQL record manager for change detection"""
    id: Optional[int] = None
    doc_id: str
    hash: str
    graph_id: Optional[str] = None
    data_type: str = "unstructured"  # unstructured or tabular
    schema: Optional[Dict] = None
    document_title: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    airtable_record_id: Optional[str] = None

class EnhancedSessionContext(BaseModel):
    """Enhanced session context with SQL backing"""
    session_id: str
    source_file: str
    processing_paths: Dict[str, str]  # path -> status
    correlation_data: Dict[str, Any]
    status: str = "active"
    created_at: datetime = datetime.now()
    completed_at: Optional[datetime] = None

class MetadataField(BaseModel):
    """Metadata field configuration for dynamic filtering"""
    metadata_name: str
    allowed_values: str
    data_type: str
    is_required: bool = False

# =====================================================
# RECORD MANAGER (SQL-based for performance)
# =====================================================

class HybridRecordManager:
    """
    Manages document records using SQL for performance-critical operations
    while syncing with Airtable for human-readable audit trails
    """
    
    def __init__(self, supabase_url: str, supabase_key: str, 
                 airtable_api_key: str, airtable_base_id: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.airtable_api_key = airtable_api_key
        self.airtable_base_id = airtable_base_id
        self.db_pool = None
        
    async def init_db(self):
        """Initialize database connection pool"""
        # Parse Supabase URL for PostgreSQL connection
        db_url = self.supabase_url.replace("https://", "postgresql://")
        self.db_pool = await asyncpg.create_pool(
            db_url,
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("✅ Record Manager database pool initialized")
    
    async def check_document(self, doc_id: str, content_hash: str) -> Tuple[Optional[RecordManagerEntry], str]:
        """
        Check if document exists and compare hash
        Returns: (existing_record, action) where action is 'skip', 'new', or 'update'
        """
        async with self.db_pool.acquire() as conn:
            # Check existing record
            row = await conn.fetchrow(
                "SELECT * FROM record_manager_v2 WHERE doc_id = $1",
                doc_id
            )
            
            if not row:
                return None, "new"
            
            existing = RecordManagerEntry(**dict(row))
            
            if existing.hash == content_hash:
                return existing, "skip"
            else:
                return existing, "update"
    
    async def create_record(self, doc_id: str, content_hash: str, 
                          document_title: str = None, data_type: str = "unstructured",
                          schema: Dict = None) -> RecordManagerEntry:
        """Create new record in both SQL and Airtable"""
        
        # Insert into SQL
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO record_manager_v2 
                (doc_id, hash, document_title, data_type, schema, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                RETURNING *
                """,
                doc_id, content_hash, document_title, data_type, 
                json.dumps(schema) if schema else None
            )
            
        record = RecordManagerEntry(**dict(row))
        
        # Sync to Airtable for audit
        airtable_record = await self._sync_to_airtable(record, "created")
        
        # Update SQL with Airtable ID
        if airtable_record:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE record_manager_v2 SET airtable_record_id = $1 WHERE id = $2",
                    airtable_record['id'], record.id
                )
                record.airtable_record_id = airtable_record['id']
        
        return record
    
    async def update_record(self, doc_id: str, content_hash: str, 
                          graph_id: str = None) -> RecordManagerEntry:
        """Update existing record with new hash"""
        
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE record_manager_v2 
                SET hash = $1, graph_id = $2, updated_at = NOW()
                WHERE doc_id = $3
                RETURNING *
                """,
                content_hash, graph_id, doc_id
            )
            
        record = RecordManagerEntry(**dict(row))
        
        # Sync to Airtable
        await self._sync_to_airtable(record, "updated")
        
        return record
    
    async def _sync_to_airtable(self, record: RecordManagerEntry, action: str) -> Optional[Dict]:
        """Sync record to Airtable for human-readable audit trail"""
        
        if not self.airtable_api_key or not self.airtable_base_id:
            logger.warning("Airtable not configured, skipping sync")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.airtable_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "fields": {
                "doc_id": record.doc_id,
                "hash": record.hash,
                "action": action,
                "document_title": record.document_title or "",
                "data_type": record.data_type,
                "timestamp": datetime.now().isoformat(),
                "sql_record_id": record.id
            }
        }
        
        url = f"{AIRTABLE_BASE_URL}/{self.airtable_base_id}/record_manager_audit"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Synced to Airtable: {action} for {record.doc_id}")
                    return result
                else:
                    logger.error(f"Airtable sync failed: {response.status}")
                    return None

# =====================================================
# ENHANCED SESSION CORRELATION MANAGER
# =====================================================

class EnhancedSessionManager:
    """
    Enhanced session manager with SQL backing for reliability
    and complex parallel processing validation
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.local_cache: Dict[str, EnhancedSessionContext] = {}
        self.lock = asyncio.Lock()
    
    async def create_session(self, session_id: str, source_file: str, 
                           processing_paths: List[str]) -> EnhancedSessionContext:
        """Create new processing session with SQL persistence"""
        
        paths_status = {path: "pending" for path in processing_paths}
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO session_correlation 
                (session_id, source_file, processing_paths, status, created_at)
                VALUES ($1, $2, $3, $4, NOW())
                """,
                session_id, source_file, json.dumps(paths_status), "active"
            )
        
        session = EnhancedSessionContext(
            session_id=session_id,
            source_file=source_file,
            processing_paths=paths_status,
            correlation_data={},
            status="active"
        )
        
        # Cache locally for performance
        async with self.lock:
            self.local_cache[session_id] = session
        
        logger.info(f"Created enhanced session: {session_id}")
        return session
    
    async def update_path_status(self, session_id: str, path: str, 
                                status: str, data: Any = None):
        """Update processing path status with validation"""
        
        async with self.db_pool.acquire() as conn:
            # Get current state
            row = await conn.fetchrow(
                "SELECT * FROM session_correlation WHERE session_id = $1",
                session_id
            )
            
            if not row:
                raise ValueError(f"Session {session_id} not found")
            
            # Update paths
            paths = json.loads(row['processing_paths'])
            paths[path] = status
            
            # Update correlation data
            correlation_data = json.loads(row['correlation_data']) if row['correlation_data'] else {}
            if data:
                correlation_data[path] = data
            
            # Check if all paths complete
            all_complete = all(s in ['completed', 'failed'] for s in paths.values())
            new_status = 'completed' if all_complete else 'active'
            
            # Update database
            await conn.execute(
                """
                UPDATE session_correlation 
                SET processing_paths = $1,
                    correlation_data = $2,
                    status = $3,
                    completed_at = $4
                WHERE session_id = $5
                """,
                json.dumps(paths),
                json.dumps(correlation_data),
                new_status,
                datetime.now() if all_complete else None,
                session_id
            )
            
        # Update local cache
        async with self.lock:
            if session_id in self.local_cache:
                self.local_cache[session_id].processing_paths[path] = status
                self.local_cache[session_id].correlation_data[path] = data
                self.local_cache[session_id].status = new_status
        
        logger.info(f"Updated session {session_id}, path {path}: {status}")
    
    async def validate_merge(self, session_id: str, required_paths: List[str]) -> Tuple[bool, Dict]:
        """
        Validate all required paths are complete before merging
        Returns: (is_valid, correlation_data)
        """
        
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM session_correlation WHERE session_id = $1",
                session_id
            )
            
            if not row:
                return False, {"error": "Session not found"}
            
            paths = json.loads(row['processing_paths'])
            
            # Check all required paths are completed
            for path in required_paths:
                if paths.get(path) != 'completed':
                    return False, {
                        "error": f"Path {path} not completed",
                        "paths": paths
                    }
            
            correlation_data = json.loads(row['correlation_data']) if row['correlation_data'] else {}
            
            return True, correlation_data

# =====================================================
# CONTEXTUAL EMBEDDINGS GENERATOR
# =====================================================

class ContextualEmbeddingGenerator:
    """
    Generates context-aware embeddings by adding document context
    to chunks before embedding - significantly improves retrieval
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def generate_chunk_context(self, chunk: str, document: str, 
                                    chunk_index: int, total_chunks: int) -> str:
        """Generate contextual description for a chunk"""
        
        # Limit document preview for context generation
        doc_preview = document[:2000] if len(document) > 2000 else document
        
        prompt = f"""
        <document>
        {doc_preview}
        </document>
        
        <chunk>
        {chunk}
        </chunk>
        
        This is chunk {chunk_index + 1} of {total_chunks} from the document.
        
        Please provide a brief contextual description (2-3 sentences) that situates 
        this chunk within the overall document. Focus on:
        1. What topic or section this chunk belongs to
        2. How it relates to the main document theme
        3. Key concepts or entities mentioned
        
        Be concise and factual. Response should be under 100 words.
        """
        
        try:
            response = await self.llm_client.generate(prompt, max_tokens=150)
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate chunk context: {e}")
            return f"Chunk {chunk_index + 1} of {total_chunks}"
    
    async def create_contextual_chunks(self, chunks: List[str], document: str,
                                      metadata: Dict = None) -> List[Dict]:
        """Create contextual chunks with enhanced embeddings"""
        
        contextual_chunks = []
        
        # Process chunks in parallel for efficiency
        tasks = []
        for i, chunk in enumerate(chunks):
            task = self.generate_chunk_context(chunk, document, i, len(chunks))
            tasks.append(task)
        
        contexts = await asyncio.gather(*tasks)
        
        for i, (chunk, context) in enumerate(zip(chunks, contexts)):
            enhanced_chunk = {
                'text': f"{context}\n\n{chunk}",  # Combine context with chunk
                'original_chunk': chunk,
                'context': context,
                'chunk_index': i,
                'chunk_count': len(chunks),
                'metadata': metadata or {}
            }
            contextual_chunks.append(enhanced_chunk)
        
        logger.info(f"Generated {len(contextual_chunks)} contextual chunks")
        return contextual_chunks

# =====================================================
# METADATA ENRICHMENT SERVICE
# =====================================================

class MetadataEnrichmentService:
    """
    Enriches documents with dynamic metadata for improved filtering
    and retrieval in vector searches
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.metadata_fields_cache = None
        self.cache_expiry = None
    
    async def get_metadata_fields(self) -> List[MetadataField]:
        """Get configured metadata fields from database"""
        
        # Cache for 5 minutes
        if self.metadata_fields_cache and self.cache_expiry and datetime.now() < self.cache_expiry:
            return self.metadata_fields_cache
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM metadata_fields")
            
        fields = [MetadataField(**dict(row)) for row in rows]
        
        # Update cache
        self.metadata_fields_cache = fields
        self.cache_expiry = datetime.now().replace(minute=datetime.now().minute + 5)
        
        return fields
    
    async def enrich_document(self, document_content: str, document_name: str,
                             llm_client) -> Dict[str, Any]:
        """Enrich document with AI-generated metadata"""
        
        # Get metadata fields
        fields = await self.get_metadata_fields()
        
        if not fields:
            logger.warning("No metadata fields configured")
            return {}
        
        # Build prompt for metadata extraction
        field_descriptions = "\n".join([
            f"- {field.metadata_name}: {field.allowed_values}"
            for field in fields
        ])
        
        prompt = f"""
        Analyze the following document and extract metadata according to these fields:
        
        {field_descriptions}
        
        Document Name: {document_name}
        
        Document Content (first 3000 chars):
        {document_content[:3000]}
        
        Return a JSON object with the metadata field names as keys and appropriate values.
        If a field doesn't apply, use "N/A" as the value.
        
        Example output format:
        {{
            "department": "Engineering",
            "doc_type": "Technical",
            "priority": "High"
        }}
        """
        
        try:
            response = await llm_client.generate(prompt, response_format="json")
            metadata = json.loads(response)
            
            # Add document summary
            summary_prompt = f"""
            Provide a one-sentence summary of this document:
            
            {document_content[:2000]}
            
            Summary should be under 100 words and capture the main topic.
            """
            
            summary = await llm_client.generate(summary_prompt, max_tokens=150)
            metadata['document_summary'] = summary.strip()
            metadata['enrichment_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Enriched document with metadata: {list(metadata.keys())}")
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata enrichment failed: {e}")
            return {
                'error': str(e),
                'enrichment_timestamp': datetime.now().isoformat()
            }

# =====================================================
# HYBRID SEARCH ORCHESTRATOR
# =====================================================

class HybridSearchOrchestrator:
    """
    Orchestrates hybrid search across vector store and knowledge graph
    with reranking for optimal results
    """
    
    def __init__(self, vector_store, graph_client=None):
        self.vector_store = vector_store
        self.graph_client = graph_client  # Optional - for future LightRAG integration
        
    async def hybrid_search(self, query: str, search_type: str = "hybrid",
                           metadata_filter: Dict = None, top_k: int = 10) -> List[Dict]:
        """
        Perform hybrid search with optional metadata filtering
        
        Args:
            query: Search query
            search_type: "vector", "graph", or "hybrid"
            metadata_filter: Metadata constraints for filtering
            top_k: Number of results to return
        """
        
        results = []
        
        # Vector search
        if search_type in ["vector", "hybrid"]:
            vector_results = await self._vector_search(query, metadata_filter, top_k * 2)
            results.extend(vector_results)
        
        # Graph search (if available)
        if search_type in ["graph", "hybrid"] and self.graph_client:
            graph_results = await self._graph_search(query, top_k * 2)
            results.extend(graph_results)
        
        # Rerank combined results
        if len(results) > 0:
            results = await self._rerank_results(query, results, top_k)
        
        return results
    
    async def _vector_search(self, query: str, metadata_filter: Dict, 
                            limit: int) -> List[Dict]:
        """Perform vector similarity search"""
        
        # This would integrate with your existing Pinecone setup
        # Adding metadata filtering capability
        
        search_params = {
            "query": query,
            "top_k": limit,
            "include_metadata": True
        }
        
        if metadata_filter:
            search_params["filter"] = metadata_filter
        
        # Placeholder for actual vector search
        # In practice, this would call your Pinecone index
        results = []
        
        return results
    
    async def _graph_search(self, query: str, limit: int) -> List[Dict]:
        """Perform knowledge graph search"""
        
        # Placeholder for LightRAG integration
        # This would query entity relationships and return relevant contexts
        
        results = []
        
        return results
    
    async def _rerank_results(self, query: str, results: List[Dict], 
                             top_k: int) -> List[Dict]:
        """Rerank results using cross-encoder or Cohere reranker"""
        
        # For now, simple score-based sorting
        # In production, integrate with Cohere or similar reranking service
        
        # Deduplicate by content hash
        seen = set()
        unique_results = []
        for result in results:
            content_hash = hashlib.md5(str(result.get('content', '')).encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                unique_results.append(result)
        
        # Sort by relevance score
        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return unique_results[:top_k]

# =====================================================
# DATABASE INITIALIZATION
# =====================================================

async def initialize_database():
    """Initialize SQL tables for hybrid approach"""
    
    # This would be called once during setup
    # Tables are created via SQL migrations
    
    sql_schema = """
    -- Record Manager table
    CREATE TABLE IF NOT EXISTS record_manager_v2 (
        id SERIAL PRIMARY KEY,
        doc_id TEXT UNIQUE NOT NULL,
        hash TEXT NOT NULL,
        graph_id TEXT,
        data_type TEXT CHECK (data_type IN ('unstructured', 'tabular')),
        schema JSONB,
        document_title TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        airtable_record_id TEXT
    );
    
    -- Session Correlation table
    CREATE TABLE IF NOT EXISTS session_correlation (
        session_id TEXT PRIMARY KEY,
        source_file TEXT NOT NULL,
        processing_paths JSONB NOT NULL,
        correlation_data JSONB,
        status TEXT DEFAULT 'active',
        created_at TIMESTAMP DEFAULT NOW(),
        completed_at TIMESTAMP
    );
    
    -- Metadata fields configuration
    CREATE TABLE IF NOT EXISTS metadata_fields (
        id SERIAL PRIMARY KEY,
        metadata_name TEXT UNIQUE NOT NULL,
        allowed_values TEXT,
        data_type TEXT,
        is_required BOOLEAN DEFAULT FALSE
    );
    
    -- Vector metadata for fast lookups
    CREATE TABLE IF NOT EXISTS vector_metadata (
        id SERIAL PRIMARY KEY,
        doc_id TEXT NOT NULL,
        vector_id TEXT NOT NULL,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_record_manager_doc_id ON record_manager_v2(doc_id);
    CREATE INDEX IF NOT EXISTS idx_session_correlation_session_id ON session_correlation(session_id);
    CREATE INDEX IF NOT EXISTS idx_vector_metadata_doc_id ON vector_metadata(doc_id);
    CREATE INDEX IF NOT EXISTS idx_vector_metadata_gin ON vector_metadata USING GIN(metadata);
    """
    
    logger.info("Database schema ready for hybrid approach")
    return sql_schema

# =====================================================
# EXPORT KEY COMPONENTS
# =====================================================

__all__ = [
    'HybridRecordManager',
    'EnhancedSessionManager',
    'ContextualEmbeddingGenerator',
    'MetadataEnrichmentService',
    'HybridSearchOrchestrator',
    'RecordManagerEntry',
    'EnhancedSessionContext',
    'MetadataField',
    'initialize_database'
]
