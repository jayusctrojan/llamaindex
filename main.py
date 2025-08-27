from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
import logging
from datetime import datetime

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor, KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Supabase integration
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
    print("SUCCESS: Supabase imported successfully")
except Exception as e:
    print(f"WARNING: Could not import Supabase: {e}")
    SUPABASE_AVAILABLE = False

# Document readers
from llama_index.readers.file import PDFReader, DocxReader
import tempfile
import aiofiles
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LlamaIndex Document Processor", 
    version="1.0.0",
    description="Document processing service using LlamaIndex for AI knowledge extraction"
)

# Configure LlamaIndex Settings
Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY")
)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Supabase Document Processing Logger
class DocumentProcessingLogger:
    """Logs document processing activities to Supabase"""
    
    def __init__(self):
        self.available = False
        
        if not SUPABASE_AVAILABLE:
            print("WARNING: Supabase not available, processing logging disabled")
            return
            
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")
            
            if not supabase_url or not supabase_key:
                print("WARNING: Supabase credentials not found")
                return
                
            self.supabase: Client = create_client(supabase_url, supabase_key)
            self.available = True
            print("Supabase DocumentProcessingLogger initialized successfully")
        except Exception as e:
            print(f"ERROR initializing Supabase logger: {e}")
            self.available = False
    
    def log_processing_start(self, document_title: str, file_type: str, content_length: int) -> str:
        """Log the start of document processing"""
        if not self.available:
            return "logger_unavailable"
            
        try:
            session_id = f"proc_{int(time.time())}_{hash(document_title)}"
            
            data = {
                "session_id": session_id,
                "document_title": document_title,
                "file_type": file_type,
                "content_length": content_length,
                "status": "processing",
                "started_at": datetime.now().isoformat(),
                "service": "llamaindex"
            }
            
            result = self.supabase.table("document_processing_log").insert(data).execute()
            return session_id
        except Exception as e:
            print(f"Error logging processing start: {e}")
            return "logging_failed"
    
    def log_processing_complete(self, session_id: str, chunks_created: int, 
                              keywords: List[str], processing_time: float, 
                              success: bool = True, error_message: str = None):
        """Log the completion of document processing"""
        if not self.available or session_id in ["logger_unavailable", "logging_failed"]:
            return
            
        try:
            update_data = {
                "status": "completed" if success else "failed",
                "chunks_created": chunks_created,
                "keywords_extracted": keywords,
                "processing_time_seconds": processing_time,
                "completed_at": datetime.now().isoformat()
            }
            
            if error_message:
                update_data["error_message"] = error_message
            
            self.supabase.table("document_processing_log").update(update_data).eq("session_id", session_id).execute()
        except Exception as e:
            print(f"Error logging processing completion: {e}")
    
    def get_processing_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get processing statistics for the last N days"""
        if not self.available:
            return {"error": "Logger not available"}
            
        try:
            # Calculate date threshold
            from datetime import timedelta
            threshold = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Query processing logs
            result = self.supabase.table("document_processing_log").select("*").gte("started_at", threshold).execute()
            
            if not result.data:
                return {"total_documents": 0, "days": days}
            
            total_docs = len(result.data)
            successful_docs = len([r for r in result.data if r.get("status") == "completed"])
            total_chunks = sum([r.get("chunks_created", 0) for r in result.data if r.get("chunks_created")])
            avg_processing_time = sum([r.get("processing_time_seconds", 0) for r in result.data]) / total_docs if total_docs > 0 else 0
            
            return {
                "total_documents": total_docs,
                "successful_documents": successful_docs,
                "success_rate": (successful_docs / total_docs * 100) if total_docs > 0 else 0,
                "total_chunks_created": total_chunks,
                "avg_processing_time": round(avg_processing_time, 2),
                "days": days
            }
        except Exception as e:
            print(f"Error getting processing stats: {e}")
            return {"error": str(e)}

# Initialize global logger
try:
    doc_logger = DocumentProcessingLogger()
    print("Global document logger initialized")
except Exception as e:
    print(f"Failed to initialize document logger: {e}")
    doc_logger = None

# Pydantic models
class ProcessTextRequest(BaseModel):
    content: str
    document_title: str
    metadata: Optional[Dict[str, Any]] = {}

class ProcessedDocument(BaseModel):
    success: bool
    document_id: str
    title: str
    chunks_created: int
    keywords_extracted: List[str]
    processing_time: float
    metadata: Dict[str, Any]

class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    keywords: List[str]

class DocumentProcessor:
    """Core document processing class using LlamaIndex"""
    
    def __init__(self):
        # Initialize text splitter
        self.text_splitter = SentenceSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize extractors
        self.title_extractor = TitleExtractor(
            nodes=5,
            llm=Settings.llm
        )
        
        self.keyword_extractor = KeywordExtractor(
            keywords=10,
            llm=Settings.llm
        )
        
        # Create ingestion pipeline
        self.pipeline = IngestionPipeline(
            transformations=[
                self.text_splitter,
                self.title_extractor,
                self.keyword_extractor,
                Settings.embed_model
            ]
        )
        
        logger.info("DocumentProcessor initialized with LlamaIndex pipeline")
    
    def process_text(self, content: str, title: str, metadata: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Process text content through LlamaIndex pipeline"""
        # Log processing start
        session_id = doc_logger.log_processing_start(title, "text", len(content)) if doc_logger else None
        
        start_time = time.time()
        
        try:
            # Create LlamaIndex Document
            document = Document(
                text=content,
                metadata={
                    "title": title,
                    "processed_at": datetime.now().isoformat(),
                    **metadata
                }
            )
            
            # Process through pipeline
            nodes = self.pipeline.run(documents=[document])
            
            # Extract information from processed nodes
            chunks = []
            all_keywords = set()
            
            for i, node in enumerate(nodes):
                chunk_keywords = node.metadata.get("keywords", [])
                if isinstance(chunk_keywords, str):
                    chunk_keywords = [chunk_keywords]
                
                chunks.append({
                    "chunk_id": f"{title}_{i+1}",
                    "content": node.text,
                    "metadata": node.metadata,
                    "keywords": chunk_keywords,
                    "embedding": node.embedding if hasattr(node, 'embedding') else None
                })
                
                all_keywords.update(chunk_keywords)
            
            processing_time = time.time() - start_time
            
            # Log successful completion
            if doc_logger:
                doc_logger.log_processing_complete(
                    session_id, len(chunks), list(all_keywords), processing_time, success=True
                )
            
            return {
                "success": True,
                "document_id": f"doc_{hash(content[:100])}_{int(time.time())}",
                "title": title,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "keywords_extracted": list(all_keywords),
                "processing_time": processing_time,
                "metadata": metadata,
                "session_id": session_id
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Log failed completion
            if doc_logger:
                doc_logger.log_processing_complete(
                    session_id, 0, [], processing_time, success=False, error_message=str(e)
                )
            
            logger.error(f"Error processing document: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "session_id": session_id
            }
    
    def process_file(self, file_content: bytes, filename: str, file_type: str) -> Dict[str, Any]:
        """Process uploaded file content"""
        try:
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            # Read based on file type
            if file_type.lower() == 'pdf':
                reader = PDFReader()
                documents = reader.load_data(tmp_file_path)
            elif file_type.lower() in ['docx', 'doc']:
                reader = DocxReader()
                documents = reader.load_data(tmp_file_path)
            else:
                # For text files, read directly
                content = file_content.decode('utf-8')
                return self.process_text(content, filename, {"file_type": file_type})
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            # Combine all document text
            combined_text = "\n\n".join([doc.text for doc in documents])
            
            return self.process_text(
                combined_text, 
                filename, 
                {"file_type": file_type, "original_filename": filename}
            )
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing file: {str(e)}"
            }

# Initialize processor
processor = DocumentProcessor()

@app.get("/")
async def root():
    return {
        "service": "LlamaIndex Document Processor",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "process_text": "/process/text",
            "process_file": "/process/file",
            "extract_keywords": "/extract/keywords"
        },
        "features": [
            "Document chunking",
            "Keyword extraction", 
            "Title extraction",
            "Embedding generation",
            "Multi-format support (PDF, DOCX, TXT)"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "LlamaIndex Document Processor",
        "timestamp": datetime.now().isoformat(),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "llama_index_version": "0.9.0+"
    }

@app.post("/process/text")
async def process_text_content(request: ProcessTextRequest):
    """Process text content through LlamaIndex pipeline"""
    if not request.content or not request.document_title:
        raise HTTPException(status_code=400, detail="Content and document_title are required")
    
    try:
        result = processor.process_text(
            content=request.content,
            title=request.document_title,
            metadata=request.metadata or {}
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
        
        return result
        
    except Exception as e:
        logger.error(f"Error in process_text_content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/process/file")
async def process_file_upload(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None)
):
    """Process uploaded file through LlamaIndex pipeline"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Determine file type
    file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    supported_types = ['pdf', 'docx', 'doc', 'txt']
    
    if file_extension not in supported_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: {', '.join(supported_types)}"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        document_title = title or file.filename
        
        # Process file
        result = processor.process_file(file_content, document_title, file_extension)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

@app.post("/extract/keywords")
async def extract_keywords_only(request: ProcessTextRequest):
    """Extract keywords from text without full processing"""
    if not request.content:
        raise HTTPException(status_code=400, detail="Content is required")
    
    try:
        # Create document
        document = Document(text=request.content)
        
        # Extract keywords only
        keywords = processor.keyword_extractor.extract([document])
        
        return {
            "success": True,
            "document_title": request.document_title,
            "keywords": keywords[0].metadata.get("keywords", []) if keywords else [],
            "content_length": len(request.content),
            "processing_time": 0.1  # Minimal processing time for keywords only
        }
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Keyword extraction error: {str(e)}")

@app.get("/process/status/{document_id}")
async def get_processing_status(document_id: str):
    """Get status of document processing (placeholder for future async processing)"""
    return {
        "document_id": document_id,
        "status": "completed",
        "message": "Document processing completed successfully"
    }

@app.get("/stats/processing")
async def get_processing_statistics(days: int = 7):
    """Get document processing statistics from Supabase"""
    if not doc_logger or not doc_logger.available:
        return {
            "error": "Document processing logging not available",
            "supabase_configured": SUPABASE_AVAILABLE
        }
    
    try:
        stats = doc_logger.get_processing_stats(days)
        return {
            "success": True,
            "statistics": stats,
            "service": "llamaindex",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting processing statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "LlamaIndex Document Processor",
        "timestamp": datetime.now().isoformat(),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "supabase_configured": bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_ANON_KEY")),
        "supabase_logging": getattr(doc_logger, 'available', False) if doc_logger else False,
        "llama_index_version": "0.9.0+"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)