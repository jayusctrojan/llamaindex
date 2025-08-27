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
            
            return {
                "success": True,
                "document_id": f"doc_{hash(content[:100])}_{int(time.time())}",
                "title": title,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "keywords_extracted": list(all_keywords),
                "processing_time": processing_time,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)