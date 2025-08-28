import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional
import json
import httpx
from datetime import datetime

# LlamaIndex imports
try:
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.readers.file import PDFReader, DocxReader
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    LLAMAINDEX_AVAILABLE = True
except ImportError as e:
    print(f"LlamaIndex not available: {e}")
    LLAMAINDEX_AVAILABLE = False

# LangExtract imports
try:
    import langextract as lx
    LANGEXTRACT_AVAILABLE = True
except ImportError as e:
    print(f"LangExtract not available: {e}")
    LANGEXTRACT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LlamaIndex + Hyperbolic.ai Document Processor", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")
HYPERBOLIC_BASE_URL = "https://api.hyperbolic.xyz/v1"

# Backblaze B2 configuration
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APPLICATION_KEY = os.getenv("B2_APPLICATION_KEY")  
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME")

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "course-knowledge")

# Pinecone imports
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = bool(PINECONE_API_KEY)
    if PINECONE_AVAILABLE:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
    else:
        index = None
except ImportError as e:
    print(f"Pinecone not available: {e}")
    PINECONE_AVAILABLE = False
    index = None

# Backblaze B2 imports
try:
    from b2sdk.v2 import InMemoryAccountInfo, B2Api
    B2_AVAILABLE = bool(B2_KEY_ID and B2_APPLICATION_KEY and B2_BUCKET_NAME)
    if B2_AVAILABLE:
        # Initialize B2 API
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", B2_KEY_ID, B2_APPLICATION_KEY)
        bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
    else:
        bucket = None
except ImportError as e:
    print(f"B2SDK not available: {e}")
    B2_AVAILABLE = False
    bucket = None

class HyperbolicLLM:
    """Custom LLM wrapper for Hyperbolic.ai"""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.api_key = api_key
        self.model = model
        self.base_url = HYPERBOLIC_BASE_URL
        
    async def complete(self, prompt: str, max_tokens: int = 2048) -> str:
        """Send completion request to Hyperbolic.ai"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"Hyperbolic API error: {e}")
                raise HTTPException(status_code=500, f"Hyperbolic API error: {e}")

class DocumentProcessor:
    """Document processing with LlamaIndex + Hyperbolic.ai + LangExtract"""
    
    def __init__(self):
        self.available = False
        
        if not LLAMAINDEX_AVAILABLE:
            logger.warning("LlamaIndex not available, document processing disabled")
            return
            
        if not HYPERBOLIC_API_KEY:
            logger.warning("Hyperbolic API key not set, using basic processing")
        else:
            self.hyperbolic_llm = HyperbolicLLM(HYPERBOLIC_API_KEY)
            
        # Initialize LlamaIndex components
        self.pdf_reader = PDFReader()
        self.docx_reader = DocxReader()
        self.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
        
        # Initialize LangExtract
        if LANGEXTRACT_AVAILABLE and HYPERBOLIC_API_KEY:
            self.lang_extractor = self._setup_langextract()
        else:
            self.lang_extractor = None
            logger.warning("LangExtract not available or Hyperbolic API key missing")
        
        # Set up OpenAI for embeddings (still need this for LlamaIndex)
        if os.getenv("OPENAI_API_KEY"):
            Settings.embed_model = OpenAIEmbedding()
        
        self.available = True
        logger.info("Document processor initialized successfully")
    
    def _setup_langextract(self):
        """Configure LangExtract to use Hyperbolic.ai as the LLM backend"""
        try:
            # Configure LangExtract to use Hyperbolic API endpoint
            extractor = lx.LLMExtractor(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                base_url=HYPERBOLIC_BASE_URL,
                api_key=HYPERBOLIC_API_KEY
            )
            return extractor
        except Exception as e:
            logger.error(f"LangExtract setup failed: {e}")
            return None
    
    async def process_document(self, file_path: str, content: bytes, filename: str) -> dict:
        """Process document with LlamaIndex, enhance with Hyperbolic.ai, and extract structured data with LangExtract"""
        if not self.available:
            raise HTTPException(status_code=503, detail="Document processor not available")
        
        try:
            # Step 1: Extract text with LlamaIndex
            if filename.endswith('.pdf'):
                # Save file temporarily for processing
                with open(f"/tmp/{filename}", "wb") as f:
                    f.write(content)
                documents = self.pdf_reader.load_data(f"/tmp/{filename}")
                os.remove(f"/tmp/{filename}")
            elif filename.endswith('.docx'):
                with open(f"/tmp/{filename}", "wb") as f:
                    f.write(content)
                documents = self.docx_reader.load_data(f"/tmp/{filename}")
                os.remove(f"/tmp/{filename}")
            else:
                # Plain text processing
                text = content.decode('utf-8')
                documents = [Document(text=text)]
            
            # Step 2: Parse into nodes/chunks
            nodes = self.node_parser.get_nodes_from_documents(documents)
            
            # Step 3: Process each chunk with LangExtract + Hyperbolic
            processed_chunks = []
            for i, node in enumerate(nodes):
                chunk_data = {
                    "text": node.text,
                    "metadata": node.metadata,
                    "chunk_id": node.node_id,
                    "source": filename,
                    "chunk_index": i
                }
                
                # Step 3a: LangExtract structured extraction with source grounding
                if self.lang_extractor:
                    try:
                        # Define extraction schema for course materials
                        extraction_schema = {
                            "key_concepts": "List of main concepts discussed",
                            "learning_objectives": "Educational goals or takeaways",
                            "technical_terms": "Industry-specific terminology used",
                            "actionable_insights": "Specific steps or recommendations",
                            "source_quotes": "Important direct quotes with exact text"
                        }
                        
                        # Extract structured data with source grounding
                        extracted_data = await self._extract_with_langextract(
                            node.text, 
                            extraction_schema,
                            filename,
                            i
                        )
                        chunk_data["structured_extraction"] = extracted_data
                        
                    except Exception as e:
                        logger.warning(f"LangExtract processing failed for chunk {i}: {e}")
                        chunk_data["structured_extraction"] = None
                
                # Step 3b: Hyperbolic.ai enhancement for additional analysis
                if hasattr(self, 'hyperbolic_llm'):
                    analysis_prompt = f"""
                    Analyze this educational content chunk and provide:
                    1. Content difficulty level (beginner/intermediate/advanced)
                    2. Subject classification (business/technical/creative/etc)
                    3. Key relationships to other topics
                    4. Suggested prerequisite knowledge
                    
                    Chunk: {node.text[:500]}...
                    
                    Respond with JSON format:
                    {{
                        "difficulty_level": "beginner|intermediate|advanced",
                        "subject_classification": "primary_subject",
                        "related_topics": ["topic1", "topic2"],
                        "prerequisites": ["prereq1", "prereq2"],
                        "content_type": "theory|practical|example|exercise"
                    }}
                    """
                    
                    try:
                        analysis = await self.hyperbolic_llm.complete(analysis_prompt)
                        enhanced_metadata = json.loads(analysis)
                        chunk_data["ai_analysis"] = enhanced_metadata
                    except Exception as e:
                        logger.warning(f"Hyperbolic analysis failed for chunk {i}: {e}")
                        chunk_data["ai_analysis"] = None
                
                processed_chunks.append(chunk_data)
            
            return {
                "status": "success",
                "filename": filename,
                "total_chunks": len(processed_chunks),
                "chunks": processed_chunks,
                "processing_summary": {
                    "llamaindex_chunks": len(nodes),
                    "langextract_processed": sum(1 for c in processed_chunks if c.get("structured_extraction")),
                    "hyperbolic_enhanced": sum(1 for c in processed_chunks if c.get("ai_analysis"))
                },
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    
    async def _classify_and_tag_content(self, text: str, filename: str) -> dict:
        """Use Hyperbolic.ai to intelligently classify content and generate tags - completely dynamic"""
        if not hasattr(self, 'hyperbolic_llm'):
            return {"industry": "general", "tags": [], "department_relevance": []}
            
        classification_prompt = f"""
        Analyze this course content and classify it for business AI departments.
        Be completely dynamic - don't limit yourself to predefined categories.
        
        Filename: {filename}
        Content: {text[:1000]}...
        
        Provide classification in this exact JSON format:
        {{
            "industry": "the actual industry this content belongs to (be specific and accurate)",
            "sub_categories": ["specific sub-areas within this industry"],
            "difficulty_level": "beginner|intermediate|advanced|expert",
            "content_type": "theory|practical|case-study|framework|tool|template|certification|interview-prep",
            "department_relevance": [
                {{"department": "most_relevant_dept", "relevance_score": 0.9, "specific_use": "how this dept would use this"}},
                {{"department": "secondary_dept", "relevance_score": 0.7, "specific_use": "secondary application"}}
            ],
            "tags": ["dynamic tags based on actual content"],
            "business_value": "specific value this knowledge provides",
            "key_frameworks": ["any named methodologies, frameworks, or systems mentioned"],
            "target_audience": "who specifically should learn this",
            "prerequisites": ["what knowledge is assumed"],
            "learning_outcomes": ["what students will be able to do after learning this"],
            "industry_keywords": ["domain-specific terms and jargon used"],
            "skill_level_indicators": ["what suggests the difficulty level"],
            "practical_applications": ["real-world use cases mentioned"]
        }}
        
        Important: Base your classification entirely on the actual content. Don't use predefined lists.
        """
        
        try:
            classification_result = await self.hyperbolic_llm.complete(classification_prompt)
            return json.loads(classification_result)
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            # Fallback with basic analysis
            return {
                "industry": "unknown", 
                "tags": [filename.replace('.pdf', '').replace('.docx', '').replace('_', ' ').replace('-', ' ')],
                "department_relevance": [],
                "sub_categories": [],
                "content_type": "unknown"
            }

    async def _store_in_backblaze(self, content: bytes, file_path: str) -> str:
        """Store processed content in Backblaze B2"""
        if not B2_AVAILABLE:
            logger.warning("Backblaze not configured, skipping storage")
            return None
            
        try:
            # Upload to B2
            uploaded_file = bucket.upload_bytes(
                content, 
                file_path,
                content_type="application/octet-stream"
            )
            
            # Return the download URL
            download_url = b2_api.get_download_url_for_file_name(B2_BUCKET_NAME, file_path)
            return download_url
            
        except Exception as e:
            logger.error(f"Backblaze upload failed: {e}")
            return None

# Initialize processor
document_processor = DocumentProcessor()

@app.get("/")
async def root():
    return {
        "service": "LlamaIndex + Hyperbolic.ai Document Processor",
        "status": "running",
        "version": "1.0.0",
        "hyperbolic_enabled": bool(HYPERBOLIC_API_KEY),
        "llamaindex_available": LLAMAINDEX_AVAILABLE
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "llamaindex": LLAMAINDEX_AVAILABLE,
            "langextract": LANGEXTRACT_AVAILABLE,
            "hyperbolic": bool(HYPERBOLIC_API_KEY),
            "backblaze": B2_AVAILABLE,
            "pinecone": PINECONE_AVAILABLE,
            "document_processor": document_processor.available
        }
    }

@app.post("/process-course")
async def process_course_material(
    file: UploadFile = File(...),
    course_name: str = Form(...),
    module_name: str = Form(default=""),
    industry: str = Form(default="auto-detect")
):
    """Process course material with enhanced classification for AI departments"""
    
    if not document_processor.available:
        raise HTTPException(status_code=503, detail="Document processor not available")
    
    try:
        content = await file.read()
        
        # Enhanced processing with course context
        result = await document_processor.process_document(
            file_path=f"courses/{course_name}/{module_name}/{file.filename}",
            content=content,
            filename=file.filename
        )
        
        # Add course-specific metadata
        result["course_metadata"] = {
            "course_name": course_name,
            "module_name": module_name,
            "requested_industry": industry,
            "detected_industry": result.get("classification", {}).get("industry", "unknown"),
            "department_recommendations": result.get("classification", {}).get("department_relevance", [])
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Course processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Course processing error: {e}")

@app.get("/courses")
async def list_courses():
    """List all courses stored in Backblaze"""
    if not B2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backblaze not configured")
    
    try:
        # List files in courses/ folder
        files_info = []
        for file_version, folder_path in bucket.ls(folder_to_list="courses/", recursive=True):
            if file_version:
                files_info.append({
                    "filename": file_version.file_name,
                    "size": file_version.size,
                    "upload_timestamp": file_version.upload_timestamp,
                    "download_url": b2_api.get_download_url_for_file_name(B2_BUCKET_NAME, file_version.file_name)
                })
        
        # Group by course
        courses = {}
        for file_info in files_info:
            path_parts = file_info["filename"].split('/')
            if len(path_parts) >= 3:  # courses/course-name/file
                course_name = path_parts[1]
                if course_name not in courses:
                    courses[course_name] = []
                courses[course_name].append(file_info)
        
        return {
            "status": "success",
            "total_courses": len(courses),
            "courses": courses,
            "total_files": len(files_info)
        }
        
    except Exception as e:
        logger.error(f"Course listing error: {e}")
        raise HTTPException(status_code=500, detail=f"Course listing error: {e}")

@app.post("/search-knowledge")
async def search_course_knowledge(
    query: str = Form(...),
    industry_filter: str = Form(default=""),
    department_filter: str = Form(default=""),
    difficulty_filter: str = Form(default=""),
    limit: int = Form(default=10)
):
    """Search your course knowledge base with dynamic filtering"""
    
    if not PINECONE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Pinecone not configured")
    
    try:
        # Generate embedding for the query (placeholder for now)
        query_embedding = [0.0] * 1536  # Replace with actual embedding generation
        
        # Build dynamic filter based on classification
        filter_conditions = {}
        
        if industry_filter:
            filter_conditions["industry"] = industry_filter
        if department_filter:
            filter_conditions["department_tags"] = {"$in": [department_filter]}
        if difficulty_filter:
            filter_conditions["difficulty_level"] = difficulty_filter
        
        # Search Pinecone
        search_results = index.query(
            vector=query_embedding,
            filter=filter_conditions if filter_conditions else None,
            top_k=limit,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in search_results.matches:
            result = {
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "source": {
                    "file": match.metadata.get("source_file", ""),
                    "chunk_index": match.metadata.get("chunk_index", 0),
                    "backblaze_url": match.metadata.get("backblaze_url", ""),
                    "course_name": match.metadata.get("course_name", "")
                },
                "classification": {
                    "industry": match.metadata.get("industry", ""),
                    "department_tags": match.metadata.get("department_tags", []),
                    "difficulty_level": match.metadata.get("difficulty_level", ""),
                    "content_type": match.metadata.get("content_type", "")
                },
                "tags": match.metadata.get("search_tags", []),
                "key_concepts": match.metadata.get("key_concepts", [])
            }
            formatted_results.append(result)
        
        return {
            "status": "success",
            "query": query,
            "filters_applied": filter_conditions,
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Knowledge search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

@app.get("/knowledge-stats")
async def get_knowledge_statistics():
    """Get statistics about your course knowledge base"""
    
    if not PINECONE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Pinecone not configured")
    
    try:
        # Get index stats
        stats = index.describe_index_stats()
        
        return {
            "status": "success",
            "pinecone_stats": {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            },
            "dynamic_classification": "Industries, departments, and tags discovered from your actual course content"
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {e}")

@app.post("/extract")
async def extract_structured_data(
    file: UploadFile = File(...),
    schema: str = Form(default='{"key_concepts": "Main concepts", "facts": "Key facts", "quotes": "Important quotes"}')
):
    """Extract structured data with precise source grounding using LangExtract"""
    
    if not document_processor.available or not document_processor.lang_extractor:
        raise HTTPException(status_code=503, detail="LangExtract not available")
    
    try:
        content = await file.read()
        
        # Extract full text first
        if file.filename.endswith('.pdf'):
            with open(f"/tmp/{file.filename}", "wb") as f:
                f.write(content)
            documents = document_processor.pdf_reader.load_data(f"/tmp/{file.filename}")
            full_text = "\n".join([doc.text for doc in documents])
            os.remove(f"/tmp/{file.filename}")
        elif file.filename.endswith('.docx'):
            with open(f"/tmp/{file.filename}", "wb") as f:
                f.write(content)
            documents = document_processor.docx_reader.load_data(f"/tmp/{file.filename}")
            full_text = "\n".join([doc.text for doc in documents])
            os.remove(f"/tmp/{file.filename}")
        else:
            full_text = content.decode('utf-8')
        
        # Parse schema
        extraction_schema = json.loads(schema)
        
        # Use LangExtract for precise extraction with source grounding
        extracted_result = document_processor.lang_extractor.extract(
            text=full_text,
            schema=extraction_schema,
            source_metadata={
                "filename": file.filename,
                "processed_at": datetime.now().isoformat()
            }
        )
        
        return {
            "status": "success",
            "filename": file.filename,
            "schema_used": extraction_schema,
            "extracted_data": extracted_result,
            "source_grounding": "Exact character positions and quotes preserved by LangExtract",
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Structured extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction error: {e}")

@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    enhance_with_ai: bool = Form(default=True),
    extract_structured: bool = Form(default=True)
):
    """Process uploaded document with LlamaIndex + optional Hyperbolic.ai enhancement + LangExtract"""
    
    if not document_processor.available:
        raise HTTPException(status_code=503, detail="Document processor not available")
    
    # Read file content
    content = await file.read()
    
    # Process with our enhanced pipeline
    result = await document_processor.process_document(
        file_path=file.filename,
        content=content,
        filename=file.filename
    )
    
    return result

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None)
):
    """Transcribe audio/video using Hyperbolic.ai Whisper models"""
    
    if not HYPERBOLIC_API_KEY:
        raise HTTPException(status_code=503, detail="Hyperbolic API key not configured")
    
    try:
        # Prepare file for Hyperbolic Whisper API
        headers = {
            "Authorization": f"Bearer {HYPERBOLIC_API_KEY}"
        }
        
        # Create form data for file upload
        files = {
            "file": (file.filename, await file.read(), file.content_type)
        }
        data = {
            "model": "whisper-large-v3"
        }
        if language:
            data["language"] = language
        
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for large files
            response = await client.post(
                f"{HYPERBOLIC_BASE_URL}/audio/transcriptions",
                headers=headers,
                files=files,
                data=data
            )
            response.raise_for_status()
            
            transcription_result = response.json()
            
            # Return structured result
            return {
                "status": "success",
                "filename": file.filename,
                "transcription": transcription_result.get("text", ""),
                "language": transcription_result.get("language"),
                "processed_at": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")

@app.get("/models")
async def list_available_models():
    """List available models from Hyperbolic.ai"""
    
    if not HYPERBOLIC_API_KEY:
        return {"error": "Hyperbolic API key not configured"}
    
    headers = {
        "Authorization": f"Bearer {HYPERBOLIC_API_KEY}"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HYPERBOLIC_BASE_URL}/models", headers=headers)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": f"Failed to fetch models: {e}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)