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
    from llama_index.vector_stores.pinecone import PineconeVectorStore
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

# Airtable imports
try:
    import requests
    AIRTABLE_AVAILABLE = True
    print("SUCCESS: Airtable integration ready")
except ImportError as e:
    print(f"WARNING: Could not import requests for Airtable: {e}")
    AIRTABLE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LlamaIndex + Hyperbolic.ai + Airtable Document Processor", version="2.1.0")

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

# 🚀 UNIFIED PINECONE CONFIGURATION
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ai-empire-organizational-intelligence")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "organizational_intelligence")

# Lakera Guard configuration
LAKERA_API_KEY = os.getenv("LAKERA_API_KEY")
LAKERA_BASE_URL = "https://api.lakera.ai/v1"

# Airtable configuration
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_BASE_URL = "https://api.airtable.com/v0"

# Unified Pinecone setup
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = bool(PINECONE_API_KEY)
    if PINECONE_AVAILABLE:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Configure LlamaIndex to use unified vector store
        vector_store = PineconeVectorStore(
            pinecone_index=index,
            namespace=PINECONE_NAMESPACE,
            add_sparse_vector=False
        )
        print(f"✅ Connected to unified Pinecone index: {PINECONE_INDEX_NAME}")
        print(f"✅ Using namespace: {PINECONE_NAMESPACE}")
    else:
        index = None
        vector_store = None
        print("❌ Pinecone not configured")
except ImportError as e:
    print(f"Pinecone not available: {e}")
    PINECONE_AVAILABLE = False
    index = None
    vector_store = None

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

class LakeraGuard:
    """Lakera Guard integration for AI security"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = LAKERA_BASE_URL
        self.available = bool(api_key)
    
    async def screen_input(self, text: str, user_id: str = "anonymous") -> dict:
        """Screen user input for security threats"""
        if not self.available:
            return {"safe": True, "message": "Security screening disabled"}
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": text,
                "response_categories": [
                    "prompt_injection",
                    "jailbreak", 
                    "pii",
                    "toxic"
                ],
                "user_id": user_id
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/detect",
                    headers=headers,
                    json=payload,
                    timeout=10.0
                )
                response.raise_for_status()
                result = response.json()
                
                return {
                    "safe": not result.get("flagged", False),
                    "categories": result.get("categories", []),
                    "message": "Input screened successfully",
                    "lakera_response": result
                }
                
        except Exception as e:
            logger.error(f"Lakera Guard screening error: {e}")
            # Fail open - allow processing if security check fails
            return {"safe": True, "message": f"Security check failed: {e}"}
    
    async def screen_output(self, text: str) -> dict:
        """Screen AI output before returning to user"""
        if not self.available:
            return {"safe": True, "message": "Output screening disabled"}
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": text,
                "response_categories": [
                    "pii",
                    "toxic",
                    "hate",
                    "self_harm"
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/detect",
                    headers=headers,
                    json=payload,
                    timeout=10.0
                )
                response.raise_for_status()
                result = response.json()
                
                return {
                    "safe": not result.get("flagged", False),
                    "categories": result.get("categories", []),
                    "message": "Output screened successfully",
                    "lakera_response": result
                }
                
        except Exception as e:
            logger.error(f"Lakera Guard output screening error: {e}")
            # Fail open - allow output if security check fails
            return {"safe": True, "message": f"Output security check failed: {e}"}

class AirtableDocumentLogger:
    """Logs document processing activities to Airtable"""
    
    def __init__(self):
        self.available = False
        
        if not AIRTABLE_AVAILABLE:
            print("WARNING: Requests library not available, Airtable logging disabled")
            return
            
        try:
            if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID:
                print("WARNING: Airtable credentials missing, processing logging disabled")
                return
                
            self.api_key = AIRTABLE_API_KEY
            self.base_id = AIRTABLE_BASE_ID
            self.base_url = f"{AIRTABLE_BASE_URL}/{self.base_id}"
            self.available = True
            print("SUCCESS: Airtable document processing logger initialized")
            
        except Exception as e:
            print(f"ERROR: Failed to initialize Airtable client: {e}")
            self.available = False
    
    def _get_headers(self):
        """Get headers for Airtable API requests"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def log_processing_start(self, filename: str, file_size: int, user_id: str = "system", 
                                 course_name: str = "", module_name: str = "") -> str:
        """Log the start of document processing to Airtable"""
        if not self.available:
            return None
            
        try:
            # Generate a unique session ID
            import uuid
            session_id = str(uuid.uuid4())
            
            processing_data = {
                "fields": {
                    "Session ID": session_id,
                    "File Name": filename,
                    "File Size (Bytes)": file_size,
                    "User ID": user_id,
                    "Course Name": course_name,
                    "Module Name": module_name,
                    "Processing Status": "Processing",
                    "Started At": datetime.now().isoformat(),
                    "Service": "llamaindex-unified-processor",
                    "Pinecone Index": PINECONE_INDEX_NAME,
                    "Pinecone Namespace": PINECONE_NAMESPACE
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/Documents",
                    headers=self._get_headers(),
                    json=processing_data,
                    timeout=10.0
                )
                response.raise_for_status()
                result = response.json()
                
                # Store the Airtable record ID for updates
                record_id = result["id"]
            
            print(f"SUCCESS: Logged processing start for {filename} (Session: {session_id}, Record: {record_id})")
            return f"{session_id}|{record_id}"  # Return both session and record ID
            
        except Exception as e:
            print(f"ERROR: Failed to log processing start to Airtable: {e}")
            return None
    
    async def log_processing_complete(self, session_record_id: str, chunks_created: int, 
                                    processing_summary: dict, classification: dict = None,
                                    security_result: dict = None, processing_time: float = None,
                                    backblaze_url: str = None) -> bool:
        """Log successful completion of document processing to Airtable"""
        if not self.available or not session_record_id:
            return False
            
        try:
            # Parse session_record_id to get Airtable record ID
            session_id, record_id = session_record_id.split("|")
            
            update_data = {
                "fields": {
                    "Processing Status": "Complete",
                    "Completed At": datetime.now().isoformat(),
                    "Chunks Created": chunks_created,
                    "Success Rate": "100%",
                    "Pipeline Version": "ai_empire_v2.1_unified"
                }
            }
            
            # Add optional fields if provided
            if processing_time:
                update_data["fields"]["Processing Time (Seconds)"] = round(processing_time, 2)
            
            if classification:
                update_data["fields"]["Industry"] = classification.get("industry", "")
                update_data["fields"]["Content Type"] = classification.get("content_type", "")
                update_data["fields"]["Difficulty Level"] = classification.get("difficulty_level", "")
            
            if security_result:
                update_data["fields"]["Security Status"] = "Safe" if security_result.get("safe") else "Flagged"
                if security_result.get("categories"):
                    update_data["fields"]["Security Notes"] = str(security_result["categories"])
            
            if backblaze_url:
                update_data["fields"]["Backblaze URL"] = backblaze_url
            
            # Add processing summary as notes
            if processing_summary:
                summary_text = f"LlamaIndex: {processing_summary.get('llamaindex_chunks', 0)} chunks, "
                summary_text += f"LangExtract: {processing_summary.get('langextract_processed', 0)} enhanced, "
                summary_text += f"Hyperbolic: {processing_summary.get('hyperbolic_enhanced', 0)} analyzed"
                update_data["fields"]["Processing Notes"] = summary_text
            
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{self.base_url}/Documents/{record_id}",
                    headers=self._get_headers(),
                    json=update_data,
                    timeout=10.0
                )
                response.raise_for_status()
            
            print(f"SUCCESS: Logged processing completion for session {session_id}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to log processing completion to Airtable: {e}")
            return False
    
    async def log_processing_error(self, session_record_id: str, error_message: str) -> bool:
        """Log processing failure to Airtable"""
        if not self.available or not session_record_id:
            return False
            
        try:
            # Parse session_record_id to get Airtable record ID
            session_id, record_id = session_record_id.split("|")
            
            update_data = {
                "fields": {
                    "Processing Status": "Failed",
                    "Completed At": datetime.now().isoformat(),
                    "Error Message": error_message,
                    "Success Rate": "0%"
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{self.base_url}/Documents/{record_id}",
                    headers=self._get_headers(),
                    json=update_data,
                    timeout=10.0
                )
                response.raise_for_status()
            
            print(f"SUCCESS: Logged processing error for session {session_id}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to log processing error to Airtable: {e}")
            return False
    
    async def get_processing_stats(self) -> dict:
        """Get processing statistics from Airtable"""
        if not self.available:
            return {"error": "Airtable not available"}
            
        try:
            # Get recent processing stats from Airtable
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/Documents?maxRecords=100&sort[0][field]=Started At&sort[0][direction]=desc",
                    headers=self._get_headers(),
                    timeout=15.0
                )
                response.raise_for_status()
                result = response.json()
            
            if not result.get("records"):
                return {"total_processed": 0, "success_rate": 0}
            
            records = result["records"]
            total = len(records)
            successful = len([r for r in records if r["fields"].get("Processing Status") == "Complete"])
            total_chunks = sum(r["fields"].get("Chunks Created", 0) for r in records)
            
            # Calculate file sizes
            total_size = sum(r["fields"].get("File Size (Bytes)", 0) for r in records)
            avg_size = total_size / total if total > 0 else 0
            
            return {
                "total_processed": total,
                "successful": successful,
                "failed": total - successful,
                "success_rate": round((successful / total * 100), 2) if total > 0 else 0,
                "total_chunks_created": total_chunks,
                "total_data_processed_mb": round(total_size / (1024*1024), 2),
                "avg_file_size_mb": round(avg_size / (1024*1024), 2),
                "pinecone_index": PINECONE_INDEX_NAME,
                "pinecone_namespace": PINECONE_NAMESPACE,
                "recent_activity": [
                    {
                        "filename": r["fields"].get("File Name", ""),
                        "status": r["fields"].get("Processing Status", ""),
                        "chunks": r["fields"].get("Chunks Created", 0),
                        "processed_at": r["fields"].get("Started At", ""),
                        "course": r["fields"].get("Course Name", "")
                    }
                    for r in records[:10]
                ]
            }
            
        except Exception as e:
            print(f"ERROR: Failed to get processing stats from Airtable: {e}")
            return {"error": f"Failed to get stats: {e}"}

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
                raise HTTPException(status_code=500, detail=f"Hyperbolic API error: {e}")

class DocumentProcessor:
    """Document processing with LlamaIndex + Hyperbolic.ai + LangExtract + Lakera Security + Airtable Logging"""
    
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
        
        # Initialize Lakera Guard for security
        self.lakera_guard = LakeraGuard(LAKERA_API_KEY) if LAKERA_API_KEY else None
        if self.lakera_guard and self.lakera_guard.available:
            logger.info("Lakera Guard security enabled")
        else:
            logger.warning("Lakera Guard not configured - security screening disabled")
        
        # Initialize Airtable logger
        self.airtable_logger = AirtableDocumentLogger()
        if self.airtable_logger.available:
            logger.info("Airtable document processing logging enabled")
        else:
            logger.warning("Airtable logging not available")
        
        # Set up OpenAI for embeddings
        if os.getenv("OPENAI_API_KEY"):
            Settings.embed_model = OpenAIEmbedding()
        
        self.available = True
        logger.info("Document processor initialized successfully with unified Pinecone architecture")
    
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

    def _create_unified_metadata(self, filename: str, chunk_index: int, total_chunks: int, 
                               course_name: str = "", module_name: str = "", 
                               classification: dict = None) -> dict:
        """Create metadata following unified schema for CrewAI agent filtering"""
        
        metadata = {
            # System identification
            "source_system": "ai_empire_pipeline",
            "processor": "llamaindex",
            
            # Document identification  
            "document_id": f"{filename}_{chunk_index}_{total_chunks}",
            "filename": filename,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            
            # Business context (for agent filtering)
            "course": course_name,
            "module": module_name,
            "department": "general",  # Default, can be enhanced
            
            # Analysis results (defaults)
            "analysis_type": "comprehensive",
            "strategic_value": "moderate",
            "executive_approved": False,
            
            # Content characteristics
            "content_type": "markdown",
            "has_visual_content": False,
            "visual_elements": [],
            
            # Processing metadata
            "processing_timestamp": datetime.now().isoformat(),
            "pipeline_version": "ai_empire_v2.1_unified",
            "embedding_model": "text-embedding-3-small"
        }
        
        # Enhance with classification if available
        if classification:
            metadata.update({
                "industry": classification.get("industry", "general"),
                "difficulty_level": classification.get("difficulty_level", "intermediate"),
                "content_type": classification.get("content_type", "theory"),
                "department": classification.get("primary_department", "general"),
                "tags": classification.get("tags", []),
                "business_value": classification.get("business_value", ""),
                "target_audience": classification.get("target_audience", "")
            })
            
            # Set strategic value based on classification
            if "strategic" in classification.get("keywords", []):
                metadata["strategic_value"] = "urgent"
            elif "executive" in classification.get("keywords", []):
                metadata["executive_approved"] = True
                
        return metadata
    
    async def process_document(self, file_path: str, content: bytes, filename: str, 
                             user_id: str = "system", course_name: str = "", 
                             module_name: str = "") -> dict:
        """Process document with full pipeline: Lakera Security + LlamaIndex + LangExtract + Hyperbolic + Airtable Logging + Unified Pinecone"""
        if not self.available:
            raise HTTPException(status_code=503, detail="Document processor not available")
        
        # Initialize processing session for logging
        session_record_id = None
        processing_start_time = datetime.now()
        
        try:
            # Step 0: Start Airtable logging
            if self.airtable_logger.available:
                session_record_id = await self.airtable_logger.log_processing_start(
                    filename=filename,
                    file_size=len(content),
                    user_id=user_id,
                    course_name=course_name,
                    module_name=module_name
                )
            
            # Step 1: Security screening of filename and initial content
            if self.lakera_guard and self.lakera_guard.available:
                # Screen filename for potential security issues
                filename_check = await self.lakera_guard.screen_input(filename, user_id)
                if not filename_check["safe"]:
                    if session_record_id:
                        await self.airtable_logger.log_processing_error(
                            session_record_id, f"Filename blocked by security: {filename_check['categories']}"
                        )
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Filename blocked by security: {filename_check['categories']}"
                    )
                
                # Screen initial content sample for security
                content_sample = content[:1000].decode('utf-8', errors='ignore')
                content_check = await self.lakera_guard.screen_input(content_sample, user_id)
                if not content_check["safe"]:
                    if session_record_id:
                        await self.airtable_logger.log_processing_error(
                            session_record_id, f"Content blocked by security: {content_check['categories']}"
                        )
                    raise HTTPException(
                        status_code=400,
                        detail=f"Content blocked by security: {content_check['categories']}"
                    )
                    
                logger.info(f"Content passed Lakera Guard security screening: {filename}")
            
            # Step 2: Extract text with LlamaIndex
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
            
            # Step 3: Parse into nodes/chunks
            nodes = self.node_parser.get_nodes_from_documents(documents)
            
            # Step 4: Create VectorStoreIndex using unified Pinecone
            processed_chunks = []
            pinecone_vectors = []
            
            for i, node in enumerate(nodes):
                chunk_data = {
                    "text": node.text,
                    "metadata": node.metadata,
                    "chunk_id": node.node_id,
                    "source": filename,
                    "chunk_index": i
                }
                
                # Step 4a: LangExtract structured extraction with source grounding
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
                
                # Step 4b: Hyperbolic.ai enhancement for additional analysis
                classification = None
                if hasattr(self, 'hyperbolic_llm'):
                    analysis_prompt = f"""
                    Analyze this educational content chunk and provide classification for AI agent filtering:
                    
                    Chunk: {node.text[:500]}...
                    Filename: {filename}
                    Course: {course_name}
                    
                    Respond with JSON format:
                    {{
                        "industry": "specific_industry_category",
                        "difficulty_level": "beginner|intermediate|advanced",
                        "content_type": "theory|practical|case-study|framework",
                        "primary_department": "operations|finance|hr|it|marketing|general",
                        "tags": ["relevant", "searchable", "keywords"],
                        "business_value": "specific value this provides",
                        "target_audience": "who should learn this",
                        "keywords": ["strategic", "executive", "technical", "operational"]
                    }}
                    """
                    
                    try:
                        analysis = await self.hyperbolic_llm.complete(analysis_prompt)
                        classification = json.loads(analysis)
                        chunk_data["ai_analysis"] = classification
                    except Exception as e:
                        logger.warning(f"Hyperbolic analysis failed for chunk {i}: {e}")
                        chunk_data["ai_analysis"] = None
                        classification = None
                
                # Step 4c: Create unified metadata for Pinecone
                unified_metadata = self._create_unified_metadata(
                    filename=filename,
                    chunk_index=i,
                    total_chunks=len(nodes),
                    course_name=course_name,
                    module_name=module_name,
                    classification=classification
                )
                
                # Update node metadata with unified schema
                node.metadata.update(unified_metadata)
                chunk_data["unified_metadata"] = unified_metadata
                
                processed_chunks.append(chunk_data)
            
            # Step 5: Store in unified Pinecone index using LlamaIndex
            if vector_store and PINECONE_AVAILABLE:
                try:
                    # Create index from nodes with unified metadata
                    index_from_nodes = VectorStoreIndex(nodes, vector_store=vector_store)
                    logger.info(f"✅ Stored {len(nodes)} chunks in unified Pinecone index: {PINECONE_INDEX_NAME}")
                    logger.info(f"✅ Used namespace: {PINECONE_NAMESPACE}")
                except Exception as e:
                    logger.error(f"Failed to store in Pinecone: {e}")
            
            # Step 6: Security screening of final output
            security_result = {"safe": True, "message": "No security screening"}
            if self.lakera_guard and self.lakera_guard.available:
                # Screen the processed content for potential issues
                output_sample = str(processed_chunks)[:2000]  # Sample for screening
                output_check = await self.lakera_guard.screen_output(output_sample)
                security_result = output_check
                
                if not output_check["safe"]:
                    logger.warning(f"Processed content flagged by Lakera Guard: {output_check['categories']}")
                    # Still allow processing but flag for review
            
            # Step 7: Optional Backblaze storage
            backblaze_url = None
            if B2_AVAILABLE:
                try:
                    backblaze_url = await self._store_in_backblaze(content, file_path)
                except Exception as e:
                    logger.warning(f"Backblaze storage failed: {e}")
            
            # Step 8: Log successful completion to Airtable
            processing_summary = {
                "llamaindex_chunks": len(nodes),
                "langextract_processed": sum(1 for c in processed_chunks if c.get("structured_extraction")),
                "hyperbolic_enhanced": sum(1 for c in processed_chunks if c.get("ai_analysis")),
                "pinecone_stored": len(nodes) if vector_store else 0
            }
            
            # Get classification from first chunk for Airtable
            classification = None
            if processed_chunks and processed_chunks[0].get("ai_analysis"):
                classification = processed_chunks[0]["ai_analysis"]
            
            # Calculate processing time
            processing_time = (datetime.now() - processing_start_time).total_seconds()
            
            if session_record_id and self.airtable_logger.available:
                await self.airtable_logger.log_processing_complete(
                    session_record_id=session_record_id,
                    chunks_created=len(processed_chunks),
                    processing_summary=processing_summary,
                    classification=classification,
                    security_result=security_result,
                    processing_time=processing_time,
                    backblaze_url=backblaze_url
                )
            
            return {
                "status": "success",
                "filename": filename,
                "total_chunks": len(processed_chunks),
                "chunks": processed_chunks,
                "processing_summary": processing_summary,
                "security": security_result,
                "session_id": session_record_id.split("|")[0] if session_record_id else None,
                "processing_time_seconds": processing_time,
                "processed_at": datetime.now().isoformat(),
                "backblaze_url": backblaze_url,
                "pinecone_index": PINECONE_INDEX_NAME,
                "pinecone_namespace": PINECONE_NAMESPACE,
                "pipeline_version": "ai_empire_v2.1_unified"
            }
            
        except Exception as e:
            # Log error to Airtable if possible
            if session_record_id and self.airtable_logger.available:
                await self.airtable_logger.log_processing_error(session_record_id, str(e))
            
            logger.error(f"Document processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    
    async def _extract_with_langextract(self, text: str, schema: dict, filename: str, chunk_index: int) -> dict:
        """Helper method for LangExtract processing"""
        if not self.lang_extractor:
            return None
            
        try:
            # Use LangExtract for precise extraction
            result = self.lang_extractor.extract(
                text=text,
                schema=schema,
                source_metadata={
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "processed_at": datetime.now().isoformat()
                }
            )
            return result
        except Exception as e:
            logger.error(f"LangExtract extraction failed: {e}")
            return None

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
        "service": "LlamaIndex + Hyperbolic.ai + Airtable Document Processor",
        "status": "running",
        "version": "2.1.0",
        "architecture": "unified_pinecone",
        "pinecone_index": PINECONE_INDEX_NAME,
        "pinecone_namespace": PINECONE_NAMESPACE,
        "hyperbolic_enabled": bool(HYPERBOLIC_API_KEY),
        "llamaindex_available": LLAMAINDEX_AVAILABLE,
        "langextract_available": LANGEXTRACT_AVAILABLE,
        "security_enabled": bool(LAKERA_API_KEY),
        "airtable_logging": bool(AIRTABLE_API_KEY and AIRTABLE_BASE_ID),
        "unified_features": [
            "✅ Single Pinecone index for all content",
            "✅ Enhanced metadata schema for agent filtering", 
            "✅ Cross-departmental insights capability",
            "✅ Executive-level content tagging",
            "✅ CrewAI agent compatibility"
        ]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "architecture": "unified_pinecone_v2.1",
        "components": {
            "llamaindex": LLAMAINDEX_AVAILABLE,
            "langextract": LANGEXTRACT_AVAILABLE,
            "hyperbolic": bool(HYPERBOLIC_API_KEY),
            "backblaze": B2_AVAILABLE,
            "pinecone": PINECONE_AVAILABLE,
            "lakera_guard": bool(LAKERA_API_KEY),
            "airtable": AIRTABLE_AVAILABLE and bool(AIRTABLE_API_KEY and AIRTABLE_BASE_ID),
            "document_processor": document_processor.available,
            "unified_vector_store": vector_store is not None
        },
        "pinecone_config": {
            "index": PINECONE_INDEX_NAME,
            "namespace": PINECONE_NAMESPACE,
            "connected": PINECONE_AVAILABLE
        }
    }

@app.post("/process-course")
async def process_course_material(
    file: UploadFile = File(...),
    course_name: str = Form(...),
    module_name: str = Form(default=""),
    industry: str = Form(default="auto-detect"),
    user_id: str = Form(default="anonymous")
):
    """Process course material with enhanced classification, security screening, and unified Pinecone storage"""
    
    if not document_processor.available:
        raise HTTPException(status_code=503, detail="Document processor not available")
    
    try:
        content = await file.read()
        
        # Enhanced processing with course context and security + Airtable logging + Unified Pinecone
        result = await document_processor.process_document(
            file_path=f"courses/{course_name}/{module_name}/{file.filename}",
            content=content,
            filename=file.filename,
            user_id=user_id,
            course_name=course_name,
            module_name=module_name
        )
        
        # Add course-specific metadata
        result["course_metadata"] = {
            "course_name": course_name,
            "module_name": module_name,
            "requested_industry": industry,
            "detected_industry": result.get("classification", {}).get("industry", "unknown"),
            "department_recommendations": result.get("classification", {}).get("department_relevance", []),
            "user_id": user_id
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Course processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Course processing error: {e}")

@app.post("/search-knowledge")
async def search_course_knowledge(
    query: str = Form(...),
    industry_filter: str = Form(default=""),
    department_filter: str = Form(default=""),
    difficulty_filter: str = Form(default=""),
    limit: int = Form(default=10)
):
    """Search your course knowledge base using unified Pinecone index with dynamic filtering"""
    
    if not PINECONE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Pinecone not configured")
    
    try:
        # Generate embedding for the query using OpenAI (same as used for storage)
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = embedding_response.data[0].embedding
        
        # Build dynamic filter based on unified metadata schema
        filter_conditions = {}
        
        if industry_filter:
            filter_conditions["industry"] = industry_filter
        if department_filter:
            filter_conditions["department"] = department_filter
        if difficulty_filter:
            filter_conditions["difficulty_level"] = difficulty_filter
        
        # Search unified Pinecone index
        search_results = index.query(
            vector=query_embedding,
            filter=filter_conditions if filter_conditions else None,
            top_k=limit,
            namespace=PINECONE_NAMESPACE,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in search_results.matches:
            result = {
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "source": {
                    "file": match.metadata.get("filename", ""),
                    "chunk_index": match.metadata.get("chunk_index", 0),
                    "document_id": match.metadata.get("document_id", ""),
                    "course_name": match.metadata.get("course", ""),
                    "module_name": match.metadata.get("module", "")
                },
                "classification": {
                    "industry": match.metadata.get("industry", ""),
                    "department": match.metadata.get("department", ""),
                    "difficulty_level": match.metadata.get("difficulty_level", ""),
                    "content_type": match.metadata.get("content_type", ""),
                    "strategic_value": match.metadata.get("strategic_value", ""),
                    "executive_approved": match.metadata.get("executive_approved", False)
                },
                "tags": match.metadata.get("tags", []),
                "business_value": match.metadata.get("business_value", "")
            }
            formatted_results.append(result)
        
        return {
            "status": "success",
            "query": query,
            "filters_applied": filter_conditions,
            "total_results": len(formatted_results),
            "results": formatted_results,
            "pinecone_index": PINECONE_INDEX_NAME,
            "pinecone_namespace": PINECONE_NAMESPACE,
            "architecture": "unified_v2.1"
        }
        
    except Exception as e:
        logger.error(f"Knowledge search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

@app.get("/knowledge-stats")
async def get_knowledge_statistics():
    """Get statistics about your unified knowledge base"""
    
    if not PINECONE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Pinecone not configured")
    
    try:
        # Get index stats
        stats = index.describe_index_stats()
        
        # Get namespace-specific stats
        namespace_stats = stats.namespaces.get(PINECONE_NAMESPACE, {})
        
        return {
            "status": "success",
            "architecture": "unified_pinecone_v2.1",
            "pinecone_stats": {
                "index_name": PINECONE_INDEX_NAME,
                "namespace": PINECONE_NAMESPACE,
                "total_vectors_in_index": stats.total_vector_count,
                "vectors_in_namespace": namespace_stats.get('vector_count', 0),
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            },
            "benefits": [
                "✅ Single source of truth for all AI agents",
                "✅ Enhanced metadata for smart agent filtering", 
                "✅ Cross-departmental insight capability",
                "✅ Executive-level content prioritization",
                "✅ Consistent intelligence across all agents"
            ]
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {e}")

@app.get("/stats/processing")
async def get_processing_statistics():
    """Get document processing statistics from Airtable"""
    
    if not document_processor.airtable_logger.available:
        raise HTTPException(status_code=503, detail="Airtable logging not available")
    
    try:
        stats = await document_processor.airtable_logger.get_processing_stats()
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "data_source": "Airtable",
            "architecture": "unified_pinecone_v2.1"
        }
        
    except Exception as e:
        logger.error(f"Failed to get processing stats from Airtable: {e}")
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
            "processed_at": datetime.now().isoformat(),
            "architecture": "unified_pinecone_v2.1"
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
    """Process uploaded document with LlamaIndex + optional Hyperbolic.ai enhancement + LangExtract + Airtable logging + Unified Pinecone"""
    
    if not document_processor.available:
        raise HTTPException(status_code=503, detail="Document processor not available")
    
    # Read file content
    content = await file.read()
    
    # Process with our enhanced unified pipeline
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
                "processed_at": datetime.now().isoformat(),
                "architecture": "unified_pinecone_v2.1"
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