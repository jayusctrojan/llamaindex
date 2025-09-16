import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from typing import Optional
import json
import httpx
from datetime import datetime
import base64
import subprocess
import tempfile

# LlamaIndex imports with error handling
try:
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.readers.file import PDFReader, DocxReader
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    LLAMAINDEX_AVAILABLE = True
    print("✅ LlamaIndex core components imported successfully")
except ImportError as e:
    print(f"❌ LlamaIndex core not available: {e}")
    LLAMAINDEX_AVAILABLE = False

# Pinecone vector store import with fallback
try:
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    PINECONE_VECTOR_STORE_AVAILABLE = True
    print("✅ LlamaIndex Pinecone vector store imported successfully")
except ImportError as e:
    print(f"⚠️  LlamaIndex Pinecone vector store not available: {e}")
    PINECONE_VECTOR_STORE_AVAILABLE = False

# LangExtract imports
try:
    import langextract as lx
    LANGEXTRACT_AVAILABLE = True
    print("✅ LangExtract imported successfully")
except ImportError as e:
    print(f"⚠️  LangExtract not available: {e}")
    LANGEXTRACT_AVAILABLE = False

# YouTube processing imports
try:
    import yt_dlp
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_AVAILABLE = True
    print("✅ YouTube processing libraries available")
except ImportError as e:
    print(f"⚠️  YouTube processing not available: {e}")
    YOUTUBE_AVAILABLE = False

# Web scraping imports
try:
    from bs4 import BeautifulSoup
    import newspaper3k
    WEB_SCRAPING_AVAILABLE = True
    print("✅ Web scraping libraries available") 
except ImportError as e:
    print(f"⚠️  Web scraping not available: {e}")
    WEB_SCRAPING_AVAILABLE = False

# Airtable imports
try:
    import requests
    AIRTABLE_AVAILABLE = True
    print("✅ Airtable integration ready")
except ImportError as e:
    print(f"⚠️  Could not import requests for Airtable: {e}")
    AIRTABLE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Empire - LlamaIndex + Organizational Intelligence", version="2.2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (web interface)
if os.path.exists("web-interface"):
    app.mount("/static", StaticFiles(directory="web-interface"), name="static")

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

# n8n webhook URL (configured during setup)
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "https://jb-n8n.onrender.com/webhook/content-upload")

# Soniox configuration for audio transcription
SONIOX_API_KEY = os.getenv("SONIOX_API_KEY")
SONIOX_BASE_URL = "https://api.soniox.com/v1"

# Pydantic models for the new endpoints
class ContentUploadRequest(BaseModel):
    contentType: str  # 'url' or 'file'
    url: Optional[str] = None
    filename: Optional[str] = None
    fileContent: Optional[str] = None  # base64 encoded
    mimeType: Optional[str] = None
    fileSize: Optional[int] = None
    course: str = "General"
    module: str = "Unknown"
    uploadedBy: str = "html_interface"

class YouTubeProcessRequest(BaseModel):
    url: str
    extract_images: bool = True
    extract_transcript: bool = True
    include_metadata: bool = True
    output_format: str = "markdown"
    force_high_quality: bool = False  # New option for Soniox fallback

class ArticleProcessRequest(BaseModel):
    url: str
    extract_images: bool = True
    extract_text: bool = True
    output_format: str = "markdown"

# Simplified Pinecone setup with robust error handling
index = None
vector_store = None
PINECONE_AVAILABLE = False

try:
    from pinecone import Pinecone
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Only create vector store if LlamaIndex vector store is available
        if PINECONE_VECTOR_STORE_AVAILABLE:
            vector_store = PineconeVectorStore(
                pinecone_index=index,
                namespace=PINECONE_NAMESPACE,
                add_sparse_vector=False
            )
            print(f"✅ Connected to unified Pinecone index: {PINECONE_INDEX_NAME}")
            print(f"✅ Using namespace: {PINECONE_NAMESPACE}")
        else:
            print("⚠️  Using basic Pinecone connection without LlamaIndex integration")
            
        PINECONE_AVAILABLE = True
    else:
        print("⚠️  Pinecone API key not configured")
except ImportError as e:
    print(f"❌ Pinecone not available: {e}")
except Exception as e:
    print(f"⚠️  Pinecone connection failed: {e}")

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
        print("✅ Backblaze B2 initialized successfully")
    else:
        bucket = None
        print("⚠️  Backblaze B2 not configured")
except ImportError as e:
    print(f"⚠️  B2SDK not available: {e}")
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
            print("⚠️  Requests library not available, Airtable logging disabled")
            return
            
        try:
            if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID:
                print("⚠️  Airtable credentials missing, processing logging disabled")
                return
                
            self.api_key = AIRTABLE_API_KEY
            self.base_id = AIRTABLE_BASE_ID
            self.base_url = f"{AIRTABLE_BASE_URL}/{self.base_id}"
            self.available = True
            print("✅ Airtable document processing logger initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize Airtable client: {e}")
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
            
            print(f"✅ Logged processing start for {filename} (Session: {session_id}, Record: {record_id})")
            return f"{session_id}|{record_id}"  # Return both session and record ID
            
        except Exception as e:
            print(f"❌ Failed to log processing start to Airtable: {e}")
            return None
    
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
            
            return {
                "total_processed": total,
                "successful": successful,
                "failed": total - successful,
                "success_rate": round((successful / total * 100), 2) if total > 0 else 0,
                "pinecone_index": PINECONE_INDEX_NAME,
                "pinecone_namespace": PINECONE_NAMESPACE
            }
            
        except Exception as e:
            print(f"❌ Failed to get processing stats from Airtable: {e}")
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
    """Simple document processing with graceful degradation"""
    
    def __init__(self):
        self.available = LLAMAINDEX_AVAILABLE
        
        if not LLAMAINDEX_AVAILABLE:
            logger.warning("LlamaIndex not available, document processing disabled")
            return
            
        # Initialize basic components
        try:
            if LLAMAINDEX_AVAILABLE:
                self.pdf_reader = PDFReader()
                self.docx_reader = DocxReader()
                self.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
        except Exception as e:
            logger.error(f"Failed to initialize document readers: {e}")
            self.available = False
            return
            
        # Initialize optional components
        self.hyperbolic_llm = None
        if HYPERBOLIC_API_KEY:
            try:
                self.hyperbolic_llm = HyperbolicLLM(HYPERBOLIC_API_KEY)
            except Exception as e:
                logger.warning(f"Failed to initialize Hyperbolic LLM: {e}")
            
        self.lang_extractor = None
        if LANGEXTRACT_AVAILABLE and HYPERBOLIC_API_KEY:
            try:
                self.lang_extractor = self._setup_langextract()
            except Exception as e:
                logger.warning(f"Failed to initialize LangExtract: {e}")
        
        # Initialize Lakera Guard
        self.lakera_guard = LakeraGuard(LAKERA_API_KEY) if LAKERA_API_KEY else None
        if self.lakera_guard and self.lakera_guard.available:
            logger.info("✅ Lakera Guard security enabled")
        
        # Initialize Airtable logger
        self.airtable_logger = AirtableDocumentLogger()
        
        # Set up OpenAI for embeddings
        if os.getenv("OPENAI_API_KEY"):
            try:
                Settings.embed_model = OpenAIEmbedding()
                print("✅ OpenAI embeddings configured")
            except Exception as e:
                logger.warning(f"Failed to configure OpenAI embeddings: {e}")
        
        logger.info("✅ Document processor initialized")
    
    def _setup_langextract(self):
        """Configure LangExtract to use Hyperbolic.ai as the LLM backend"""
        try:
            extractor = lx.LLMExtractor(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                base_url=HYPERBOLIC_BASE_URL,
                api_key=HYPERBOLIC_API_KEY
            )
            return extractor
        except Exception as e:
            logger.error(f"LangExtract setup failed: {e}")
            return None

    async def process_document_basic(self, file_path: str, content: bytes, filename: str, 
                                   user_id: str = "system", course_name: str = "", 
                                   module_name: str = "") -> dict:
        """Basic document processing with graceful degradation"""
        if not self.available:
            raise HTTPException(status_code=503, detail="Document processor not available")
        
        processing_start_time = datetime.now()
        session_record_id = None
        
        try:
            # Log processing start
            if self.airtable_logger.available:
                session_record_id = await self.airtable_logger.log_processing_start(
                    filename=filename,
                    file_size=len(content),
                    user_id=user_id,
                    course_name=course_name,
                    module_name=module_name
                )
            
            # Basic text extraction
            if filename.endswith('.pdf'):
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
                text = content.decode('utf-8')
                documents = [Document(text=text)]
            
            # Parse into chunks
            nodes = self.node_parser.get_nodes_from_documents(documents)
            
            processed_chunks = []
            for i, node in enumerate(nodes):
                chunk_data = {
                    "text": node.text,
                    "metadata": {
                        **node.metadata,
                        "source_system": "ai_empire_pipeline",
                        "processor": "llamaindex",
                        "document_id": f"{filename}_{i}",
                        "filename": filename,
                        "chunk_index": i,
                        "total_chunks": len(nodes),
                        "course": course_name,
                        "module": module_name,
                        "processing_timestamp": datetime.now().isoformat(),
                        "pipeline_version": "ai_empire_v2.2_three_tier_transcription"
                    },
                    "chunk_id": node.node_id,
                    "source": filename,
                    "chunk_index": i
                }
                processed_chunks.append(chunk_data)
            
            # Store in Pinecone if available
            if vector_store and PINECONE_AVAILABLE:
                try:
                    # Update node metadata
                    for i, node in enumerate(nodes):
                        node.metadata.update(processed_chunks[i]["metadata"])
                    
                    # Create index from nodes
                    index_from_nodes = VectorStoreIndex(nodes, vector_store=vector_store)
                    logger.info(f"✅ Stored {len(nodes)} chunks in unified Pinecone index")
                except Exception as e:
                    logger.warning(f"Failed to store in Pinecone: {e}")
            elif index and PINECONE_AVAILABLE:
                # Fallback: store directly in Pinecone without LlamaIndex integration
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    
                    vectors_to_upsert = []
                    for i, chunk in enumerate(processed_chunks):
                        # Generate embedding
                        embedding_response = client.embeddings.create(
                            model="text-embedding-3-small",
                            input=chunk["text"]
                        )
                        embedding = embedding_response.data[0].embedding
                        
                        vectors_to_upsert.append({
                            "id": f"{filename}_{i}_{int(datetime.now().timestamp())}",
                            "values": embedding,
                            "metadata": chunk["metadata"]
                        })
                    
                    # Upsert to Pinecone
                    index.upsert(vectors=vectors_to_upsert, namespace=PINECONE_NAMESPACE)
                    logger.info(f"✅ Stored {len(vectors_to_upsert)} vectors directly in Pinecone")
                except Exception as e:
                    logger.warning(f"Failed to store directly in Pinecone: {e}")
            
            processing_time = (datetime.now() - processing_start_time).total_seconds()
            
            return {
                "status": "success",
                "filename": filename,
                "total_chunks": len(processed_chunks),
                "chunks": processed_chunks,
                "processing_summary": {
                    "llamaindex_chunks": len(nodes),
                    "pinecone_stored": len(nodes) if (vector_store or index) and PINECONE_AVAILABLE else 0
                },
                "session_id": session_record_id.split("|")[0] if session_record_id else None,
                "processing_time_seconds": processing_time,
                "processed_at": datetime.now().isoformat(),
                "pinecone_index": PINECONE_INDEX_NAME if PINECONE_AVAILABLE else "not_configured",
                "pinecone_namespace": PINECONE_NAMESPACE if PINECONE_AVAILABLE else "not_configured",
                "pipeline_version": "ai_empire_v2.2_three_tier_transcription"
            }
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Processing error: {e}")

class YouTubeProcessor:
    """Process YouTube URLs with three-tier transcription system"""
    
    def __init__(self):
        self.available = YOUTUBE_AVAILABLE
        
    async def _call_youtube_mcp_server(self, video_id: str) -> str:
        """Call YouTube transcription MCP server as fallback"""
        try:
            # Try @kimtaeyoon83/mcp-server-youtube-transcript via npx
            # This simulates what an MCP server call would look like
            logger.info(f"🔄 Attempting YouTube MCP server transcription for {video_id}")
            
            # In a real implementation, this would be an MCP server call
            # For now, we'll simulate it by trying to use a local installation
            cmd = [
                "npx", "-y", "@kimtaeyoon83/mcp-server-youtube-transcript",
                "--url", f"https://www.youtube.com/watch?v={video_id}",
                "--lang", "en"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Parse the MCP server response
                try:
                    mcp_response = json.loads(result.stdout)
                    transcript = mcp_response.get("transcript", "")
                    if transcript and len(transcript.strip()) > 50:
                        logger.info(f"✅ YouTube MCP server provided transcript ({len(transcript)} chars)")
                        return transcript
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text transcript
                    if len(result.stdout.strip()) > 50:
                        logger.info(f"✅ YouTube MCP server provided transcript ({len(result.stdout)} chars)")
                        return result.stdout.strip()
            
            logger.warning(f"⚠️ YouTube MCP server failed: {result.stderr}")
            return None
            
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ YouTube MCP server timed out")
            return None
        except Exception as e:
            logger.warning(f"⚠️ YouTube MCP server error: {e}")
            return None
    
    async def _download_and_transcribe_with_soniox(self, url: str, video_id: str) -> str:
        """Download audio and transcribe with Soniox as final fallback"""
        if not SONIOX_API_KEY:
            logger.warning("⚠️ Soniox API key not configured, skipping audio transcription")
            return None
            
        try:
            logger.info(f"🔄 Attempting Soniox transcription for {video_id}")
            
            # Download audio using yt-dlp
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = os.path.join(temp_dir, f"{video_id}.wav")
                
                # yt-dlp command to extract audio
                ydl_opts = {
                    'format': 'bestaudio[ext=m4a]/bestaudio/best',
                    'outtmpl': audio_path.replace('.wav', '.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                # Find the downloaded audio file
                audio_files = [f for f in os.listdir(temp_dir) if f.startswith(video_id)]
                if not audio_files:
                    logger.warning("⚠️ No audio file downloaded")
                    return None
                
                audio_file_path = os.path.join(temp_dir, audio_files[0])
                
                # Convert to wav if needed using ffmpeg
                if not audio_file_path.endswith('.wav'):
                    wav_path = audio_path
                    subprocess.run([
                        'ffmpeg', '-i', audio_file_path, '-acodec', 'pcm_s16le', 
                        '-ar', '16000', '-ac', '1', wav_path, '-y'
                    ], check=True, capture_output=True)
                    audio_file_path = wav_path
                
                # Read audio file
                with open(audio_file_path, 'rb') as f:
                    audio_data = f.read()
                
                # Call Soniox API
                headers = {
                    "Authorization": f"Bearer {SONIOX_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                # Upload audio and get transcript
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                
                payload = {
                    "audio": {
                        "data": audio_b64
                    },
                    "model": "nova-2-general",
                    "include_nonfinal": False,
                    "enable_speaker_diarization": True,
                    "enable_punctuation": True,
                    "languages": ["en"]
                }
                
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(
                        f"{SONIOX_BASE_URL}/transcribe",
                        headers=headers,
                        json=payload
                    )
                    response.raise_for_status()
                    
                    soniox_result = response.json()
                    transcript = soniox_result.get("transcript", "")
                    
                    if transcript and len(transcript.strip()) > 50:
                        logger.info(f"✅ Soniox provided high-quality transcript ({len(transcript)} chars)")
                        return transcript
                    else:
                        logger.warning("⚠️ Soniox returned empty or short transcript")
                        return None
                        
        except Exception as e:
            logger.warning(f"⚠️ Soniox transcription failed: {e}")
            return None

    async def process_youtube_url(self, url: str, extract_images: bool = True, 
                                extract_transcript: bool = True, 
                                include_metadata: bool = True,
                                force_high_quality: bool = False) -> dict:
        """Process YouTube URL with three-tier transcription system"""
        if not self.available:
            raise HTTPException(status_code=503, detail="YouTube processing not available")
        
        try:
            # Extract video ID from URL
            if "youtu.be/" in url:
                video_id = url.split("youtu.be/")[1].split("?")[0]
            elif "youtube.com/watch?v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
            else:
                raise HTTPException(status_code=400, detail="Invalid YouTube URL")
            
            # Get video metadata
            ydl_opts = {
                'quiet': True,
                'no_warnings': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            
            # THREE-TIER TRANSCRIPTION SYSTEM
            transcript_text = ""
            transcription_method = "none"
            transcription_quality = "unknown"
            
            if extract_transcript and not force_high_quality:
                # TIER 1: Try YouTube's built-in captions (fastest, free)
                try:
                    logger.info(f"🔄 Tier 1: Attempting YouTube built-in captions for {video_id}")
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    transcript_text = "\n".join([t['text'] for t in transcript])
                    
                    if transcript_text and len(transcript_text.strip()) > 50:
                        transcription_method = "youtube_captions"
                        transcription_quality = "good"
                        logger.info(f"✅ Tier 1: YouTube captions successful ({len(transcript_text)} chars)")
                    else:
                        transcript_text = ""
                        
                except Exception as e:
                    logger.info(f"⚠️ Tier 1: YouTube captions failed: {e}")
                
                # TIER 2: Try YouTube transcription MCP server (moderate speed, better quality)
                if not transcript_text:
                    logger.info("🔄 Tier 2: Attempting YouTube MCP server transcription")
                    mcp_transcript = await self._call_youtube_mcp_server(video_id)
                    
                    if mcp_transcript and len(mcp_transcript.strip()) > 50:
                        transcript_text = mcp_transcript
                        transcription_method = "youtube_mcp_server"
                        transcription_quality = "better"
                        logger.info(f"✅ Tier 2: YouTube MCP server successful ({len(transcript_text)} chars)")
            
            # TIER 3: Try Soniox high-quality transcription (slowest, best quality)
            if (not transcript_text and extract_transcript) or force_high_quality:
                logger.info("🔄 Tier 3: Attempting Soniox high-quality transcription")
                soniox_transcript = await self._download_and_transcribe_with_soniox(url, video_id)
                
                if soniox_transcript and len(soniox_transcript.strip()) > 50:
                    transcript_text = soniox_transcript
                    transcription_method = "soniox_audio_transcription"
                    transcription_quality = "best"
                    logger.info(f"✅ Tier 3: Soniox transcription successful ({len(transcript_text)} chars)")
                elif not transcript_text:
                    # Final fallback message
                    transcript_text = "Transcript not available - all transcription methods failed"
                    transcription_method = "failed"
                    transcription_quality = "none"
                    logger.warning("⚠️ All transcription tiers failed")
            
            # Create enhanced markdown content with transcription details
            markdown_content = f"""# {info.get('title', 'YouTube Video')}

**Channel:** {info.get('uploader', 'Unknown')}
**Duration:** {info.get('duration', 0)} seconds
**Upload Date:** {info.get('upload_date', 'Unknown')}
**View Count:** {info.get('view_count', 0):,}
**URL:** {url}

**Transcription Method:** {transcription_method}
**Transcription Quality:** {transcription_quality}

## Description
{info.get('description', 'No description available')[:500]}...

## Transcript
{transcript_text}

---
*Processed by AI Empire YouTube Processor v2.2 - Three-Tier Transcription*
"""
            
            return {
                "status": "success",
                "video_id": video_id,
                "title": info.get('title', 'YouTube Video'),
                "channel": info.get('uploader', 'Unknown'),
                "duration": info.get('duration', 0),
                "view_count": info.get('view_count', 0),
                "markdown_content": markdown_content,
                "transcript": transcript_text,
                "transcription_details": {
                    "method": transcription_method,
                    "quality": transcription_quality,
                    "length": len(transcript_text),
                    "available": bool(transcript_text and transcription_method != "failed")
                },
                "metadata": {
                    "source_type": "youtube_video",
                    "processor": "youtube_three_tier_v2.2", 
                    "processing_timestamp": datetime.now().isoformat(),
                    "transcription_system": "three_tier_fallback"
                }
            }
            
        except Exception as e:
            logger.error(f"YouTube processing error: {e}")
            raise HTTPException(status_code=500, detail=f"YouTube processing error: {e}")

class ArticleProcessor:
    """Process article URLs to extract clean markdown content"""
    
    def __init__(self):
        self.available = WEB_SCRAPING_AVAILABLE
        
    async def process_article_url(self, url: str, extract_images: bool = True, 
                                extract_text: bool = True) -> dict:
        """Process article URL and return markdown content"""
        if not self.available:
            raise HTTPException(status_code=503, detail="Article processing not available")
        
        try:
            # Use newspaper3k for article extraction
            from newspaper import Article
            
            article = Article(url)
            article.download()
            article.parse()
            
            # Create markdown content
            markdown_content = f"""# {article.title or 'Article'}

**Source:** {url}
**Authors:** {', '.join(article.authors) if article.authors else 'Unknown'}
**Publish Date:** {article.publish_date or 'Unknown'}

## Summary
{article.summary or 'No summary available'}

## Content
{article.text}

---
*Processed by AI Empire Article Processor*
"""
            
            return {
                "status": "success",
                "title": article.title or "Article",
                "authors": article.authors,
                "publish_date": str(article.publish_date) if article.publish_date else None,
                "markdown_content": markdown_content,
                "text_content": article.text,
                "summary": article.summary,
                "metadata": {
                    "source_type": "web_article",
                    "processor": "article_workflow",
                    "processing_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Article processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Article processing error: {e}")

# Initialize processors
document_processor = DocumentProcessor()
youtube_processor = YouTubeProcessor()
article_processor = ArticleProcessor()

# NEW ENDPOINTS FOR AI EMPIRE WORKFLOW

@app.get("/", response_class=HTMLResponse)
async def serve_web_interface():
    """Serve the main web interface"""
    try:
        with open("web-interface/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Web interface not found</h1><p>Please ensure web-interface/index.html exists.</p>")

@app.get("/upload", response_class=HTMLResponse)
async def upload_interface():
    """Alternative route for upload interface"""
    return await serve_web_interface()

@app.post("/content-upload")
async def content_upload_endpoint(request: ContentUploadRequest):
    """Main endpoint for content upload from HTML interface"""
    
    try:
        # Generate processing ID
        processing_id = f"ai_empire_{int(datetime.now().timestamp())}"
        
        # Log the upload
        logger.info(f"Content upload received: {request.contentType} - {request.filename or request.url}")
        
        # For now, forward to n8n webhook if configured
        if N8N_WEBHOOK_URL:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        N8N_WEBHOOK_URL,
                        json=request.dict(),
                        timeout=30.0
                    )
                    logger.info(f"Forwarded to n8n: {response.status_code}")
            except Exception as e:
                logger.warning(f"Failed to forward to n8n: {e}")
        
        # Return immediate response
        return {
            "status": "received",
            "processing_id": processing_id,
            "timestamp": datetime.now().isoformat(),
            "content_type": request.contentType,
            "filename": request.filename,
            "url": request.url,
            "course": request.course,
            "module": request.module,
            "message": "Content received and queued for processing",
            "transcription_system": "three_tier_fallback_enabled"
        }
        
    except Exception as e:
        logger.error(f"Content upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload error: {e}")

@app.post("/process-youtube")
async def process_youtube_endpoint(request: YouTubeProcessRequest):
    """Process YouTube URL with three-tier transcription system"""
    
    try:
        result = await youtube_processor.process_youtube_url(
            url=request.url,
            extract_images=request.extract_images,
            extract_transcript=request.extract_transcript,
            include_metadata=request.include_metadata,
            force_high_quality=request.force_high_quality
        )
        
        # If Backblaze is available, save to youtube-content folder
        if B2_AVAILABLE and bucket:
            try:
                filename = f"youtube_{result['video_id']}_{result['transcription_details']['method']}_{int(datetime.now().timestamp())}.md"
                
                # Upload to youtube-content folder
                bucket.upload_bytes(
                    data_bytes=result['markdown_content'].encode('utf-8'),
                    file_name=f"youtube-content/{filename}",
                    content_type="text/markdown"
                )
                logger.info(f"YouTube content saved to B2: youtube-content/{filename}")
                
                result["backblaze_file"] = f"youtube-content/{filename}"
            except Exception as e:
                logger.warning(f"Failed to save YouTube content to B2: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"YouTube processing error: {e}")
        raise HTTPException(status_code=500, detail=f"YouTube processing error: {e}")

@app.post("/process-article") 
async def process_article_endpoint(request: ArticleProcessRequest):
    """Process article URL and return clean markdown content"""
    
    try:
        result = await article_processor.process_article_url(
            url=request.url,
            extract_images=request.extract_images,
            extract_text=request.extract_text
        )
        
        # If Backblaze is available, save to youtube-content folder
        if B2_AVAILABLE and bucket:
            try:
                # Create safe filename from title
                safe_title = "".join(c for c in result['title'][:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = f"article_{safe_title}_{int(datetime.now().timestamp())}.md"
                
                # Upload to youtube-content folder (reusing for articles)
                bucket.upload_bytes(
                    data_bytes=result['markdown_content'].encode('utf-8'),
                    file_name=f"youtube-content/{filename}",
                    content_type="text/markdown"
                )
                logger.info(f"Article content saved to B2: youtube-content/{filename}")
                
                result["backblaze_file"] = f"youtube-content/{filename}"
            except Exception as e:
                logger.warning(f"Failed to save article content to B2: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Article processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Article processing error: {e}")

@app.post("/process-content")
async def process_content_unified(request: Request):
    """Unified content processing endpoint for n8n workflow"""
    
    try:
        # Parse JSON request
        content_data = await request.json()
        
        content = content_data.get("content", "")
        metadata = content_data.get("metadata", {})
        processing_options = content_data.get("processing_options", {})
        
        # Create document from content
        document = Document(text=content)
        
        # Process with LlamaIndex
        if document_processor.available:
            nodes = document_processor.node_parser.get_nodes_from_documents([document])
            
            # Generate embeddings if OpenAI key is available
            embeddings = []
            if os.getenv("OPENAI_API_KEY"):
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    
                    for node in nodes:
                        embedding_response = client.embeddings.create(
                            model="text-embedding-3-small",
                            input=node.text
                        )
                        embeddings.append(embedding_response.data[0].embedding)
                except Exception as e:
                    logger.warning(f"Failed to generate embeddings: {e}")
            
            # Prepare response
            chunks = []
            for i, node in enumerate(nodes):
                chunk_data = {
                    "text": node.text,
                    "metadata": {
                        **node.metadata,
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(nodes)
                    }
                }
                chunks.append(chunk_data)
            
            return {
                "status": "success",
                "original_content": content,
                "chunks": chunks,
                "embeddings": embeddings,
                "summary": content[:500] + "..." if len(content) > 500 else content,
                "keywords": [],  # Could add keyword extraction
                "entities": [],  # Could add entity extraction
                "topics": [],
                "questions": [],
                "processing_time_ms": 0,
                "pipeline_version": "ai_empire_v2.2_three_tier_transcription"
            }
        else:
            raise HTTPException(status_code=503, detail="Document processor not available")
            
    except Exception as e:
        logger.error(f"Unified content processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

# EXISTING ENDPOINTS (kept for backward compatibility)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.2.0",
        "architecture": "ai_empire_unified_three_tier_transcription",
        "components": {
            "llamaindex": LLAMAINDEX_AVAILABLE,
            "pinecone": PINECONE_AVAILABLE,
            "pinecone_vector_store": PINECONE_VECTOR_STORE_AVAILABLE,
            "langextract": LANGEXTRACT_AVAILABLE,
            "hyperbolic": bool(HYPERBOLIC_API_KEY),
            "backblaze": B2_AVAILABLE,
            "lakera_guard": bool(LAKERA_API_KEY),
            "airtable": AIRTABLE_AVAILABLE and bool(AIRTABLE_API_KEY and AIRTABLE_BASE_ID),
            "youtube_processing": YOUTUBE_AVAILABLE,
            "web_scraping": WEB_SCRAPING_AVAILABLE,
            "document_processor": document_processor.available,
            "soniox_transcription": bool(SONIOX_API_KEY)
        },
        "transcription_tiers": {
            "tier_1": "youtube_captions",
            "tier_2": "youtube_mcp_server", 
            "tier_3": "soniox_audio_transcription",
            "soniox_available": bool(SONIOX_API_KEY)
        },
        "pinecone_config": {
            "index": PINECONE_INDEX_NAME,
            "namespace": PINECONE_NAMESPACE,
            "connected": PINECONE_AVAILABLE,
            "vector_store_integration": PINECONE_VECTOR_STORE_AVAILABLE
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
    """Process course material with unified Pinecone storage (basic version)"""
    
    if not document_processor.available:
        raise HTTPException(status_code=503, detail="Document processor not available")
    
    try:
        content = await file.read()
        
        # Basic processing with unified Pinecone
        result = await document_processor.process_document_basic(
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
            "user_id": user_id
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Course processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Course processing error: {e}")

@app.post("/search-knowledge")
async def search_course_knowledge(
    query: str = Form(...),
    department_filter: str = Form(default=""),
    limit: int = Form(default=10)
):
    """Search your course knowledge base using unified Pinecone index"""
    
    if not PINECONE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Pinecone not configured")
    
    try:
        # Generate embedding for the query using OpenAI
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = embedding_response.data[0].embedding
        
        # Build filter
        filter_conditions = {}
        if department_filter:
            filter_conditions["department"] = department_filter
        
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
                    "course_name": match.metadata.get("course", ""),
                    "module_name": match.metadata.get("module", "")
                },
                "metadata": match.metadata
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
            "architecture": "unified_three_tier_v2.2"
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
            "architecture": "ai_empire_unified_three_tier_v2.2",
            "pinecone_stats": {
                "index_name": PINECONE_INDEX_NAME,
                "namespace": PINECONE_NAMESPACE,
                "total_vectors_in_index": stats.total_vector_count,
                "vectors_in_namespace": namespace_stats.get('vector_count', 0),
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            },
            "components_status": {
                "llamaindex": LLAMAINDEX_AVAILABLE,
                "pinecone": PINECONE_AVAILABLE,
                "vector_store_integration": PINECONE_VECTOR_STORE_AVAILABLE
            },
            "transcription_system": {
                "version": "three_tier_fallback",
                "tiers": ["youtube_captions", "youtube_mcp_server", "soniox_audio"]
            }
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
            "architecture": "ai_empire_unified_three_tier_v2.2"
        }
        
    except Exception as e:
        logger.error(f"Failed to get processing stats from Airtable: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {e}")

@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    enhance_with_ai: bool = Form(default=True),
    extract_structured: bool = Form(default=True)
):
    """Process uploaded document with basic unified pipeline"""
    
    if not document_processor.available:
        raise HTTPException(status_code=503, detail="Document processor not available")
    
    # Read file content
    content = await file.read()
    
    # Process with basic unified pipeline
    result = await document_processor.process_document_basic(
        file_path=file.filename,
        content=content,
        filename=file.filename
    )
    
    return result

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)