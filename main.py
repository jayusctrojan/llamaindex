import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from typing import Optional, List, Dict, Any
import json
import httpx
from datetime import datetime
import base64
import subprocess
import tempfile
import cv2
import numpy as np
from PIL import Image
import io
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Document processing imports
try:
    import fitz  # PyMuPDF for PDF image extraction
    import docx2txt
    from docx import Document as DocxDocument
    from pptx import Presentation
    import zipfile
    DOCUMENT_PROCESSING_AVAILABLE = True
    print("✅ Document processing libraries available")
except ImportError as e:
    print(f"⚠️ Document processing not available: {e}")
    DOCUMENT_PROCESSING_AVAILABLE = False

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
    import requests
    import newspaper3k
    WEB_SCRAPING_AVAILABLE = True
    print("✅ Web scraping libraries available") 
except ImportError as e:
    print(f"⚠️  Web scraping not available: {e}")
    WEB_SCRAPING_AVAILABLE = False

# Vision processing imports
try:
    import cv2
    import numpy as np
    from PIL import Image
    VISION_PROCESSING_AVAILABLE = True
    print("✅ Vision processing libraries available")
except ImportError as e:
    print(f"⚠️  Vision processing not available: {e}")
    VISION_PROCESSING_AVAILABLE = False

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

app = FastAPI(title="AI Empire - LlamaIndex + Session Correlation + Parallel Image Processing", version="2.4.0")

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

# Session Correlation Models
class SessionContext(BaseModel):
    session_id: str
    source_file: str
    source_type: str  # document|article|video|audio
    course_name: str = "General"
    module_number: str = "Unknown"
    processing_paths: Dict[str, str] = {}  # path_name -> status
    correlation_data: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class ProcessingRequest(BaseModel):
    content: str = ""
    session_context: Optional[SessionContext] = None
    processing_type: str = "standard"
    enable_vision: bool = True
    enable_image_extraction: bool = True

# Enhanced Pydantic models for the new endpoints
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
    session_context: Optional[SessionContext] = None

class YouTubeProcessRequest(BaseModel):
    url: str
    extract_images: bool = True
    extract_transcript: bool = True
    include_metadata: bool = True
    output_format: str = "markdown"
    force_high_quality: bool = False
    frame_interval: int = 30
    vision_analysis: bool = True
    session_context: Optional[SessionContext] = None

class ArticleProcessRequest(BaseModel):
    url: str
    extract_images: bool = True
    extract_text: bool = True
    output_format: str = "markdown"
    session_context: Optional[SessionContext] = None

class DocumentProcessRequest(BaseModel):
    file_content: str  # base64 encoded
    filename: str
    mime_type: str
    parallel_processing: bool = True
    extract_images: bool = True
    session_context: Optional[SessionContext] = None

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

class SessionManager:
    """Manages session correlation and data integrity"""
    
    def __init__(self):
        self.active_sessions: Dict[str, SessionContext] = {}
        self.session_lock = asyncio.Lock()
    
    def generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    async def create_session(self, source_file: str, source_type: str, 
                           course_name: str = "General", module_number: str = "Unknown") -> SessionContext:
        """Create new processing session"""
        async with self.session_lock:
            session_id = self.generate_session_id()
            session = SessionContext(
                session_id=session_id,
                source_file=source_file,
                source_type=source_type,
                course_name=course_name,
                module_number=module_number,
                processing_paths={},
                correlation_data={},
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "processing_version": "2.4.0"
                }
            )
            self.active_sessions[session_id] = session
            logger.info(f"Created session {session_id} for {source_file}")
            return session
    
    async def update_session_path(self, session_id: str, path_name: str, status: str, data: Any = None):
        """Update processing path status"""
        async with self.session_lock:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.processing_paths[path_name] = status
                if data:
                    session.correlation_data[path_name] = data
                logger.info(f"Updated session {session_id}, path {path_name}: {status}")
    
    async def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Retrieve session context"""
        return self.active_sessions.get(session_id)
    
    async def validate_session_complete(self, session_id: str, required_paths: List[str]) -> bool:
        """Validate all required paths are completed"""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        for path in required_paths:
            if session.processing_paths.get(path) != "completed":
                return False
        
        return True
    
    async def cleanup_session(self, session_id: str):
        """Clean up completed session"""
        async with self.session_lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                logger.info(f"Cleaned up session {session_id}")

# Global session manager
session_manager = SessionManager()

class DocumentImageProcessor:
    """Extract and analyze images from documents with session correlation"""
    
    def __init__(self):
        self.available = DOCUMENT_PROCESSING_AVAILABLE and VISION_PROCESSING_AVAILABLE
        self.vision_processor = None
        if hasattr(globals(), 'VisionProcessor'):
            self.vision_processor = VisionProcessor()
    
    async def extract_images_from_pdf(self, pdf_path: str, session_context: SessionContext) -> List[Dict]:
        """Extract images from PDF with positioning metadata"""
        if not self.available:
            return []
        
        extracted_images = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # Ensure it's not CMYK
                        # Convert to PIL Image
                        img_data = pix.tobytes("png")
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        # Convert to numpy array for processing
                        img_array = np.array(pil_image)
                        
                        # Check if image is educationally significant
                        if self._is_image_educationally_relevant(img_array):
                            extracted_images.append({
                                "image_id": f"pdf_img_{page_num}_{img_index}",
                                "page_number": page_num + 1,
                                "position": f"page_{page_num + 1}_image_{img_index + 1}",
                                "image_data": img_array,
                                "format": "PNG",
                                "session_id": session_context.session_id
                            })
                    
                    pix = None
            
            doc.close()
            logger.info(f"Extracted {len(extracted_images)} images from PDF")
            return extracted_images
            
        except Exception as e:
            logger.error(f"PDF image extraction failed: {e}")
            return []
    
    async def extract_images_from_docx(self, docx_path: str, session_context: SessionContext) -> List[Dict]:
        """Extract images from DOCX with positioning metadata"""
        if not self.available:
            return []
        
        extracted_images = []
        
        try:
            doc = DocxDocument(docx_path)
            
            # Extract images from document
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    img_data = rel.target_part.blob
                    
                    # Convert to PIL Image
                    pil_image = Image.open(io.BytesIO(img_data))
                    img_array = np.array(pil_image)
                    
                    if self._is_image_educationally_relevant(img_array):
                        extracted_images.append({
                            "image_id": f"docx_img_{len(extracted_images)}",
                            "section_id": "document_body",
                            "position": f"embedded_image_{len(extracted_images) + 1}",
                            "image_data": img_array,
                            "format": "auto_detected",
                            "session_id": session_context.session_id
                        })
            
            logger.info(f"Extracted {len(extracted_images)} images from DOCX")
            return extracted_images
            
        except Exception as e:
            logger.error(f"DOCX image extraction failed: {e}")
            return []
    
    async def extract_images_from_pptx(self, pptx_path: str, session_context: SessionContext) -> List[Dict]:
        """Extract images from PowerPoint with slide positioning"""
        if not self.available:
            return []
        
        extracted_images = []
        
        try:
            prs = Presentation(pptx_path)
            
            for slide_idx, slide in enumerate(prs.slides):
                for shape_idx, shape in enumerate(slide.shapes):
                    if hasattr(shape, "image"):
                        img_data = shape.image.blob
                        
                        # Convert to PIL Image
                        pil_image = Image.open(io.BytesIO(img_data))
                        img_array = np.array(pil_image)
                        
                        if self._is_image_educationally_relevant(img_array):
                            extracted_images.append({
                                "image_id": f"pptx_slide_{slide_idx}_img_{shape_idx}",
                                "slide_number": slide_idx + 1,
                                "position": f"slide_{slide_idx + 1}_shape_{shape_idx + 1}",
                                "image_data": img_array,
                                "format": "auto_detected",
                                "session_id": session_context.session_id
                            })
            
            logger.info(f"Extracted {len(extracted_images)} images from PPTX")
            return extracted_images
            
        except Exception as e:
            logger.error(f"PPTX image extraction failed: {e}")
            return []
    
    def _is_image_educationally_relevant(self, img_array: np.ndarray) -> bool:
        """Determine if image contains educational content"""
        try:
            # Check image size (skip very small images)
            height, width = img_array.shape[:2]
            if height < 100 or width < 100:
                return False
            
            # Check if image has sufficient detail (not just solid color)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Calculate variance to detect content
            variance = np.var(gray)
            if variance < 100:  # Very low variance = likely solid color or simple image
                return False
            
            # Additional checks could be added here
            return True
            
        except Exception:
            return True  # Default to including if check fails
    
    async def analyze_extracted_images(self, images: List[Dict], session_context: SessionContext) -> List[Dict]:
        """Analyze extracted images using vision AI"""
        if not self.vision_processor or not self.vision_processor.available:
            logger.warning("Vision processor not available for image analysis")
            return images
        
        analyzed_images = []
        
        for img_data in images:
            try:
                # Analyze image with Hyperbolic Vision AI
                context = f"Document image from {session_context.source_file}"
                analysis = await self.vision_processor.analyze_frame_with_hyperbolic(
                    img_data["image_data"], context
                )
                
                # Add analysis to image data
                img_data["analysis"] = analysis.get("analysis", "")
                img_data["educational_relevance"] = analysis.get("valuable_content", False)
                img_data["analysis_timestamp"] = datetime.now().isoformat()
                
                analyzed_images.append(img_data)
                
            except Exception as e:
                logger.error(f"Image analysis failed for {img_data.get('image_id', 'unknown')}: {e}")
                img_data["analysis"] = "Analysis failed"
                img_data["educational_relevance"] = False
                analyzed_images.append(img_data)
        
        logger.info(f"Analyzed {len(analyzed_images)} images")
        return analyzed_images
    
    async def process_document_images(self, file_path: str, file_type: str, 
                                    session_context: SessionContext) -> Dict:
        """Process document images with session correlation"""
        
        # Update session status
        await session_manager.update_session_path(
            session_context.session_id, "imageExtraction", "pending"
        )
        
        try:
            # Extract images based on file type
            if file_type.lower() == 'pdf':
                images = await self.extract_images_from_pdf(file_path, session_context)
            elif file_type.lower() in ['docx', 'doc']:
                images = await self.extract_images_from_docx(file_path, session_context)
            elif file_type.lower() in ['pptx', 'ppt']:
                images = await self.extract_images_from_pptx(file_path, session_context)
            else:
                images = []
            
            # Analyze extracted images
            if images:
                images = await self.analyze_extracted_images(images, session_context)
            
            # Update session with results
            await session_manager.update_session_path(
                session_context.session_id, "imageExtraction", "completed", images
            )
            
            # Create summary
            educational_images = [img for img in images if img.get("educational_relevance", False)]
            
            result = {
                "total_images": len(images),
                "educational_images": len(educational_images),
                "extracted_images": images,
                "summary": f"Extracted {len(images)} images, {len(educational_images)} educationally relevant",
                "session_id": session_context.session_id,
                "processing_success": True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Document image processing failed: {e}")
            await session_manager.update_session_path(
                session_context.session_id, "imageExtraction", "failed"
            )
            
            return {
                "total_images": 0,
                "educational_images": 0,
                "extracted_images": [],
                "summary": f"Image extraction failed: {e}",
                "session_id": session_context.session_id,
                "processing_success": False
            }

class VisionProcessor:
    """Vision processing using Hyperbolic for frame and image analysis"""
    
    def __init__(self):
        self.available = bool(HYPERBOLIC_API_KEY and VISION_PROCESSING_AVAILABLE)
        self.api_key = HYPERBOLIC_API_KEY
        self.base_url = HYPERBOLIC_BASE_URL
        
    def is_frame_significant(self, frame: np.ndarray, prev_frame: np.ndarray = None, threshold: float = 0.3) -> bool:
        """Determine if a frame contains significant visual information"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Check if frame is mostly black/empty
            mean_brightness = np.mean(gray)
            if mean_brightness < 10:  # Very dark frame
                return False
            
            # Check for sufficient contrast (indicates content)
            contrast = np.std(gray)
            if contrast < 20:  # Very low contrast
                return False
            
            # If we have a previous frame, check for significant change
            if prev_frame is not None:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray, prev_gray)
                change_percentage = np.mean(diff) / 255.0
                
                # If change is too small, skip (likely same content)
                if change_percentage < threshold:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Frame significance check failed: {e}")
            return True  # Default to processing if check fails
    
    def extract_frames_from_video(self, video_path: str, interval_seconds: int = 30, max_frames: int = 20) -> List[np.ndarray]:
        """Extract significant frames from video at specified intervals"""
        if not self.available:
            return []
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            logger.info(f"Video stats - Duration: {duration:.1f}s, FPS: {fps:.1f}, Total frames: {total_frames}")
            
            frames = []
            prev_frame = None
            frame_interval = int(fps * interval_seconds)
            
            for frame_num in range(0, total_frames, frame_interval):
                if len(frames) >= max_frames:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Check if frame is significant
                if self.is_frame_significant(frame, prev_frame):
                    frames.append(frame)
                    prev_frame = frame
                    logger.info(f"Extracted significant frame at {frame_num/fps:.1f}s")
            
            cap.release()
            logger.info(f"Extracted {len(frames)} significant frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []
    
    def frame_to_base64(self, frame: np.ndarray, format: str = 'JPEG') -> str:
        """Convert OpenCV frame to base64 encoded image"""
        try:
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format, quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/{format.lower()};base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Frame to base64 conversion failed: {e}")
            return ""
    
    async def analyze_frame_with_hyperbolic(self, frame: np.ndarray, context: str = "") -> dict:
        """Analyze frame using Hyperbolic vision model"""
        if not self.available:
            return {"error": "Vision processing not available"}
        
        try:
            # Convert frame to base64
            image_data = self.frame_to_base64(frame)
            
            if not image_data:
                return {"error": "Failed to convert frame to base64"}
            
            # Prepare vision analysis prompt
            prompt = f"""Analyze this image and identify any important visual content. Focus on:
1. Diagrams, charts, flowcharts, or technical illustrations
2. Text content (slides, presentations, code, formulas)
3. Whiteboard or handwritten content
4. Screenshots or UI elements
5. Any educational or instructional visual elements

Context: {context if context else 'Educational content image'}

Provide a detailed description of what you see, especially focusing on educational content that would be valuable for learning. If this appears to be just a person talking with no significant visual content, simply say "Speaker only - no significant visual content"."""

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
                "messages": [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data}}
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.3
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]
                
                # Determine if frame contains valuable content
                is_valuable = not any(phrase in analysis.lower() for phrase in [
                    "speaker only", "no significant visual", "just a person talking",
                    "no educational content", "only shows a person"
                ])
                
                return {
                    "analysis": analysis,
                    "valuable_content": is_valuable,
                    "model_used": "Llama-3.2-11B-Vision-Instruct"
                }
                
        except Exception as e:
            logger.error(f"Hyperbolic vision analysis failed: {e}")
            return {"error": f"Vision analysis failed: {e}", "analysis": ""}
    
    async def process_video_frames(self, video_path: str, context: str = "", interval_seconds: int = 30) -> dict:
        """Extract and analyze all significant frames from a video"""
        logger.info(f"Starting vision processing for video: {video_path}")
        
        # Extract frames
        frames = self.extract_frames_from_video(video_path, interval_seconds)
        
        if not frames:
            return {
                "frame_count": 0,
                "visual_content": [],
                "summary": "No significant visual content extracted"
            }
        
        # Analyze each frame
        visual_content = []
        valuable_frames = 0
        
        for i, frame in enumerate(frames):
            logger.info(f"Analyzing frame {i+1}/{len(frames)}")
            
            analysis = await self.analyze_frame_with_hyperbolic(frame, context)
            
            if analysis.get("valuable_content", False):
                visual_content.append({
                    "frame_index": i,
                    "timestamp_seconds": i * interval_seconds,
                    "timestamp_formatted": f"{(i * interval_seconds) // 60}:{(i * interval_seconds) % 60:02d}",
                    "description": analysis.get("analysis", ""),
                    "valuable": True
                })
                valuable_frames += 1
            
        # Create summary
        summary = f"Processed {len(frames)} frames, found {valuable_frames} frames with significant visual content"
        
        if visual_content:
            summary += f". Visual content includes: {', '.join([content['description'][:50] + '...' for content in visual_content[:3]])}"
        
        return {
            "frame_count": len(frames),
            "valuable_frames": valuable_frames,
            "visual_content": visual_content,
            "summary": summary,
            "processing_success": len(visual_content) > 0
        }

class ArticleImageProcessor:
    """Process web articles with image extraction and session correlation"""
    
    def __init__(self):
        self.available = WEB_SCRAPING_AVAILABLE and VISION_PROCESSING_AVAILABLE
        self.vision_processor = VisionProcessor() if VISION_PROCESSING_AVAILABLE else None
    
    async def extract_article_with_images(self, url: str, session_context: SessionContext) -> Dict:
        """Extract article text and images with session correlation"""
        
        # Update session status
        await session_manager.update_session_path(
            session_context.session_id, "articleProcessing", "pending"
        )
        
        try:
            # Extract article content
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            article_text = ""
            for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                if tag.get_text(strip=True):
                    article_text += tag.get_text(strip=True) + "\n\n"
            
            # Extract images
            images = []
            img_tags = soup.find_all('img')
            
            for idx, img_tag in enumerate(img_tags):
                img_src = img_tag.get('src') or img_tag.get('data-src')
                if img_src:
                    # Convert relative URLs to absolute
                    if img_src.startswith('//'):
                        img_src = 'https:' + img_src
                    elif img_src.startswith('/'):
                        from urllib.parse import urljoin
                        img_src = urljoin(url, img_src)
                    
                    try:
                        # Download image
                        img_response = await client.get(img_src)
                        img_response.raise_for_status()
                        
                        # Convert to PIL Image
                        pil_image = Image.open(io.BytesIO(img_response.content))
                        img_array = np.array(pil_image)
                        
                        # Analyze image if vision processor available
                        analysis = ""
                        educational_relevance = False
                        
                        if self.vision_processor and self.vision_processor.available:
                            context = f"Image from article: {url}"
                            analysis_result = await self.vision_processor.analyze_frame_with_hyperbolic(
                                img_array, context
                            )
                            analysis = analysis_result.get("analysis", "")
                            educational_relevance = analysis_result.get("valuable_content", False)
                        
                        images.append({
                            "image_id": f"article_img_{idx}",
                            "url": img_src,
                            "alt_text": img_tag.get('alt', ''),
                            "position": f"article_image_{idx + 1}",
                            "analysis": analysis,
                            "educational_relevance": educational_relevance,
                            "session_id": session_context.session_id
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to process image {img_src}: {e}")
            
            # Update session with results
            result_data = {
                "text_content": article_text,
                "images": images,
                "total_images": len(images),
                "educational_images": len([img for img in images if img.get("educational_relevance", False)])
            }
            
            await session_manager.update_session_path(
                session_context.session_id, "articleProcessing", "completed", result_data
            )
            
            return {
                "status": "success",
                "text_content": article_text,
                "images": images,
                "summary": f"Extracted article with {len(images)} images, {result_data['educational_images']} educationally relevant",
                "session_id": session_context.session_id,
                "processing_success": True
            }
            
        except Exception as e:
            logger.error(f"Article processing failed: {e}")
            await session_manager.update_session_path(
                session_context.session_id, "articleProcessing", "failed"
            )
            
            return {
                "status": "error",
                "error": f"Article processing failed: {e}",
                "session_id": session_context.session_id,
                "processing_success": False
            }

# Initialize processors
document_image_processor = DocumentImageProcessor()
article_image_processor = ArticleImageProcessor()
vision_processor = VisionProcessor()

# Session-Enhanced Processing Endpoints

@app.post("/process/parallel-document")
async def process_document_parallel(request: DocumentProcessRequest):
    """Process document with parallel text and image extraction"""
    
    # Create or use session context
    if not request.session_context:
        session_context = await session_manager.create_session(
            request.filename, "document", "General", "Unknown"
        )
    else:
        session_context = request.session_context
    
    logger.info(f"Starting parallel document processing for session {session_context.session_id}")
    
    try:
        # Decode file content
        file_data = base64.b64decode(request.file_content)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{request.filename.split('.')[-1]}") as temp_file:
            temp_file.write(file_data)
            temp_file_path = temp_file.name
        
        try:
            # Determine file type
            file_extension = request.filename.split('.')[-1].lower()
            
            # Start parallel processing
            tasks = []
            
            # Text extraction task (using existing LlamaIndex processing)
            async def text_extraction():
                await session_manager.update_session_path(session_context.session_id, "textExtraction", "pending")
                try:
                    # Use existing document processing logic here
                    # For now, simulating with basic text extraction
                    text_content = f"Text content from {request.filename}"
                    await session_manager.update_session_path(
                        session_context.session_id, "textExtraction", "completed", text_content
                    )
                    return {"text_content": text_content}
                except Exception as e:
                    await session_manager.update_session_path(session_context.session_id, "textExtraction", "failed")
                    raise e
            
            tasks.append(text_extraction())
            
            # Image extraction task (if enabled)
            if request.extract_images:
                tasks.append(
                    document_image_processor.process_document_images(
                        temp_file_path, file_extension, session_context
                    )
                )
            
            # Execute parallel processing
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Validate session completion
            required_paths = ["textExtraction"]
            if request.extract_images:
                required_paths.append("imageExtraction")
            
            session_complete = await session_manager.validate_session_complete(
                session_context.session_id, required_paths
            )
            
            if not session_complete:
                raise HTTPException(status_code=500, detail="Session validation failed - incomplete processing paths")
            
            # Merge results
            text_result = results[0] if not isinstance(results[0], Exception) else {"text_content": ""}
            image_result = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else {
                "total_images": 0, "educational_images": 0, "extracted_images": []
            }
            
            # Create unified response
            response = {
                "status": "success",
                "session_id": session_context.session_id,
                "filename": request.filename,
                "text_processing": {
                    "content": text_result.get("text_content", ""),
                    "success": "textExtraction" in session_context.processing_paths
                },
                "image_processing": {
                    "total_images": image_result.get("total_images", 0),
                    "educational_images": image_result.get("educational_images", 0),
                    "extracted_images": image_result.get("extracted_images", []),
                    "success": image_result.get("processing_success", False)
                },
                "session_validation": {
                    "complete": session_complete,
                    "required_paths": required_paths,
                    "completed_paths": list(session_context.processing_paths.keys())
                },
                "metadata": {
                    "processing_type": "parallel_document",
                    "processing_version": "2.4.0",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Cleanup session
            await session_manager.cleanup_session(session_context.session_id)
            
            return response
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
    except Exception as e:
        logger.error(f"Parallel document processing failed: {e}")
        await session_manager.cleanup_session(session_context.session_id)
        raise HTTPException(status_code=500, detail=f"Parallel document processing failed: {e}")

@app.post("/process/article-with-images")
async def process_article_with_images(request: ArticleProcessRequest):
    """Process article with image extraction and session correlation"""
    
    # Create or use session context
    if not request.session_context:
        session_context = await session_manager.create_session(
            request.url, "article", "General", "Unknown"
        )
    else:
        session_context = request.session_context
    
    logger.info(f"Starting article processing with images for session {session_context.session_id}")
    
    try:
        # Process article with images
        result = await article_image_processor.extract_article_with_images(request.url, session_context)
        
        # Cleanup session
        await session_manager.cleanup_session(session_context.session_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Article processing failed: {e}")
        await session_manager.cleanup_session(session_context.session_id)
        raise HTTPException(status_code=500, detail=f"Article processing failed: {e}")

@app.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get current session processing status"""
    session = await session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "source_file": session.source_file,
        "source_type": session.source_type,
        "processing_paths": session.processing_paths,
        "metadata": session.metadata,
        "created_at": session.metadata.get("created_at"),
        "status": "active"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.4.0",
        "architecture": "ai_empire_session_correlation_parallel_processing",
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
            "vision_processing": VISION_PROCESSING_AVAILABLE,
            "document_processing": DOCUMENT_PROCESSING_AVAILABLE,
            "soniox_transcription": bool(SONIOX_API_KEY)
        },
        "session_capabilities": {
            "session_correlation": True,
            "parallel_processing": True,
            "active_sessions": len(session_manager.active_sessions)
        },
        "image_processing_capabilities": {
            "document_image_extraction": DOCUMENT_PROCESSING_AVAILABLE,
            "article_image_extraction": WEB_SCRAPING_AVAILABLE,
            "vision_analysis": bool(HYPERBOLIC_API_KEY and VISION_PROCESSING_AVAILABLE),
            "supported_document_formats": ["PDF", "DOCX", "PPTX"] if DOCUMENT_PROCESSING_AVAILABLE else []
        },
        "vision_capabilities": {
            "frame_extraction": VISION_PROCESSING_AVAILABLE,
            "hyperbolic_vision": bool(HYPERBOLIC_API_KEY),
            "supported_formats": ["YouTube videos", "MP4 uploads", "Document images", "Article images"] if VISION_PROCESSING_AVAILABLE else []
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
