import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from typing import Optional, List
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

app = FastAPI(title="AI Empire - LlamaIndex + Vision Processing", version="2.3.0")

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
    force_high_quality: bool = False
    frame_interval: int = 30  # Extract frame every N seconds
    vision_analysis: bool = True  # Use Hyperbolic for vision analysis

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

class VisionProcessor:
    """Vision processing using Hyperbolic for frame analysis"""
    
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
            prompt = f"""Analyze this video frame and identify any important visual content. Focus on:
1. Diagrams, charts, flowcharts, or technical illustrations
2. Text content (slides, presentations, code, formulas)
3. Whiteboard or handwritten content
4. Screenshots or UI elements
5. Any educational or instructional visual elements

Context: {context if context else 'Video frame from course content'}

Provide a detailed description of what you see, especially focusing on educational content that would be valuable for learning. If this appears to be just a person talking with no significant visual content, simply say "Speaker only - no significant visual content"."""

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",  # Hyperbolic vision model
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

# ... [Continue with existing classes - LakeraGuard, AirtableDocumentLogger, HyperbolicLLM, DocumentProcessor] ...

class YouTubeProcessor:
    """Process YouTube URLs with three-tier transcription and vision analysis"""
    
    def __init__(self):
        self.available = YOUTUBE_AVAILABLE
        self.vision_processor = VisionProcessor()
        
    async def _call_youtube_mcp_server(self, video_id: str) -> str:
        """Call YouTube transcription MCP server as fallback"""
        try:
            logger.info(f"🔄 Attempting YouTube MCP server transcription for {video_id}")
            
            cmd = [
                "npx", "-y", "@kimtaeyoon83/mcp-server-youtube-transcript",
                "--url", f"https://www.youtube.com/watch?v={video_id}",
                "--lang", "en"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                try:
                    mcp_response = json.loads(result.stdout)
                    transcript = mcp_response.get("transcript", "")
                    if transcript and len(transcript.strip()) > 50:
                        logger.info(f"✅ YouTube MCP server provided transcript ({len(transcript)} chars)")
                        return transcript
                except json.JSONDecodeError:
                    if len(result.stdout.strip()) > 50:
                        logger.info(f"✅ YouTube MCP server provided transcript ({len(result.stdout)} chars)")
                        return result.stdout.strip()
            
            logger.warning(f"⚠️ YouTube MCP server failed: {result.stderr}")
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
            
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = os.path.join(temp_dir, f"{video_id}.wav")
                
                ydl_opts = {
                    'format': 'bestaudio[ext=m4a]/bestaudio/best',
                    'outtmpl': audio_path.replace('.wav', '.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                audio_files = [f for f in os.listdir(temp_dir) if f.startswith(video_id)]
                if not audio_files:
                    logger.warning("⚠️ No audio file downloaded")
                    return None
                
                audio_file_path = os.path.join(temp_dir, audio_files[0])
                
                if not audio_file_path.endswith('.wav'):
                    wav_path = audio_path
                    subprocess.run([
                        'ffmpeg', '-i', audio_file_path, '-acodec', 'pcm_s16le', 
                        '-ar', '16000', '-ac', '1', wav_path, '-y'
                    ], check=True, capture_output=True)
                    audio_file_path = wav_path
                
                with open(audio_file_path, 'rb') as f:
                    audio_data = f.read()
                
                headers = {
                    "Authorization": f"Bearer {SONIOX_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                
                payload = {
                    "audio": {"data": audio_b64},
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

    async def _extract_video_frames(self, url: str, video_id: str, interval_seconds: int = 30) -> dict:
        """Download video and extract frames for vision analysis"""
        if not self.vision_processor.available:
            return {"frame_count": 0, "visual_content": [], "summary": "Vision processing not available"}
        
        try:
            logger.info(f"🔄 Extracting video frames for vision analysis: {video_id}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = os.path.join(temp_dir, f"{video_id}.mp4")
                
                # Download video using yt-dlp
                ydl_opts = {
                    'format': 'best[height<=720]/best',  # Limit resolution for processing speed
                    'outtmpl': video_path,
                    'quiet': True,
                    'no_warnings': True
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                if not os.path.exists(video_path):
                    logger.warning("⚠️ Video download failed for frame extraction")
                    return {"frame_count": 0, "visual_content": [], "summary": "Video download failed"}
                
                # Process frames with vision analysis
                context = f"YouTube video: {video_id}"
                vision_results = await self.vision_processor.process_video_frames(
                    video_path, context, interval_seconds
                )
                
                return vision_results
                
        except Exception as e:
            logger.warning(f"⚠️ Video frame extraction failed: {e}")
            return {"frame_count": 0, "visual_content": [], "summary": f"Frame extraction failed: {e}"}

    async def process_youtube_url(self, url: str, extract_images: bool = True, 
                                extract_transcript: bool = True, 
                                include_metadata: bool = True,
                                force_high_quality: bool = False,
                                frame_interval: int = 30,
                                vision_analysis: bool = True) -> dict:
        """Process YouTube URL with three-tier transcription and vision analysis"""
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
                
                # TIER 2: Try YouTube transcription MCP server
                if not transcript_text:
                    logger.info("🔄 Tier 2: Attempting YouTube MCP server transcription")
                    mcp_transcript = await self._call_youtube_mcp_server(video_id)
                    
                    if mcp_transcript and len(mcp_transcript.strip()) > 50:
                        transcript_text = mcp_transcript
                        transcription_method = "youtube_mcp_server"
                        transcription_quality = "better"
                        logger.info(f"✅ Tier 2: YouTube MCP server successful ({len(transcript_text)} chars)")
            
            # TIER 3: Try Soniox high-quality transcription
            if (not transcript_text and extract_transcript) or force_high_quality:
                logger.info("🔄 Tier 3: Attempting Soniox high-quality transcription")
                soniox_transcript = await self._download_and_transcribe_with_soniox(url, video_id)
                
                if soniox_transcript and len(soniox_transcript.strip()) > 50:
                    transcript_text = soniox_transcript
                    transcription_method = "soniox_audio_transcription"
                    transcription_quality = "best"
                    logger.info(f"✅ Tier 3: Soniox transcription successful ({len(transcript_text)} chars)")
                elif not transcript_text:
                    transcript_text = "Transcript not available - all transcription methods failed"
                    transcription_method = "failed"
                    transcription_quality = "none"
                    logger.warning("⚠️ All transcription tiers failed")
            
            # VISION ANALYSIS: Extract and analyze video frames
            vision_results = {"frame_count": 0, "visual_content": [], "summary": "Vision analysis disabled"}
            
            if extract_images and vision_analysis and self.vision_processor.available:
                vision_results = await self._extract_video_frames(url, video_id, frame_interval)
            
            # Create enhanced markdown content with transcription and vision details
            vision_section = ""
            if vision_results["frame_count"] > 0:
                vision_section = f"""
## Visual Content Analysis
**Frames Analyzed:** {vision_results["frame_count"]}
**Significant Visual Content:** {vision_results.get("valuable_frames", 0)} frames

### Visual Elements Found:
"""
                for content in vision_results.get("visual_content", []):
                    vision_section += f"- **{content['timestamp_formatted']}**: {content['description']}\n"
            
            markdown_content = f"""# {info.get('title', 'YouTube Video')}

**Channel:** {info.get('uploader', 'Unknown')}
**Duration:** {info.get('duration', 0)} seconds
**Upload Date:** {info.get('upload_date', 'Unknown')}
**View Count:** {info.get('view_count', 0):,}
**URL:** {url}

**Transcription Method:** {transcription_method}
**Transcription Quality:** {transcription_quality}
**Vision Analysis:** {"Enabled" if vision_analysis else "Disabled"}

## Description
{info.get('description', 'No description available')[:500]}...

## Transcript
{transcript_text}
{vision_section}

---
*Processed by AI Empire YouTube Processor v2.3 - Three-Tier Transcription + Vision Analysis*
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
                "visual_analysis": vision_results,
                "transcription_details": {
                    "method": transcription_method,
                    "quality": transcription_quality,
                    "length": len(transcript_text),
                    "available": bool(transcript_text and transcription_method != "failed")
                },
                "vision_details": {
                    "enabled": vision_analysis,
                    "frames_processed": vision_results["frame_count"],
                    "valuable_content_found": vision_results.get("valuable_frames", 0),
                    "summary": vision_results["summary"]
                },
                "metadata": {
                    "source_type": "youtube_video",
                    "processor": "youtube_vision_v2.3", 
                    "processing_timestamp": datetime.now().isoformat(),
                    "transcription_system": "three_tier_fallback",
                    "vision_system": "hyperbolic_llama_vision"
                }
            }
            
        except Exception as e:
            logger.error(f"YouTube processing error: {e}")
            raise HTTPException(status_code=500, detail=f"YouTube processing error: {e}")

# ... [Continue with existing classes and endpoints] ...

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.3.0",
        "architecture": "ai_empire_vision_processing",
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
            "soniox_transcription": bool(SONIOX_API_KEY)
        },
        "vision_capabilities": {
            "frame_extraction": VISION_PROCESSING_AVAILABLE,
            "hyperbolic_vision": bool(HYPERBOLIC_API_KEY),
            "supported_formats": ["YouTube videos", "MP4 uploads"] if VISION_PROCESSING_AVAILABLE else []
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