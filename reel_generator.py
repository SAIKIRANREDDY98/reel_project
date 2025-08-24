#!/usr/bin/env python3
"""
AI Reel Generator — PRODUCTION READY VERSION WITH SUBTITLE FIX
- AssemblyAI transcription with guards
- Shotstack render (9:16 vertical) with FIXED subtitles
- Cloudinary upload with retries and validation
- JSONBin update with URL verification
- iOS-ready with all fixes applied

FRAMING MODES:
- LETTERBOX (default): Shows full video frame with black bars (fit="contain")
- FILL: Crops to fill entire screen, TikTok/Reels style (fit="cover")

Setup:
1. Copy .env.template to .env and fill in your API keys
2. Update iOS configs (Info.plist and capacitor.config.json)
3. Run: python reel_generator_production.py
"""

import os
import time
import re
import requests
import cloudinary
import cloudinary.uploader
from datetime import datetime
from difflib import SequenceMatcher
from dotenv import load_dotenv
import random
import json

# Shotstack SDK 0.2.8
from shotstack_sdk.api import edit_api
from shotstack_sdk.configuration import Configuration
from shotstack_sdk.api_client import ApiClient
from shotstack_sdk.exceptions import ApiException

from shotstack_sdk.model.edit import Edit
from shotstack_sdk.model.output import Output
from shotstack_sdk.model.timeline import Timeline
from shotstack_sdk.model.track import Track
from shotstack_sdk.model.clip import Clip
from shotstack_sdk.model.video_asset import VideoAsset
from shotstack_sdk.model.title_asset import TitleAsset
from shotstack_sdk.model.html_asset import HtmlAsset  # Added for alternative subtitle method
from shotstack_sdk.model.offset import Offset  # FIXED: Import Offset model

# ====================
# CONFIGURATION
# ====================
load_dotenv()

# API Keys
SHOTSTACK_API_KEY     = os.getenv("SHOTSTACK_API_KEY")
ASSEMBLYAI_API_KEY    = os.getenv("ASSEMBLYAI_API_KEY")
JSONBIN_API_KEY       = os.getenv("JSONBIN_API_KEY")  # Required for PUT/writes
JSONBIN_BIN_ID        = os.getenv("JSONBIN_BIN_ID", "688ef7c3f7e7a370d1f2a12a")

# Cloudinary
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY    = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Source video
VIDEO_URL = os.getenv("VIDEO_URL", "https://storage.googleapis.com/my-reels-bucket/friends_episode.mp4")

# Dialogues to find (fuzzy match)
DIALOGUES = [
    "telling you that girl winked",
    "did not wink you",
    "chandler ross robert",
]

# Render settings
SIMILARITY_THRESHOLD    = 0.70
OUTPUT_ASPECT_RATIO     = "9:16"     # Vertical for mobile
OUTPUT_RESOLUTION       = "1080"     # 1080x1920
SUBTITLE_SIZE_NAME      = "x-large"  # Options: xx-small, x-small, small, medium, large, x-large, xx-large
SUBTITLE_STYLE          = "subtitle" # Options: subtitle, minimal, blockbuster, vogue, sketchy, skinny
SUBTITLE_METHOD         = "TITLE"     # Options: "TITLE" (TitleAsset) or "HTML" (HtmlAsset)
POLL_SECONDS_SHOTSTACK  = 5
POLL_SECONDS_TRANSCRIPT = 5
RENDER_TIMEOUT_SEC      = 8 * 60     # 8 minutes max

# Framing mode
#   "LETTERBOX" = full frame visible with black bars (no crop)
#   "FILL"      = crop to fill screen (TikTok/Reels style)
FRAMING_MODE = os.getenv("FRAMING_MODE", "LETTERBOX").upper()

# Cloudinary upload settings
CLOUD_CHUNK_SIZE    = 4_000_000      # 4MB chunks
CLOUD_TIMEOUT_SEC   = 600            # 10 minutes
CLOUD_RETRY_MAX     = 4              # Max retry attempts
STABLE_PUBLIC_IDS   = os.getenv("STABLE_PUBLIC_IDS", "false").lower() == "true"

# URL validation
VERIFY_URLS_BEFORE_INDEX = True      # HEAD check before JSONBin save

# Debug settings
DEBUG_SHOTSTACK_PAYLOAD = os.getenv("DEBUG_SHOTSTACK_PAYLOAD", "false").lower() == "true"

# ====================
# Cloudinary Setup
# ====================
if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET
    )

# ====================
# Helper Functions
# ====================
def clean_text(text: str) -> str:
    """Remove punctuation and lowercase for matching"""
    return re.sub(r"[^\w\s]", "", text).lower().strip()

def calculate_similarity(a, b) -> float:
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def validate_video_url(url: str) -> bool:
    """
    Check if video URL is accessible
    Accepts video/*, application/octet-stream, and mpegurl
    """
    try:
        r = requests.get(
            url, 
            stream=True, 
            timeout=20, 
            allow_redirects=True, 
            headers={"Range": "bytes=0-0"}
        )
        if r.status_code not in (200, 206):
            print(f"   ⚠️ URL returned status {r.status_code}")
            return False
        
        ctype = r.headers.get("Content-Type", "").lower()
        # Accept video/*, application/octet-stream, or mpegurl
        valid_types = ["video", "octet-stream", "mpegurl", "mp4"]
        
        if any(vtype in ctype for vtype in valid_types):
            print(f"   ✓ Content-Type: {ctype}")
            return True
        else:
            print(f"   ⚠️ Unexpected Content-Type: {ctype}")
            print("   (Will proceed anyway - may still work)")
            return True  # Proceed anyway for cloud storage edge cases
            
    except requests.RequestException as e:
        print(f"   ⚠️ Connection error: {str(e)[:100]}")
        return False

def find_dialogue_timestamps_fuzzy(transcript, dialogue_text):
    """Find dialogue with fuzzy matching"""
    if not transcript:
        print("ERROR: No transcript provided")
        return None
    
    # Guard for words array
    if "words" not in transcript or not transcript["words"]:
        print("ERROR: Transcript missing 'words' array (needed for timestamps)")
        print("   This can happen with very short audio or API limits")
        return None
    
    words = transcript["words"]
    q = clean_text(dialogue_text).split()
    if not q:
        return None

    print(f"\n🔍 Searching for: '{dialogue_text}'")
    best = None
    best_score = 0.0

    min_window = max(1, int(len(q) * 0.5))
    max_window = min(len(words), int(len(q) * 1.5) + 2)

    for w in range(min_window, max_window + 1):
        for i in range(len(words) - w + 1):
            win = [clean_text(t["text"]) for t in words[i:i+w]]
            score = calculate_similarity(" ".join(q), " ".join(win))
            if score > best_score:
                best_score = score
                best = (i, i + w - 1)

    if best and best_score >= SIMILARITY_THRESHOLD:
        s, e = best
        start_time = words[s]["start"] / 1000.0
        end_time = words[e]["end"] / 1000.0
        actual = " ".join([t["text"] for t in words[s:e+1]])
        print(f"✅ Match (score {best_score:.2f}) {start_time:.2f}s - {end_time:.2f}s")
        print(f"   Actual text: \"{actual}\"")
        return start_time, end_time, actual

    print(f"❌ No match found (best score: {best_score:.2f} < threshold: {SIMILARITY_THRESHOLD})")
    return None

def transcribe_video():
    """Transcribe video using AssemblyAI"""
    print(f"\n🎤 Starting transcription with AssemblyAI...")
    print(f"   Video URL: {VIDEO_URL}")
    
    headers = {
        "authorization": ASSEMBLYAI_API_KEY, 
        "content-type": "application/json"
    }
    payload = {
        "audio_url": VIDEO_URL, 
        "punctuate": True, 
        "format_text": True, 
        "speaker_labels": False, 
        "filter_profanity": False,
        "word_boost": [],
        "boost_param": "high"
    }
    
    try:
        print("   Submitting transcription request...")
        r = requests.post(
            "https://api.assemblyai.com/v2/transcript", 
            json=payload, 
            headers=headers, 
            timeout=30
        )
        r.raise_for_status()
        tid = r.json()["id"]
        print(f"   Transcription ID: {tid}")
    except Exception as e:
        print(f"❌ Transcription request failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"   Response: {e.response.text[:500]}")
        return None

    # Poll for completion
    t0 = time.time()
    while True:
        try:
            g = requests.get(
                f"https://api.assemblyai.com/v2/transcript/{tid}", 
                headers=headers, 
                timeout=15
            )
            g.raise_for_status()
            data = g.json()
            status = data.get("status")
            elapsed = int(time.time() - t0)
            
            if status == "completed":
                print(f"✅ Transcription completed in {elapsed}s")
                
                # Guard for words array
                if not data.get("words"):
                    print("❌ Transcript missing 'words' array (needed for timestamps)")
                    print("   This can happen with very short audio or API limits")
                    return None
                    
                print(f"   Found {len(data['words'])} words in transcript")
                return data
                
            elif status == "error":
                print(f"❌ Transcription error: {data.get('error', 'Unknown error')}")
                return None
            
            print(f"⏳ Transcription status: {status} ({elapsed}s elapsed)")
            time.sleep(POLL_SECONDS_TRANSCRIPT)
            
        except Exception as e:
            print(f"⚠️ Polling error: {e}")
            time.sleep(POLL_SECONDS_TRANSCRIPT)

def render_reel(dialogue, start_time, end_time, reel_index, actual_text=None):
    """
    Render 9:16 vertical video with FIXED subtitles
    - LETTERBOX: Shows full frame with black bars (fit="contain")
    - FILL: Crops to fill screen (fit="cover")
    """
    print(f"\n🎬 Rendering reel {reel_index}: {dialogue}")
    print(f"   Timestamp: {start_time:.2f}s - {end_time:.2f}s")
    print(f"   Subtitle method: {SUBTITLE_METHOD}")
    
    clip_start = max(0.0, float(start_time) - 0.25)
    duration = float(end_time - start_time) + 0.5
    print(f"   Duration: {duration:.2f}s")

    conf = Configuration(
        host="https://api.shotstack.io/v1",
        api_key={"DeveloperKey": SHOTSTACK_API_KEY}
    )

    # Create video asset
    video_asset = VideoAsset(src=VIDEO_URL, trim=clip_start)

    # Use fit property for proper framing (no manual scale needed)
    if FRAMING_MODE == "FILL":
        fit_mode = "cover"    # Scales up to fill, crops overflow
        print("   Mode: FILL (crop to fill screen)")
    else:  # LETTERBOX
        fit_mode = "contain"  # Scales to fit, adds black bars
        print("   Mode: LETTERBOX (full frame with bars)")

    video_clip = Clip(
        asset=video_asset,
        start=0.0,
        length=duration,
        position="center",
        fit=fit_mode,
        scale=1.0
    )

    # Create subtitle
    subtitle_text = (actual_text or dialogue).upper()
    print(f"   Subtitle: \"{subtitle_text[:50]}...\"" if len(subtitle_text) > 50 else f"   Subtitle: \"{subtitle_text}\"")
    
    # Choose subtitle method
    if SUBTITLE_METHOD == "HTML":
        # HTML Asset method (more control, better compatibility)
        html_content = f"""
        <div style="
            color: white;
            font-size: 56px;
            font-weight: bold;
            text-align: center;
            background: rgba(0,0,0,0.8);
            padding: 20px 40px;
            font-family: 'Arial Black', Arial, sans-serif;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.9);
            border-radius: 10px;
            line-height: 1.2;
        ">
        {subtitle_text}
        </div>
        """
        
        subtitle_asset = HtmlAsset(
            html=html_content,
            width=1080,
            height=250,
            background="transparent"
        )
        
        # FIXED: Use Offset object instead of dict
        subtitle_offset = Offset(x=0.0, y=0.15)  # Move up 15% from bottom
        
        subtitle_clip = Clip(
            asset=subtitle_asset,
            start=0.0,
            length=duration,
            position="bottom",
            offset=subtitle_offset,
            opacity=1.0
        )
    else:
        # TitleAsset method (simpler but sometimes problematic)
        title_asset = TitleAsset(
            text=subtitle_text,
            style=SUBTITLE_STYLE,  # "subtitle", "minimal", "blockbuster", etc.
            size=SUBTITLE_SIZE_NAME,  # "x-large"
            color="#ffffff",
            background="#000000CC",  # Semi-transparent black background
            position="center"  # Center text within the asset
        )
        
        # FIXED: Use Offset object instead of dict
        subtitle_offset = Offset(x=0.0, y=0.1)  # Move up 10% from bottom edge
        
        subtitle_clip = Clip(
            asset=title_asset,
            start=0.0,
            length=duration,
            position="bottom",
            offset=subtitle_offset,
            opacity=1.0  # Ensure full opacity
        )
        
        # ALTERNATIVE: If offset still causes issues, remove it entirely:
        # subtitle_clip = Clip(
        #     asset=title_asset,
        #     start=0.0,
        #     length=duration,
        #     position="bottom",  # or try "center"
        #     opacity=1.0
        # )

    # Build timeline with CORRECT track order
    # IMPORTANT: In Shotstack, tracks are layered bottom-to-top
    # First track = bottom layer, Last track = top layer
    timeline = Timeline(
        background="#000000",
        tracks=[
            Track(clips=[video_clip]),     # Track 0: Video (bottom layer)
            Track(clips=[subtitle_clip])   # Track 1: Subtitles (top layer - renders on top)
        ]
    )
    
    # Output settings
    output = Output(
        format="mp4", 
        resolution=OUTPUT_RESOLUTION,  # "1080"
        aspect_ratio=OUTPUT_ASPECT_RATIO  # "9:16"
    )
    
    edit = Edit(timeline=timeline, output=output)

    # Debug: Print the payload if debugging is enabled
    if DEBUG_SHOTSTACK_PAYLOAD:
        edit_dict = edit.to_dict()
        print("\n📋 Shotstack Edit JSON:")
        print(json.dumps(edit_dict, indent=2))

    print("🚀 Submitting to Shotstack API...")
    try:
        with ApiClient(conf) as client:
            api = edit_api.EditApi(client)
            res = api.post_render(edit)
            render_id = res["response"]["id"]
            print(f"   Render ID: {render_id}")
            print(f"   Dashboard: https://dashboard.shotstack.io/renders/{render_id}")
    except ApiException as e:
        print(f"❌ Shotstack API error: {str(e)[:200]}")
        if hasattr(e, 'body'):
            print(f"   Details: {e.body}")
        return None

    # Poll for render completion
    headers = {
        "x-api-key": SHOTSTACK_API_KEY, 
        "Content-Type": "application/json"
    }
    t0 = time.time()
    last_status = None
    
    while time.time() - t0 < RENDER_TIMEOUT_SEC:
        try:
            r = requests.get(
                f"https://api.shotstack.io/v1/render/{render_id}", 
                headers=headers, 
                timeout=20
            )
            r.raise_for_status()
            j = r.json().get("response", {})
            status = j.get("status", "unknown")
            elapsed = int(time.time() - t0)
            
            # Only print if status changed
            if status != last_status:
                print(f"📊 Render status: {status} ({elapsed}s)")
                last_status = status

            if status == "done":
                url = j.get("url")
                if not url:
                    print("❌ No video URL in Shotstack response")
                    return None
                print(f"✅ Render complete! Downloading video...")
                vid = requests.get(url, timeout=180)
                vid.raise_for_status()
                print(f"   Downloaded {len(vid.content) / 1024 / 1024:.2f} MB")
                return vid.content

            elif status in ("failed", "cancelled"):
                error_msg = j.get('error', 'No error details provided')
                print(f"❌ Render failed: {error_msg}")
                return None

            time.sleep(POLL_SECONDS_SHOTSTACK)
            
        except Exception as e:
            print(f"⚠️ Status check error: {str(e)[:100]}")
            time.sleep(POLL_SECONDS_SHOTSTACK)

    print(f"❌ Render timeout after {RENDER_TIMEOUT_SEC/60:.1f} minutes")
    return None

def test_subtitle_only():
    """Test function to render ONLY subtitles without video (for debugging)"""
    print("\n🧪 TESTING SUBTITLE RENDERING (no video)")
    
    conf = Configuration(
        host="https://api.shotstack.io/v1",
        api_key={"DeveloperKey": SHOTSTACK_API_KEY}
    )
    
    test_text = "TEST SUBTITLES ARE WORKING"
    
    if SUBTITLE_METHOD == "HTML":
        html_content = f"""
        <div style="
            color: yellow;
            font-size: 72px;
            font-weight: bold;
            text-align: center;
            background: rgba(255,0,0,0.8);
            padding: 40px;
            font-family: Arial, sans-serif;
        ">
        {test_text}
        </div>
        """
        subtitle_asset = HtmlAsset(
            html=html_content,
            width=1080,
            height=400
        )
    else:
        subtitle_asset = TitleAsset(
            text=test_text,
            style="blockbuster",
            size="xx-large",
            color="#ffff00",
            background="#ff0000"
        )
    
    # FIXED: Don't use offset in test or use Offset object
    subtitle_clip = Clip(
        asset=subtitle_asset,
        start=0.0,
        length=5.0,
        position="center"
        # No offset needed for center position
    )
    
    timeline = Timeline(
        background="#0000ff",  # Blue background to see contrast
        tracks=[Track(clips=[subtitle_clip])]
    )
    
    output = Output(format="mp4", resolution="1080", aspect_ratio="9:16")
    edit = Edit(timeline=timeline, output=output)
    
    with ApiClient(conf) as client:
        api = edit_api.EditApi(client)
        res = api.post_render(edit)
        print(f"Test render ID: {res['response']['id']}")
        print(f"Check: https://dashboard.shotstack.io/renders/{res['response']['id']}")
    
    return res['response']['id']

def upload_to_cloudinary(video_bytes: bytes, filename: str) -> str:
    """Upload video to Cloudinary with retries and validation"""
    print(f"\n☁️ Uploading to Cloudinary: {filename}")
    print(f"   File size: {len(video_bytes) / 1024 / 1024:.2f} MB")
    
    # Save to temp file
    temp = f"/tmp/{filename}"
    try:
        with open(temp, "wb") as f:
            f.write(video_bytes)
        print(f"   Temp file created: {temp}")
    except Exception as e:
        print(f"❌ Failed to write temp file: {e}")
        return None

    size = os.path.getsize(temp)
    use_large = size >= 100 * 1024 * 1024  # Use large upload for >100MB
    
    if use_large:
        print("   Using large file upload (>100MB)")

    for attempt in range(1, CLOUD_RETRY_MAX + 1):
        try:
            print(f"   Upload attempt {attempt}/{CLOUD_RETRY_MAX}...")
            
            if use_large:
                result = cloudinary.uploader.upload_large(
                    temp, 
                    resource_type="video",
                    public_id=filename.replace(".mp4", ""),
                    folder="reels",
                    overwrite=True, 
                    invalidate=True,
                    chunk_size=CLOUD_CHUNK_SIZE, 
                    timeout=CLOUD_TIMEOUT_SEC
                )
            else:
                result = cloudinary.uploader.upload(
                    temp, 
                    resource_type="video",
                    public_id=filename.replace(".mp4", ""),
                    folder="reels",
                    overwrite=True, 
                    invalidate=True,
                    timeout=CLOUD_TIMEOUT_SEC
                )
            
            url = result.get("secure_url")
            if url:
                print(f"✅ Upload successful!")
                print(f"   URL: {url}")
                
                # Verify URL is accessible
                if VERIFY_URLS_BEFORE_INDEX:
                    try:
                        verify_r = requests.head(url, timeout=5)
                        if verify_r.status_code == 200:
                            print("   ✓ URL verified accessible")
                        else:
                            print(f"   ⚠️ URL returned {verify_r.status_code} (may need CDN propagation)")
                    except:
                        print("   ⚠️ Could not verify URL (CDN propagation delay)")
                
                # Cleanup temp file
                try:
                    os.remove(temp)
                except:
                    pass
                    
                return url
            
            print(f"   ⚠️ No secure_url in response")
            
        except Exception as e:
            print(f"   ⚠️ Upload failed: {str(e)[:100]}")
            if attempt < CLOUD_RETRY_MAX:
                backoff = 2 ** (attempt - 1) + random.uniform(0, 0.8)
                print(f"   ⏳ Retrying in {backoff:.1f}s...")
                time.sleep(backoff)

    # Cleanup temp file on failure
    try:
        if os.path.exists(temp):
            os.remove(temp)
    except:
        pass
    
    print("❌ Cloudinary upload failed after all retries")
    return None

def save_reels_to_jsonbin(reel_infos):
    """Update JSONBin with reel index (with URL validation)"""
    if not JSONBIN_BIN_ID:
        print("\n⚠️ No JSONBIN_BIN_ID provided - skipping index save")
        print("   iOS app won't be able to fetch reels without this")
        return False
    
    if not JSONBIN_API_KEY:
        print("\n⚠️ No JSONBIN_API_KEY provided")
        print("   ❌ JSONBin requires X-Master-Key for PUT/write operations")
        print("   Set JSONBIN_API_KEY in your .env file")
        return False
    
    # Validate URLs are accessible before saving
    if VERIFY_URLS_BEFORE_INDEX:
        print("\n🔍 Verifying Cloudinary URLs are accessible...")
        for reel in reel_infos:
            try:
                r = requests.head(reel["url"], timeout=5)
                if r.status_code == 200:
                    print(f"   ✓ Reel {reel['id']}: Accessible")
                else:
                    print(f"   ⚠️ Reel {reel['id']}: Status {r.status_code} (may still work)")
            except Exception as e:
                print(f"   ⚠️ Reel {reel['id']}: Could not verify (CDN delay?)")
    
    # Prepare document
    doc = {
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "reels": reel_infos,
        "total_count": len(reel_infos),
        "storage_type": "cloudinary",
        "framing_mode": FRAMING_MODE,
        "subtitle_method": SUBTITLE_METHOD,
        "version": "7.0"
    }
    
    url = f"https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}"
    headers = {
        "Content-Type": "application/json",
        "X-Master-Key": JSONBIN_API_KEY  # Required for PUT
    }

    print(f"\n📤 Updating JSONBin index...")
    print(f"   Bin ID: {JSONBIN_BIN_ID}")
    print(f"   Reels: {len(reel_infos)}")
    
    try:
        resp = requests.put(url, headers=headers, json=doc, timeout=30)
        resp.raise_for_status()
        print("✅ JSONBin index updated successfully!")
        print(f"   Public URL: https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}/latest")
        return True
    except requests.HTTPError as e:
        if e.response.status_code == 401:
            print("❌ JSONBin authentication failed")
            print("   Make sure JSONBIN_API_KEY is correct in .env")
        else:
            print(f"❌ JSONBin HTTP error: {e.response.status_code}")
            print(f"   Response: {e.response.text[:200]}")
        return False
    except Exception as e:
        print(f"❌ JSONBin save failed: {str(e)[:200]}")
        return False

def validate_environment():
    """Check all required environment variables"""
    print("\n🔧 Validating environment...")
    
    required = {
        "SHOTSTACK_API_KEY": SHOTSTACK_API_KEY,
        "ASSEMBLYAI_API_KEY": ASSEMBLYAI_API_KEY,
        "CLOUDINARY_CLOUD_NAME": CLOUDINARY_CLOUD_NAME,
        "CLOUDINARY_API_KEY": CLOUDINARY_API_KEY,
        "CLOUDINARY_API_SECRET": CLOUDINARY_API_SECRET
    }
    
    optional = {
        "JSONBIN_API_KEY": JSONBIN_API_KEY,
        "JSONBIN_BIN_ID": JSONBIN_BIN_ID
    }
    
    missing_required = []
    missing_optional = []
    
    for key, value in required.items():
        if not value:
            missing_required.append(key)
        else:
            print(f"   ✓ {key}: Set ({len(value)} chars)")
    
    for key, value in optional.items():
        if not value:
            missing_optional.append(key)
            print(f"   ⚠️ {key}: Not set (optional but recommended)")
        else:
            print(f"   ✓ {key}: Set")
    
    if missing_required:
        print(f"\n❌ Missing required environment variables:")
        for key in missing_required:
            print(f"   - {key}")
        print("\nPlease set these in your .env file")
        print("See .env.template for examples")
        return False
    
    if missing_optional:
        print(f"\n⚠️ Optional variables not set:")
        for key in missing_optional:
            print(f"   - {key}")
        print("   (Reels will render but iOS app may not work)")
    
    print("\n✅ Environment validation passed")
    return True

# ====================
# Main Function
# ====================
def main():
    print("\n" + "=" * 70)
    print("🎬 AI REEL GENERATOR - PRODUCTION READY (SUBTITLE FIX)")
    print("=" * 70)
    print(f"📱 Output: {OUTPUT_RESOLUTION}p vertical ({OUTPUT_ASPECT_RATIO})")
    print(f"🖼️ Framing: {FRAMING_MODE}")
    if FRAMING_MODE == "LETTERBOX":
        print("   → Full video frame with black bars (no cropping)")
    else:
        print("   → Video cropped to fill screen (TikTok/Reels style)")
    print(f"📝 Subtitles: {SUBTITLE_SIZE_NAME} size, {SUBTITLE_STYLE} style")
    print(f"   Method: {SUBTITLE_METHOD} {'(more compatible)' if SUBTITLE_METHOD == 'HTML' else '(simpler)'}")
    print(f"💾 Public IDs: {'Stable (overwrites)' if STABLE_PUBLIC_IDS else 'Timestamped (unique)'}")
    print(f"🐛 Debug mode: {'ON - will show Shotstack JSON' if DEBUG_SHOTSTACK_PAYLOAD else 'OFF'}")
    print("=" * 70)

    # Validate environment
    if not validate_environment():
        return

    # Optional: Run subtitle test
    if input("\n🧪 Run subtitle-only test first? (y/N): ").lower() == 'y':
        test_id = test_subtitle_only()
        print(f"\n✅ Test render submitted. Check the dashboard link above.")
        print("   Wait for it to complete and verify subtitles are visible.")
        if input("   Continue with main process? (Y/n): ").lower() == 'n':
            return

    # Validate video URL
    print(f"\n🔎 Validating source video...")
    print(f"   URL: {VIDEO_URL}")
    if not validate_video_url(VIDEO_URL):
        print("❌ Video URL validation failed")
        print("   Proceeding anyway - Shotstack may still be able to access it")
    else:
        print("✅ Video URL validated")

    # Transcribe video
    transcript = transcribe_video()
    if not transcript:
        print("\n❌ Transcription failed. Cannot proceed without transcript.")
        return

    # Process dialogues
    total = len(DIALOGUES)
    reel_infos = []
    print(f"\n📊 Starting to process {total} dialogues")
    print("=" * 70)

    for idx, dialogue in enumerate(DIALOGUES, 1):
        print(f"\n🎯 REEL {idx}/{total}")
        print(f"   Target: \"{dialogue}\"")
        print("-" * 50)

        # Find dialogue in transcript
        found = find_dialogue_timestamps_fuzzy(transcript, dialogue)
        if not found:
            print(f"   ⏭️ Skipping: Dialogue not found in transcript")
            print(f"   📊 Progress: {len(reel_infos)}/{total} completed")
            continue

        start, end, actual_text = found

        # Render video
        try:
            video_bytes = render_reel(dialogue, start, end, idx, actual_text)
        except Exception as e:
            print(f"   ❌ Render error: {str(e)[:200]}")
            print(f"   📊 Progress: {len(reel_infos)}/{total} completed")
            continue

        if not video_bytes:
            print(f"   ❌ Render failed")
            print(f"   📊 Progress: {len(reel_infos)}/{total} completed")
            continue

        # Generate filename
        if STABLE_PUBLIC_IDS:
            filename = f"reel_{idx}.mp4"
        else:
            filename = f"reel_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        # Upload to Cloudinary
        public_url = upload_to_cloudinary(video_bytes, filename)
        if not public_url:
            print(f"   ❌ Upload failed")
            print(f"   📊 Progress: {len(reel_infos)}/{total} completed")
            continue

        # Add to index
        reel_info = {
            "id": idx,
            "title": f"Reel #{idx}",
            "subtitle": actual_text or dialogue,
            "url": public_url,
            "timestamp_range": f"{start:.2f}s - {end:.2f}s",
            "duration_seconds": round((end - start) + 0.5, 2),
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        reel_infos.append(reel_info)
        
        print(f"\n   ✅ REEL {idx} COMPLETE!")
        print(f"   📊 Progress: {len(reel_infos)}/{total} completed")
        print("-" * 50)

    # Final summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    if not reel_infos:
        print("❌ No reels were created successfully")
        print("\nPossible issues:")
        print("  - Dialogues not found in transcript")
        print("  - Render failures")
        print("  - Upload failures")
        print("\n💡 TIP: Check if subtitles are rendering by running the test first")
        return

    print(f"✅ Successfully created {len(reel_infos)}/{total} reels")
    print("\n📝 Subtitle configuration used:")
    print(f"   Method: {SUBTITLE_METHOD}")
    print(f"   Style: {SUBTITLE_STYLE}")
    print(f"   Size: {SUBTITLE_SIZE_NAME}")
    
    # Save to JSONBin
    if save_reels_to_jsonbin(reel_infos):
        print("\n" + "🎉" * 35)
        print("🎉 SUCCESS! ALL SYSTEMS OPERATIONAL!")
        print("🎉" * 35)
        print(f"\n✅ {len(reel_infos)} reels created and indexed")
        print(f"✅ Videos hosted on Cloudinary (permanent URLs)")
        print(f"✅ JSONBin index updated")
        print(f"✅ Subtitles configured with {SUBTITLE_METHOD} method")
        
        print("\n📱 NEXT STEPS FOR iOS APP:")
        print("   1. Make sure iOS configs are updated:")
        print("      - Info.plist (ATS exceptions)")
        print("      - capacitor.config.json (allowNavigation)")
        print("   2. Run: npx cap sync ios")
        print("   3. Open Xcode: npx cap open ios")
        print("   4. Run in simulator")
        print("   5. Click ↻ Retry button in app")
        
        print(f"\n🔗 Resources:")
        print(f"   JSONBin: https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}/latest")
        print(f"   Cloudinary: https://cloudinary.com/console/media_library/folders/reels")
        print(f"   Shotstack: https://dashboard.shotstack.io/renders")
    else:
        print("\n⚠️ PARTIAL SUCCESS")
        print(f"   ✅ {len(reel_infos)} reels created on Cloudinary")
        print("   ❌ JSONBin index update failed")
        print("\n   Your videos are ready but iOS app won't see them")
        print("   Check JSONBIN_API_KEY in .env")

    print("\n" + "=" * 70)
    print("✨ PROCESS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
