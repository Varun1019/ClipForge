# ClipForge 🎬⚡

**ClipForge** is an open-source, AI-powered automatic clipping application designed to analyze long podcasts and livestreams and extract engaging, standalone short video clips with optional synchronized subtitles.

ClipForge is built on a **Bring-Your-Own-Key (BYOK)** architecture. Users connect their own Google Gemini API credentials, ensuring complete control over model selection, privacy, and quotas without middleman markups or mandatory platform subscriptions.

---

## 📌 Project Status

**Current Status**: Working Prototype (v0.5.0)

- **Video Ingestion & Local Inspection**: Operational (MP4, MOV, MKV via FFmpeg/ffprobe).
- **Manual Clipping Engine**: Operational (Local FFmpeg sub-clipping with timestamp bounds validation).
- **BYOK AI Architecture**: Operational (Provider-independent abstractions, structured JSON schemas).
- **Dynamic Model Discovery**: Operational (Queries available models via Google Gen AI SDK).
- **AI Highlight Detection & Automatic Batch Clipping**: Operational (Multimodal analysis with Google Gemini, candidate validation, and local batch FFmpeg rendering).
- **Tested Model Configuration**: **Gemini 3.6 Flash** is the verified working model in our current environment.
- **Optional Automatic Subtitles**: Operational (Local Faster-Whisper speech-to-text, SRT export, and FFmpeg hardsub burning).
- **Vertical & Square Video Reframing**: Operational (9:16 Shorts/Reels/TikTok and 1:1 Feed export with Center Crop or Blurred Background modes).
- **Cloud & Production Ready**: Operational (Per-session workspace isolation, automated TTL cleanup, containerized Dockerfile, and health probes).

---

## 💻 Installation & Quickstart

### Prerequisites
- **Linux, macOS, or Windows**
- **Python 3.12** (managed locally via `uv` or system Python)
- **FFmpeg & ffprobe** installed with `libass` and accessible in your system `PATH`

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/ClipForge.git
cd ClipForge

# Create isolated Python 3.12 virtual environment using uv
uv venv .venv --python 3.12

# Activate virtual environment
source .venv/bin/activate       # Bash / Zsh
# source .venv/bin/activate.fish # Fish shell
```

### 2. Install Dependencies
```bash
uv pip install -r requirements.txt
```

### 3. Launch the Studio Interface
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Automated Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py"
```
*(All 82 unit tests run offline without network access or real API charges).*

### 5. Run Performance Benchmarks
```bash
python benchmarks/run_baseline.py
```
*(Measures local metadata caching, clipping, reframing, subtitle rendering, Faster-Whisper, and peak RAM).*

---

## 📱 Vertical & Square Video Export (9:16 & 1:1)

ClipForge provides built-in FFmpeg reframing for vertical social formats (YouTube Shorts, Instagram Reels, TikTok) and square feed posts (Instagram, LinkedIn).

### Aspect Ratio Options
1. **Original Aspect Ratio (Default)**: Preserves the source camera ratio without scaling or cropping.
2. **Vertical 9:16**: Converts landscape footage to vertical canvas:
   - **1080x1920 (Full HD)**
   - **720x1280 (HD)**
3. **Square 1:1**: Converts footage to square canvas:
   - **1080x1080 (Full HD)**
   - **720x720 (Standard)**

### Framing Modes
- **Blurred Background (Fit Frame)**: Scales the entire original video frame so nothing is cut off or distorted, centering it over a stylized blurred background fill created from the footage.
- **Center Crop (Fill Canvas)**: Scales the video to completely fill the canvas, cropping excess horizontal borders while keeping the center focused.

### Synchronization & Subtitles
- **Zero Distortion**: Footage is scaled proportionally using FFmpeg filtergraphs.
- **Audio Preservation**: Audio streams are copied (`-c:a copy`) with zero transcoding latency or drift.
- **Integrated Subtitles**: When subtitles are burned into reframed clips, captions are rendered cleanly on the reframed canvas.
- **Clean Video Intact**: The original aspect ratio clip is **always preserved** alongside reframed files.

---

## 💬 Optional Automatic Subtitles

ClipForge includes local, hardware-accelerated speech-to-text transcription for generated clips.

### How It Works
1. **Per-Clip Transcription**: Transcription runs on the **individual exported clip**, never on the entire source video. Word and segment timestamps start natively at `00:00:00,000` for seamless caption sync.
2. **Local & Private Engine**: Powered by **Faster-Whisper (CTranslate2)**. Runs entirely on your machine. Audio is never sent to third-party transcription APIs.
3. **Hardware Considerations**:
   - **NVIDIA GPU (CUDA float16)**: Automatically accelerated on NVIDIA RTX/GTX GPUs with 4GB+ VRAM (e.g. RTX 3050 Laptop GPU).
   - **CPU Fallback (int8)**: If CUDA libraries (e.g. `libcublas.so.12`) are absent, ClipForge automatically detects this and falls back to quantized 8-bit CPU inference in seconds.
4. **Export Outputs**:
   - **Clean MP4**: The un-subtitled original clip is **always preserved** in `exports/`.
   - **SRT Subtitle File**: Saved alongside the clip as `exports/{clip_name}.srt`.
   - **Burned-In MP4 (Optional)**: If enabled, FFmpeg burns readable subtitles into `exports/{clip_name}_subtitled.mp4`.
5. **In-App Previews**: The UI allows toggling between the clean clip and the subtitled clip, downloading either MP4, downloading the SRT file, and reviewing the transcribed dialogue text.
6. **No-Speech Handling**: If a clip contains no detectable speech (e.g. silence or background music), ClipForge logs an informational notice, preserves the clean clip, and gracefully skips subtitle burn-in.

---

## 🔑 Gemini Setup & Model Selection

### 1. Obtain an API Key
1. Visit [Google AI Studio](https://aistudio.google.com/apikey).
2. Sign in with your Google account.
3. Click **Create API key** and copy your generated credential.
4. ClipForge accepts all standard Google credential formats (including `AIza...` and newer `AQ...` formats).

### 2. Secure Credential Handling
- **Held in memory only**: Your API key exists in session memory while the tab is open.
- **Never persisted**: ClipForge never logs, prints, or saves your API key to disk, cookies, `.env`, or Git.
- **Session Caching**: Key hashes are validated once and cached per session to prevent redundant network calls across Streamlit widget reruns.

### 3. Dynamic Model Discovery & Selecting Gemini 3.6 Flash
1. Enter your API key in the **Gemini API Key** field.
2. ClipForge immediately queries Google's `client.models.list()` endpoint and filters models that support:
   - Content generation (`generateContent`)
   - Multimodal video and audio understanding
3. In the **Gemini Model** dropdown, select **`gemini-3.6-flash`**.
4. Use the **🔄 Refresh** button anytime to re-query Google for an updated model catalog.

> [!IMPORTANT]
> **Tested Model Verification**:
> In our current setup, **Gemini 3.6 Flash** (`gemini-3.6-flash`) has been verified and tested successfully for end-to-end highlight detection.
>
> Model availability and compatibility can vary significantly depending on your specific Google API credentials, account tier, preview enrollment, and geographical region. Other models returned in the dropdown have not yet been verified in our current environment (this does not mean they are universally non-functional, only that they remain unverified under our setup).

---

## 🔒 Media Processing, Privacy & Quotas

- **Local Media Operations**: Video clipping, remuxing, metadata extraction, audio transcription, and subtitle burning are executed **100% locally on your machine via FFmpeg and Faster-Whisper**. Original media files are never modified or overwritten.
- **Remote AI Processing**: When you click **Find Clips**, media (either the video container or an optimized 16kHz audio track depending on your selected strategy) is transmitted securely over TLS to Google's Gen AI Files API.
- **Automatic Cloud Cleanup**: As soon as Gemini finishes generating structured highlight candidates, ClipForge immediately invokes `client.files.delete()` to purge the uploaded file from Google's servers.
- **User Billing & Quotas**: ClipForge is free, open-source software. You pay for your own API usage directly to Google according to Google's pricing terms and free-tier quotas. Users are responsible for managing their own quotas and billing limits in the Google Cloud / AI Studio console.

---

## 🚀 Public Cloud & Production Deployment

ClipForge is architected for cloud deployment as a multi-user Streamlit web application. It includes per-session workspace isolation, automatic TTL garbage collection, health check endpoints, and container configurations.

### Multi-Tenant Architecture & Privacy
- **Session Isolation**: Every active browser tab is assigned a cryptographically random session workspace (`temp/<session_id>/` and `exports/<session_id>/`). Users never see, access, or overwrite other users' media or exported clips.
- **Automated TTL Garbage Collection**: Background cleanup automatically purges session folders older than 1 hour while preserving cached Faster-Whisper weights in `temp/models/`.
- **Memory-Only BYOK Secrets**: API keys exist strictly within individual session state and are never written to disk, server logs, or shared memory.

### Option 1: Hugging Face Spaces (Recommended Free Tier)
Hugging Face Spaces provides a **free 16 GB RAM / 2 vCPU tier**, ideal for handling Faster-Whisper transcription and FFmpeg video encoding without out-of-memory crashes.

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Select **Streamlit** (or **Docker**) as the Space SDK.
3. Push the ClipForge repository to your Space repository:
   - When using the **Streamlit SDK**, Hugging Face automatically installs system FFmpeg from `packages.txt`.
   - When using the **Docker SDK**, Hugging Face automatically builds the provided multi-stage `Dockerfile`.
4. The application will build and launch on port 7860/8501 automatically.

### Option 2: Docker Container / Self-Hosted VPS
ClipForge includes a production-ready, non-root `Dockerfile` based on `python:3.12-slim-bookworm` with system `ffmpeg` pre-configured.

```bash
# Build the production Docker image
docker build -t clipforge .

# Run the container
docker run -d \
  -p 8501:8501 \
  --name clipforge-app \
  --restart unless-stopped \
  clipforge
```
Access the application at `http://your-server-ip:8501`. Container health can be checked at `http://your-server-ip:8501/_stcore/health`.

### Option 3: Streamlit Community Cloud
1. Fork or push the ClipForge repository to your GitHub account.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) and create a new app.
3. Select your repository, branch `master`, and main file `app.py`.
4. Streamlit Cloud automatically detects `packages.txt` to install system `ffmpeg`.
5. *(Note: Streamlit Community Cloud free tier provides ~1 GB RAM. For large video files or long clips, Hugging Face Spaces or a VPS is strongly recommended to avoid container OOM restarts during Whisper transcription).*

---

## 📋 Features & Roadmap

### Current Features
- [x] **Video Upload & Inspection**: Supports MP4, MOV, and MKV files up to local disk capacity.
- [x] **Local Metadata Display**: Instant ffprobe readout (dimensions, duration, codecs, size).
- [x] **Manual Clipping**: Precise start/end timestamp selection with duration preview and validation.
- [x] **BYOK Gemini Integration**: Dynamic model discovery, secure key input, and quota error reporting.
- [x] **Smart Media Modality**: Automatic lightweight audio extraction for large files (>30MB / >5min) to save bandwidth and upload time.
- [x] **Structured AI Highlight Moments**: Extracts title, timestamps, opening spoken hook, explanation, and confidence score.
- [x] **Highlight Selection & Batch Rendering**: Review AI suggestions, select individual moments, and batch render standalone MP4 clips into `exports/`.
- [x] **Optional Subtitle Generation**: Local Faster-Whisper transcription generating aligned `.srt` files.
- [x] **Optional Subtitle Burn-In**: Local FFmpeg hardsubbing with readable styling while keeping clean video untouched.
- [x] **Vertical (9:16) & Square (1:1) Framing**: Full HD / HD outputs with Blurred Background or Center Crop modes.
- [x] **In-App Video Preview & Downloads**: Watch clean, reframed, or subtitled clips directly in the browser and download MP4 or SRT with one click.
- [x] **Session Isolation & Cloud Hardening**: Per-user ephemeral workspaces, automated TTL cleanup, and CSRF protection.

### Current Limitations
- **Center-Focused Framing**: Vertical framing uses center crop or blurred background. Multi-person AI active face tracking / auto-pan is not yet implemented.
- **Single Active AI Highlight Provider**: Google Gemini is currently the active cloud highlight detection provider.

### Planned Features
- [ ] **AI Active Speaker Tracking**: Multi-camera face detection and dynamic auto-panning across active podcast speakers.
- [ ] **Animated Karaoke Word-by-Word Captions**: Styled dynamic captions generated from word timestamps.
- [ ] **Additional Providers**: Adapters for OpenAI, Anthropic, and local LLMs (Ollama).

---

## 🛠️ Troubleshooting & Common Issues

| Issue | Cause | Resolution |
|---|---|---|
| **"Authentication Failed: timed out"** | Network connectivity issue or slow proxy/firewall | Verify internet access. ClipForge enforces a 10s timeout on live authentication. |
| **"Invalid API key or unauthorized credentials"** | Key was mistyped or revoked by Google | Verify and copy your API key from [Google AI Studio](https://aistudio.google.com/apikey). Ensure no extra whitespace. |
| **"Rate limit or quota exhausted"** | Google API free-tier RPM/RPD or daily token quota reached | Check your usage in Google AI Studio console or wait for quota reset. |
| **"Model not found or not enabled"** | The selected model is not available for your account tier or region | Click **🔄 Refresh** and select **`gemini-3.6-flash`** (or another model enabled on your account). |
| **"FFmpeg / ffprobe is unavailable"** | FFmpeg binaries are not found in system `PATH` | Install FFmpeg via your package manager (`sudo pacman -S ffmpeg`, `brew install ffmpeg`, or `apt install ffmpeg`). |
| **"No detectable speech found in clip"** | Audio track has low volume, background music, or no dialogue | The clean video clip is preserved without subtitles. |

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
