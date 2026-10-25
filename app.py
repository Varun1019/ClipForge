"""ClipForge - Open-source AI automatic clipping studio (BYOK Google Gemini Integration).

A modern, dark-glassmorphism video editing studio for podcasts, livestreams, and creator media.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid
import streamlit as st

from src.clipper import (
    batch_create_clips,
    create_video_clip,
    validate_timestamps,
    TimestampValidationError,
    VideoClipperError,
)
from src.framing import (
    AspectRatioMode,
    FramingConfig,
    FramingStyle,
    reframe_video,
)
from src.storage import (
    calculate_dir_size_str,
    cleanup_stale_files,
    init_session_workspace,
)
from src.providers.base import (
    HighlightCandidate,
    HighlightRequest,
    HighlightResponse,
    InputModality,
    InvalidAPIKeyError,
    ModelCapability,
    ModelNotFoundError,
    ProviderError,
    QuotaExceededError,
)
from src.providers.registry import provider_registry
from src.video_info import (
    check_ffprobe_available,
    extract_video_metadata,
    format_duration,
    format_file_size,
    FFmpegNotFoundError,
    InvalidVideoError,
)

# Base storage directories
BASE_TEMP_DIR = Path("temp")
BASE_TEMP_DIR.mkdir(exist_ok=True)
BASE_EXPORTS_DIR = Path("exports")
BASE_EXPORTS_DIR.mkdir(exist_ok=True)

# Configure page metadata and layout
st.set_page_config(
    page_title="ClipForge Studio — AI Video Clipper",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize isolated session workspace
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex[:12]
    # Run stale cleanup once on session start
    cleanup_stale_files(base_dirs=[BASE_TEMP_DIR, BASE_EXPORTS_DIR], max_age_seconds=3600)

session_id = st.session_state["session_id"]
TEMP_DIR, EXPORTS_DIR = init_session_workspace(BASE_TEMP_DIR, BASE_EXPORTS_DIR, session_id)

# Refined Dark Glassmorphism Studio Theme CSS
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --cf-bg: #07090E;
        --cf-panel: rgba(15, 19, 32, 0.75);
        --cf-panel-solid: #0F1320;
        --cf-card: rgba(21, 27, 44, 0.65);
        --cf-card-hover: rgba(28, 36, 58, 0.85);
        --cf-border: rgba(255, 255, 255, 0.08);
        --cf-border-subtle: rgba(255, 255, 255, 0.05);
        --cf-border-glow: rgba(99, 102, 241, 0.35);
        --cf-primary: #6366F1;
        --cf-primary-hover: #4F46E5;
        --cf-cyan: #06B6D4;
        --cf-cyan-bright: #22D3EE;
        --cf-emerald: #10B981;
        --cf-amber: #F59E0B;
        --cf-rose: #F43F5E;
        --cf-text-1: #F8FAFC;
        --cf-text-2: #94A3B8;
        --cf-text-3: #64748B;
        --cf-radius-sm: 6px;
        --cf-radius-md: 10px;
        --cf-radius-lg: 14px;
        --cf-font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        --cf-font-mono: 'JetBrains Mono', SFMono-Regular, Menlo, Consolas, monospace;
    }

    /* Core Application Background */
    .stApp {
        background: radial-gradient(ellipse 90% 60% at 50% -15%, rgba(99, 102, 241, 0.10) 0%, rgba(7, 9, 14, 0.98) 75%), #07090E;
        color: var(--cf-text-1);
        font-family: var(--cf-font-sans);
        -webkit-font-smoothing: antialiased;
    }

    /* Studio Header Topbar */
    .studio-topbar {
        background: rgba(15, 19, 32, 0.8);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--cf-border);
        border-radius: var(--cf-radius-lg);
        padding: 20px 24px;
        margin-bottom: 24px;
        display: flex;
        flex-direction: column;
        gap: 12px;
        box-shadow: 0 8px 32px -4px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }
    .studio-brand-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    }
    .studio-logo-group {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .studio-logo-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #6366F1 0%, #06B6D4 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        box-shadow: 0 0 16px rgba(99, 102, 241, 0.4);
    }
    .studio-title {
        color: var(--cf-text-1);
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .studio-badge {
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.3);
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 9999px;
        font-family: var(--cf-font-mono);
        letter-spacing: 0.04em;
    }
    .studio-telemetry-row {
        display: flex;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
    }
    .telemetry-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-family: var(--cf-font-mono);
        color: var(--cf-text-2);
        background: rgba(255, 255, 255, 0.03);
        padding: 4px 10px;
        border-radius: var(--cf-radius-sm);
        border: 1px solid var(--cf-border-subtle);
    }
    .telemetry-dot-ok {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: var(--cf-emerald);
        box-shadow: 0 0 8px var(--cf-emerald);
    }
    .telemetry-dot-violet {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #818CF8;
        box-shadow: 0 0 8px #818CF8;
    }
    .telemetry-dot-cyan {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: var(--cf-cyan);
        box-shadow: 0 0 8px var(--cf-cyan);
    }
    .studio-tagline {
        color: var(--cf-text-2);
        font-size: 0.92rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Section Panels */
    .studio-card {
        background: var(--cf-panel);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--cf-border);
        border-radius: var(--cf-radius-md);
        padding: 20px 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .studio-card:hover {
        border-color: rgba(255, 255, 255, 0.12);
    }

    /* Section Step Header */
    .studio-section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
    }
    .studio-step-num {
        background: rgba(6, 182, 212, 0.12);
        color: var(--cf-cyan-bright);
        border: 1px solid rgba(6, 182, 212, 0.25);
        font-family: var(--cf-font-mono);
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: var(--cf-radius-sm);
        letter-spacing: 0.05em;
    }
    .studio-section-title {
        color: var(--cf-text-1);
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin: 0;
    }

    /* Media Inspector Metrics Grid */
    .inspector-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin: 16px 0;
    }
    .inspector-card {
        background: var(--cf-card);
        border: 1px solid var(--cf-border-subtle);
        border-radius: var(--cf-radius-sm);
        padding: 12px 14px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .inspector-label {
        color: var(--cf-text-3);
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .inspector-val {
        color: var(--cf-text-1);
        font-family: var(--cf-font-mono);
        font-size: 1.15rem;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .inspector-val-cyan {
        color: var(--cf-cyan-bright);
    }

    /* Highlight Candidate Proposal Cards */
    .proposal-card {
        background: var(--cf-card);
        border: 1px solid var(--cf-border);
        border-radius: var(--cf-radius-sm);
        padding: 16px;
        margin-bottom: 12px;
        transition: transform 0.15s ease, border-color 0.15s ease, background-color 0.15s ease;
    }
    .proposal-card:hover {
        background: var(--cf-card-hover);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .proposal-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 8px;
    }
    .proposal-title {
        color: #E0E7FF;
        font-size: 1.05rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .proposal-time-chip {
        background: rgba(6, 182, 212, 0.1);
        color: var(--cf-cyan-bright);
        border: 1px solid rgba(6, 182, 212, 0.25);
        font-family: var(--cf-font-mono);
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 4px;
    }
    .proposal-hook {
        color: #F8FAFC;
        font-style: italic;
        font-size: 0.92rem;
        background: rgba(99, 102, 241, 0.08);
        border-left: 3px solid var(--cf-primary);
        padding: 6px 12px;
        border-radius: 0 4px 4px 0;
        margin: 8px 0;
        line-height: 1.4;
    }
    .proposal-explanation {
        color: var(--cf-text-2);
        font-size: 0.86rem;
        line-height: 1.45;
    }

    /* Notice and Security Pills */
    .studio-notice {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: var(--cf-radius-sm);
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #C7D2FE;
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 12px 0;
        line-height: 1.4;
    }

    /* Empty state hero */
    .empty-hero {
        background: var(--cf-panel);
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-radius: var(--cf-radius-lg);
        padding: 48px 24px;
        text-align: center;
        margin: 20px 0;
    }
    .empty-hero-icon {
        font-size: 3rem;
        margin-bottom: 12px;
    }
    .empty-hero-title {
        color: var(--cf-text-1);
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .empty-hero-sub {
        color: var(--cf-text-2);
        font-size: 0.95rem;
        max-width: 520px;
        margin: 0 auto 20px auto;
        line-height: 1.5;
    }

    /* Roadmap grid */
    .roadmap-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 14px;
        margin-top: 14px;
    }
    .roadmap-card {
        background: var(--cf-card);
        border: 1px solid var(--cf-border-subtle);
        border-radius: var(--cf-radius-sm);
        padding: 16px;
    }
    .roadmap-title {
        color: #E2E8F0;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .roadmap-desc {
        color: var(--cf-text-2);
        font-size: 0.84rem;
        line-height: 1.4;
    }

    /* Streamlit Widget Polish Overrides */
    div[data-testid="stSidebar"] {
        background-color: #0A0D15 !important;
        border-right: 1px solid var(--cf-border) !important;
    }
    div[data-testid="stSidebar"] .block-container {
        padding-top: 2rem !important;
    }

    /* Primary and Secondary Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(180deg, #6366F1 0%, #4F46E5 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        border-radius: var(--cf-radius-sm) !important;
        padding: 0.55rem 1.25rem !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(180deg, #7175f5 0%, #5850ee 100%) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid var(--cf-border) !important;
        color: var(--cf-text-1) !important;
        font-weight: 500 !important;
        border-radius: var(--cf-radius-sm) !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* Download Buttons */
    div.stDownloadButton > button {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        color: #E2E8F0 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        border-radius: var(--cf-radius-sm) !important;
        padding: 0.45rem 0.85rem !important;
        transition: all 0.15s ease !important;
    }
    div.stDownloadButton > button:hover {
        background: rgba(99, 102, 241, 0.2) !important;
        border-color: var(--cf-primary) !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.25) !important;
    }

    /* Form Inputs */
    div[data-baseweb="input"], div[data-baseweb="select"] > div {
        background-color: rgba(13, 17, 26, 0.8) !important;
        border-color: var(--cf-border) !important;
        border-radius: var(--cf-radius-sm) !important;
        color: var(--cf-text-1) !important;
    }
    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
        border-color: var(--cf-primary) !important;
        box-shadow: 0 0 0 1px var(--cf-primary) !important;
    }

    /* Status and Alert Containers */
    div[data-testid="stStatusWidget"] {
        background-color: var(--cf-panel) !important;
        border: 1px solid var(--cf-border) !important;
        border-radius: var(--cf-radius-sm) !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] [role="slider"] {
        background-color: var(--cf-primary) !important;
        border: 2px solid #FFFFFF !important;
    }

    /* Video Player Styling */
    div[data-testid="stVideo"] video {
        border-radius: var(--cf-radius-sm) !important;
        border: 1px solid var(--cf-border) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Verify FFmpeg & Hardware Status
ffprobe_ok = check_ffprobe_available()

# Check speech engine hardware detection (cached in session state)
if "speech_device_str" not in st.session_state:
    try:
        from src.subtitles.whisper_provider import FasterWhisperProvider
        fwp = FasterWhisperProvider()
        dev, ctype = fwp._determine_device()
        st.session_state["speech_device_str"] = f"{dev.upper()} ({ctype})"
    except Exception:
        st.session_state["speech_device_str"] = "CPU (int8)"
speech_device_str = st.session_state["speech_device_str"]

# Sidebar: Studio Inspector & Telemetry
with st.sidebar:
    st.markdown("### 🎛️ Studio Inspector")

    st.markdown(
        f"""
        <div class="studio-card" style="padding: 14px; margin-bottom: 14px;">
            <div style="font-size:0.72rem; color:var(--cf-text-3); text-transform:uppercase; letter-spacing:0.08em; font-weight:700; margin-bottom:6px;">RUNTIME ENGINES</div>
            <div style="font-size:0.82rem; margin-bottom:6px;">
                <span class="telemetry-dot-ok" style="display:inline-block; vertical-align:middle; margin-right:4px;"></span>
                <b>FFmpeg Engine</b>: <span style="font-family:var(--cf-font-mono); color:#38BDF8;">{'Active (Local)' if ffprobe_ok else 'Missing'}</span>
            </div>
            <div style="font-size:0.82rem; margin-bottom:6px;">
                <span class="telemetry-dot-violet" style="display:inline-block; vertical-align:middle; margin-right:4px;"></span>
                <b>Highlight AI</b>: <span style="font-family:var(--cf-font-mono); color:#A5B4FC;">Gemini BYOK</span>
            </div>
            <div style="font-size:0.82rem;">
                <span class="telemetry-dot-cyan" style="display:inline-block; vertical-align:middle; margin-right:4px;"></span>
                <b>Speech ASR</b>: <span style="font-family:var(--cf-font-mono); color:#67E8F9;">{speech_device_str}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 💾 Storage & Isolation")
    st.markdown(
        f"""
        <div style="font-size:0.75rem; color:var(--cf-text-3); font-family:var(--cf-font-mono); margin-bottom:8px;">
            SESSION: <span style="color:var(--cf-cyan);">{session_id}</span> (ISOLATED)
        </div>
        """,
        unsafe_allow_html=True,
    )

    temp_sz = calculate_dir_size_str(TEMP_DIR)
    exp_sz = calculate_dir_size_str(EXPORTS_DIR)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.caption(f"Session Temp: `{temp_sz}`")
    with col_s2:
        st.caption(f"Session Exports: `{exp_sz}`")

    if st.button("🧹 Clear Workspace Cache", help="Safely removes cached upload buffers in this session"):
        for f in TEMP_DIR.iterdir():
            try:
                if f.is_file():
                    f.unlink()
            except Exception:
                pass
        st.success("Session cache purged.")
        st.rerun()

    st.divider()

    st.markdown("#### 🔒 BYOK Privacy Shield")
    st.caption(
        "• API keys exist only in active browser memory.\n"
        "• Zero server logging or disk persistence.\n"
        "• Cloud files deleted immediately after AI parsing.\n"
        "• Video trimming and subtitles run 100% locally."
    )

    st.divider()
    st.markdown("#### 🧭 Studio Navigation")
    st.caption("1. Ingest Media & Inspector\n2. AI Highlight Detection\n3. Format & Subtitle Studio\n4. Export Library\n5. Manual Precision Trimmer")

# Topbar Application Shell
st.markdown(
    """
    <div class="studio-topbar">
        <div class="studio-brand-row">
            <div class="studio-logo-group">
                <div class="studio-logo-icon">🎬</div>
                <div>
                    <h1 class="studio-title">
                        ClipForge <span style="font-weight:300; opacity:0.8;">Studio</span>
                        <span class="studio-badge">v0.5.0-pro</span>
                    </h1>
                </div>
            </div>
            <div class="studio-telemetry-row">
                <div class="telemetry-chip">
                    <span class="telemetry-dot-ok"></span> NLE CORE: FFMPEG 9
                </div>
                <div class="telemetry-chip">
                    <span class="telemetry-dot-violet"></span> BYOK: GEMINI 3.6 FLASH
                </div>
                <div class="telemetry-chip">
                    <span class="telemetry-dot-cyan"></span> WHISPER ASR
                </div>
            </div>
        </div>
        <p class="studio-tagline">
            Professional AI highlight extraction and social reframing studio. Connect your own Gemini API key for zero-markup automatic clipping.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Global warning if FFmpeg is missing
if not ffprobe_ok:
    st.error(
        "🚨 **FFmpeg / ffprobe is unavailable on your system PATH**. "
        "Media inspection, video clipping, and subtitle burning require FFmpeg. "
        "Install via `sudo pacman -S ffmpeg` or `sudo apt install ffmpeg`.",
        icon="🚨",
    )

# Section 1: Video Ingestion
st.markdown(
    """
    <div class="studio-section-header">
        <span class="studio-step-num">01</span>
        <h2 class="studio-section-title">Media Ingest & Source Inspector</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Select or drop a podcast or livestream video file",
    type=["mp4", "mov", "mkv"],
    help="Accepts MP4, MOV, and MKV files. Files are inspected locally on your system.",
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="empty-hero">
            <div class="empty-hero-icon">🎞️</div>
            <div class="empty-hero-title">No Media Loaded into Studio</div>
            <div class="empty-hero-sub">
                Drop an MP4, MOV, or MKV podcast or livestream file above to inspect audio/video tracks,
                detect viral moments with Gemini, and reframe for TikTok, Shorts, and Reels.
            </div>
            <div class="roadmap-grid" style="text-align: left; max-width: 840px; margin: 0 auto;">
                <div class="roadmap-card">
                    <div class="roadmap-title">⚡ Gemini Multimodal BYOK</div>
                    <div class="roadmap-desc">Direct native video analysis or lightweight audio extraction using your own Google Gemini API key.</div>
                </div>
                <div class="roadmap-card">
                    <div class="roadmap-title">💬 Word-Accurate Subtitles</div>
                    <div class="roadmap-desc">Local Faster-Whisper transcription generating SRT and burned hardsubs with zero API costs.</div>
                </div>
                <div class="roadmap-card">
                    <div class="roadmap-title">📱 9:16 & 1:1 Social Framing</div>
                    <div class="roadmap-desc">High-definition vertical export with blurred background filling or centered subject crop.</div>
                </div>
                <div class="roadmap-card">
                    <div class="roadmap-title">🔒 100% Client-Side Privacy</div>
                    <div class="roadmap-desc">Your keys stay in session memory. Local media never leaves your workstation without your consent.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Validate upload size
    if uploaded_file.size <= 0:
        st.error("Uploaded video file is empty (0 bytes). Please upload a valid media file.", icon="⚠️")
        st.stop()

    # Validate file extension
    file_suffix = Path(uploaded_file.name).suffix.lower()
    if file_suffix not in (".mp4", ".mov", ".mkv"):
        st.error(f"Unsupported file extension '{file_suffix}'. ClipForge supports MP4, MOV, and MKV files.", icon="🚨")
        st.stop()

    # Path traversal protection: strip directory components and special characters from stem
    raw_stem = Path(uploaded_file.name).stem
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_stem).strip("_") or "media_source"
    safe_name = f"{safe_stem}{file_suffix}"
    source_temp_path = (TEMP_DIR / f"source_{safe_name}").resolve()

    # Ensure resolved path is strictly within TEMP_DIR
    if not str(source_temp_path).startswith(str(TEMP_DIR.resolve())):
        st.error("Invalid upload filename path structure.", icon="🚨")
        st.stop()

    if not source_temp_path.exists() or source_temp_path.stat().st_size != uploaded_file.size:
        # Stream file in 1MB chunks to prevent allocating full contiguous buffer in memory
        with open(source_temp_path, "wb") as f:
            uploaded_file.seek(0)
            shutil.copyfileobj(uploaded_file, f, length=1024 * 1024)

    try:
        # Retrieve video metadata via ffprobe
        metadata = extract_video_metadata(source_temp_path, original_filename=uploaded_file.name)

        # Inspector metrics bar
        st.markdown(
            f"""
            <div class="inspector-grid">
                <div class="inspector-card">
                    <div class="inspector-label">SOURCE FILE</div>
                    <div class="inspector-val" title="{metadata.filename}">{metadata.filename}</div>
                </div>
                <div class="inspector-card">
                    <div class="inspector-label">TIMECODE DURATION</div>
                    <div class="inspector-val inspector-val-cyan">{metadata.duration_formatted} <span style="font-size:0.75rem; color:var(--cf-text-3);">({metadata.duration_seconds:.1f}s)</span></div>
                </div>
                <div class="inspector-card">
                    <div class="inspector-label">FRAME RESOLUTION</div>
                    <div class="inspector-val">{metadata.resolution_str}</div>
                </div>
                <div class="inspector-card">
                    <div class="inspector-label">PAYLOAD SIZE</div>
                    <div class="inspector-val">{metadata.file_size_formatted}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📺 Expand Source Video Preview Monitor", expanded=False):
            st.video(uploaded_file)

        st.divider()

        # Section 2: AI Highlight Detection (Google Gemini BYOK)
        st.markdown(
            """
            <div class="studio-section-header">
                <span class="studio-step-num">02</span>
                <h2 class="studio-section-title">AI Moment Discovery (Google Gemini BYOK)</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Session state initialization for API key verification and model caches
        if "verified_key_hashes" not in st.session_state:
            st.session_state["verified_key_hashes"] = {}
        if "gemini_models_cache" not in st.session_state:
            st.session_state["gemini_models_cache"] = {}

        st.markdown(
            """
            <div class="studio-card">
            """,
            unsafe_allow_html=True,
        )

        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            provider_choice = st.selectbox(
                "AI Platform Provider",
                options=["Google Gemini (Active)", "OpenAI (Planned)", "Anthropic (Planned)", "Local Ollama (Planned)"],
                index=0,
                help="Select which AI engine analyzes your video content.",
            )
        with col_p2:
            user_api_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                placeholder="Enter Gemini API key (e.g. AIza... or AQ...)",
                help="Your key is held strictly in session memory. It is never logged or written to disk.",
            )

        gemini_provider = provider_registry.get_provider("gemini")
        model_options = []
        selected_model = None
        key_valid_syntax = False

        if "Gemini" in provider_choice and gemini_provider:
            if user_api_key:
                clean_key = user_api_key.strip()
                k_hash = hashlib.sha256(clean_key.encode("utf-8")).hexdigest()
                valid_syntax, _ = gemini_provider.validate_api_key_format(clean_key)
                key_valid_syntax = valid_syntax

                if k_hash in st.session_state["gemini_models_cache"]:
                    model_options = st.session_state["gemini_models_cache"][k_hash]
                elif key_valid_syntax:
                    with st.spinner("Discovering available Gemini models for your credentials..."):
                        try:
                            discovered = gemini_provider.list_available_models(clean_key, timeout_secs=10.0)
                            st.session_state["gemini_models_cache"][k_hash] = discovered
                            st.session_state["verified_key_hashes"][k_hash] = True
                            model_options = discovered
                        except Exception as exc:
                            st.error(f"⚠️ Could not retrieve Gemini models: {exc}")
                            model_options = []

            col_m1, col_m2 = st.columns([4, 1])
            with col_m1:
                if model_options:
                    default_m = gemini_provider.choose_default_model(model_options)
                    default_idx = model_options.index(default_m) if default_m in model_options else 0
                    selected_model = st.selectbox(
                        "Gemini Generative Model",
                        options=model_options,
                        index=default_idx,
                        help="Models discovered dynamically from your Google Gemini account.",
                    )
                elif user_api_key and not key_valid_syntax:
                    selected_model = st.selectbox(
                        "Gemini Generative Model",
                        options=["(Please enter a valid Gemini API key format)"],
                        disabled=True,
                    )
                else:
                    selected_model = st.selectbox(
                        "Gemini Generative Model",
                        options=["(Enter Gemini API key above to load available models)"],
                        disabled=True,
                    )

            with col_m2:
                st.write("")
                st.write("")
                if user_api_key and key_valid_syntax:
                    if st.button("🔄 Refresh", help="Re-query Google for updated model list"):
                        k_hash = hashlib.sha256(user_api_key.strip().encode("utf-8")).hexdigest()
                        if k_hash in st.session_state.get("gemini_models_cache", {}):
                            del st.session_state["gemini_models_cache"][k_hash]
                        st.rerun()

        # Media Strategy & Parameters
        col_st1, col_st2, col_st3 = st.columns([2, 1, 1])
        with col_st1:
            modality_label = st.radio(
                "Analysis Ingestion Strategy",
                options=[
                    "Smart Auto (Optimized Audio for >50MB, Direct Video for <=50MB)",
                    "Audio Only (Recommended for podcasts / fast bandwidth)",
                    "Native Video (Direct video analysis for visual cues)",
                ],
                index=0,
                help="Choose how media is packaged before transmission to Google Gen AI.",
            )
        with col_st2:
            target_clip_count = st.slider("Target Clip Count", min_value=1, max_value=8, value=3)
        with col_st3:
            target_clip_duration = st.slider("Target Duration (secs)", min_value=20, max_value=90, value=45, step=5)

        custom_prompt = st.text_input(
            "Director's Editorial Focus (Optional Prompt)",
            placeholder="e.g. Prioritize high-energy debates, funny punchlines, or actionable takeaways",
        )

        st.markdown(
            """
            <div class="studio-notice">
                <span>🔒</span>
                <div>
                    <b>Direct Cloud Processing</b>: Media is transmitted securely via TLS to Google's Gemini API with your personal key.
                    Files are instantly deleted from Google's servers once analysis finishes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Trigger AI Highlight Detection
        if st.button("⚡ Find Engaging Highlights", type="primary", help="Analyze video with Gemini to discover standalone moments."):
            if not user_api_key:
                st.warning("Please enter your Google Gemini API key to proceed.", icon="🔑")
            else:
                gemini_p = provider_registry.get_provider("gemini")
                if not gemini_p:
                    st.error("Google Gemini provider is not registered.", icon="🚨")
                else:
                    status_ui = st.status("🎬 Processing highlight discovery with Google Gemini...", expanded=True)

                    def ui_progress_callback(msg: str):
                        status_ui.write(f"• {msg}")

                    # Step 1: Live authentication check
                    clean_key = user_api_key.strip()
                    k_hash = hashlib.sha256(clean_key.encode("utf-8")).hexdigest()

                    if k_hash in st.session_state["verified_key_hashes"]:
                        is_auth_valid, auth_msg = True, "API key verified (cached)."
                        status_ui.write("✅ API key verified (cached for this session).")
                    else:
                        status_ui.write("🔑 Authenticating API key with Google servers...")
                        is_auth_valid, auth_msg = gemini_p.verify_api_key(
                            clean_key,
                            check_live=True,
                            timeout_secs=10.0,
                        )
                        if is_auth_valid:
                            st.session_state["verified_key_hashes"][k_hash] = True

                    if not is_auth_valid:
                        status_ui.update(label="❌ Authentication Failed", state="error", expanded=True)
                        st.error(f"🔑 **API Key Authentication Failed**: {auth_msg}", icon="🚨")
                    else:
                        status_ui.write("✅ API key verified successfully.")

                        # Step 2: Selected model verification
                        if not selected_model or selected_model.startswith("("):
                            status_ui.update(label="❌ No Model Selected", state="error", expanded=True)
                            st.error("Please enter a valid API key and select an available Gemini model.", icon="🚨")
                            st.stop()

                        status_ui.write(f"🔍 Verifying access to model '{selected_model}'...")
                        is_model_ok, model_err = gemini_p.verify_model_access(
                            clean_key,
                            selected_model,
                            timeout_secs=10.0,
                        )
                        if not is_model_ok:
                            status_ui.update(label="❌ Model Unavailable", state="error", expanded=True)
                            st.error(f"🚨 **Model Verification Failed**: {model_err}", icon="🚨")
                            st.stop()
                        status_ui.write(f"✅ Model '{selected_model}' ready.")

                        # Resolve preferred modality
                        pref_modality = None
                        if "Audio Only" in modality_label:
                            pref_modality = InputModality.AUDIO_EXTRACTED
                        elif "Native Video" in modality_label:
                            pref_modality = InputModality.NATIVE_VIDEO

                        req = HighlightRequest(
                            video_path=source_temp_path,
                            video_duration=metadata.duration_seconds,
                            file_size_bytes=metadata.file_size_bytes,
                            target_clip_count=target_clip_count,
                            target_clip_duration_secs=float(target_clip_duration),
                            custom_instructions=custom_prompt,
                            preferred_modality=pref_modality,
                        )

                        try:
                            resp = gemini_p.analyze_highlights(
                                request=req,
                                api_key=user_api_key,
                                model_name=selected_model,
                                progress_callback=ui_progress_callback,
                            )

                            if not resp.highlights:
                                status_ui.update(label="⚠️ No Highlights Found", state="complete", expanded=False)
                                st.warning("Gemini did not identify any highlight moments. Try adjusting duration or focus prompt.", icon="⚠️")
                            else:
                                status_ui.update(label=f"✅ Discovered {len(resp.highlights)} Highlights!", state="complete", expanded=False)
                                st.session_state["gemini_highlights"] = resp.highlights
                                st.session_state["gemini_response_meta"] = {
                                    "model": resp.model_name,
                                    "modality": resp.input_modality_used.value,
                                    "usage": resp.usage_metadata,
                                }
                                if "batch_clips_created" in st.session_state:
                                    del st.session_state["batch_clips_created"]
                                st.success(
                                    f"Discovered **{len(resp.highlights)}** candidate moments using `{resp.model_name}` "
                                    f"(Modality: `{resp.input_modality_used.value}`)!",
                                    icon="✨",
                                )
                        except InvalidAPIKeyError as err:
                            status_ui.update(label="❌ Invalid API Key", state="error", expanded=True)
                            st.error(f"🔑 **Invalid API Key**: {err}", icon="🚨")
                        except QuotaExceededError as err:
                            status_ui.update(label="⚠️ Quota Exceeded", state="error", expanded=True)
                            st.error(f"⏳ **Quota / Rate Limit Exceeded**: {err}", icon="⚠️")
                        except ModelNotFoundError as err:
                            status_ui.update(label="❌ Model Unavailable", state="error", expanded=True)
                            st.error(f"🔍 **Model Unavailable**: {err}", icon="❌")
                        except ProviderError as err:
                            status_ui.update(label="❌ Gemini Provider Error", state="error", expanded=True)
                            st.error(f"🤖 **Gemini Provider Error**: {err}", icon="🚨")
                        except Exception as err:
                            status_ui.update(label="❌ Unexpected Error", state="error", expanded=True)
                            st.error(f"⚠️ **Unexpected Error**: {err}", icon="🚨")

        st.markdown("</div>", unsafe_allow_html=True)

        # Section 3: Review Proposed AI Highlights
        discovered_highlights = st.session_state.get("gemini_highlights", [])
        if discovered_highlights:
            st.divider()
            st.markdown(
                """
                <div class="studio-section-header">
                    <span class="studio-step-num">03</span>
                    <h2 class="studio-section-title">Review Proposed Highlights & Export Studio</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="studio-card">
                    <div style="font-size:0.85rem; color:var(--cf-text-2); margin-bottom:12px;">
                        Select which candidate moments you want to export as standalone MP4 clips:
                    </div>
                """,
                unsafe_allow_html=True,
            )

            selected_candidates = []
            for idx, h in enumerate(discovered_highlights, 1):
                col_chk, col_info = st.columns([1, 15])
                with col_chk:
                    is_selected = st.checkbox(
                        label=f"Select Clip {idx}",
                        value=True,
                        key=f"chk_cand_{idx}_{safe_name}",
                        label_visibility="collapsed",
                    )
                with col_info:
                    start_str = format_duration(h.start_time)
                    end_str = format_duration(h.end_time)
                    st.markdown(
                        f"""
                        <div class="proposal-card">
                            <div class="proposal-header">
                                <div class="proposal-title">
                                    <span style="color:var(--cf-primary);">#{idx:02d}</span> {h.title}
                                </div>
                                <div class="proposal-time-chip">
                                    ⏱️ {start_str} ➔ {end_str} • {h.duration:.1f}s
                                </div>
                            </div>
                            <div class="proposal-hook">"{h.hook}"</div>
                            <div class="proposal-explanation">{h.explanation}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if is_selected:
                    selected_candidates.append(h)

            st.markdown("</div>", unsafe_allow_html=True)

            # Export Studio Controls
            st.markdown(
                """
                <div class="studio-card">
                    <div style="font-size:0.95rem; font-weight:700; color:#F8FAFC; margin-bottom:12px; display:flex; align-items:center; gap:8px;">
                        <span>🎚️</span> Output Aspect Ratio & Subtitle Settings
                    </div>
                """,
                unsafe_allow_html=True,
            )

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                gen_subs_batch = st.checkbox(
                    "Generate Subtitles (.srt)",
                    value=False,
                    key="chk_gen_subs_batch",
                    help="Transcribes dialogue locally using Faster-Whisper and exports an SRT subtitle file.",
                )
            with col_opt2:
                burn_subs_batch = st.checkbox(
                    "Burn Subtitles into Video (Hardsub)",
                    value=True,
                    disabled=not gen_subs_batch,
                    key="chk_burn_subs_batch",
                    help="Creates an additional video copy with burned-in captions. Clean video is always preserved.",
                )

            col_ar1, col_ar2, col_ar3 = st.columns(3)
            with col_ar1:
                batch_ar_choice = st.selectbox(
                    "Output Canvas Format",
                    options=[
                        "Original (Source Ratio)",
                        "Vertical 9:16 (Shorts / Reels / TikTok)",
                        "Square 1:1 (Feed / Post)",
                    ],
                    index=0,
                    key="sel_ar_batch",
                    help="Convert clips to 9:16 vertical or 1:1 square. Clean original ratio is always preserved.",
                )

            batch_target_w, batch_target_h = 1080, 1920
            batch_framing_style = FramingStyle.BLURRED_BACKGROUND

            if "Vertical 9:16" in batch_ar_choice:
                with col_ar2:
                    batch_res_choice = st.selectbox(
                        "Vertical Resolution Preset",
                        options=["1080x1920 (Full HD 9:16)", "720x1280 (HD 9:16)"],
                        index=0,
                        key="sel_vert_res_batch",
                    )
                    batch_target_w, batch_target_h = (1080, 1920) if "1080x1920" in batch_res_choice else (720, 1280)
                with col_ar3:
                    batch_style_choice = st.selectbox(
                        "Canvas Framing Style",
                        options=["Blurred Background (Fit Entire Frame)", "Center Crop (Fill Canvas)"],
                        index=0,
                        key="sel_vert_style_batch",
                        help="Blurred Background fits complete frame with blurred backdrop. Center Crop fills 9:16 canvas and crops sides.",
                    )
                    batch_framing_style = FramingStyle.CENTER_CROP if "Center Crop" in batch_style_choice else FramingStyle.BLURRED_BACKGROUND
            elif "Square 1:1" in batch_ar_choice:
                with col_ar2:
                    batch_res_choice = st.selectbox(
                        "Square Resolution Preset",
                        options=["1080x1080 (Full HD 1:1)", "720x720 (Standard 1:1)"],
                        index=0,
                        key="sel_sq_res_batch",
                    )
                    batch_target_w, batch_target_h = (1080, 1080) if "1080x1080" in batch_res_choice else (720, 720)
                with col_ar3:
                    batch_style_choice = st.selectbox(
                        "Canvas Framing Style",
                        options=["Blurred Background (Fit Entire Frame)", "Center Crop (Fill Canvas)"],
                        index=0,
                        key="sel_sq_style_batch",
                        help="Blurred Background fits complete frame with blurred backdrop. Center Crop crops sides to fill 1:1 square.",
                    )
                    batch_framing_style = FramingStyle.CENTER_CROP if "Center Crop" in batch_style_choice else FramingStyle.BLURRED_BACKGROUND

            if "Vertical 9:16" in batch_ar_choice:
                batch_framing_config = FramingConfig(
                    aspect_ratio=AspectRatioMode.VERTICAL_9_16,
                    style=batch_framing_style,
                    target_width=batch_target_w,
                    target_height=batch_target_h,
                )
            elif "Square 1:1" in batch_ar_choice:
                batch_framing_config = FramingConfig(
                    aspect_ratio=AspectRatioMode.SQUARE_1_1,
                    style=batch_framing_style,
                    target_width=batch_target_w,
                    target_height=batch_target_h,
                )
            else:
                batch_framing_config = FramingConfig(aspect_ratio=AspectRatioMode.ORIGINAL)

            col_b1, col_b2 = st.columns([3, 2])
            with col_b1:
                if st.button(f"🎬 Batch Render {len(selected_candidates)} Selected Clips with FFmpeg", type="primary"):
                    if not selected_candidates:
                        st.warning("Please select at least one highlight candidate moment above.", icon="⚠️")
                    else:
                        status_batch = st.status("🎬 Rendering clips with local FFmpeg...", expanded=True)

                        def batch_progress_cb(msg: str):
                            status_batch.write(f"• {msg}")

                        batch_results = batch_create_clips(
                            source_path=source_temp_path,
                            candidates=selected_candidates,
                            output_dir=EXPORTS_DIR,
                            max_duration=metadata.duration_seconds,
                            generate_subtitles=gen_subs_batch,
                            burn_subtitles=burn_subs_batch and gen_subs_batch,
                            framing_config=batch_framing_config,
                            progress_callback=batch_progress_cb,
                        )
                        st.session_state["batch_clips_created"] = batch_results
                        status_batch.update(label="✅ Batch Rendering Complete!", state="complete", expanded=False)

            st.markdown("</div>", unsafe_allow_html=True)

            # Section 4: Render Generated Batch Clips in Studio Gallery
            generated_clips = st.session_state.get("batch_clips_created", [])
            if generated_clips:
                st.divider()
                st.markdown(
                    """
                    <div class="studio-section-header">
                        <span class="studio-step-num">04</span>
                        <h2 class="studio-section-title">Exported Clips Bin & Delivery Assets</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                success_count = sum(1 for c in generated_clips if c.get("success"))
                st.success(
                    f"Successfully generated **{success_count} / {len(generated_clips)}** clips into `exports/` folder!",
                    icon="🎉",
                )

                for c_idx, c_info in enumerate(generated_clips, 1):
                    if c_info.get("success") and c_info.get("output_path"):
                        clip_file = Path(c_info["output_path"])
                        subtitled_file = Path(c_info["subtitled_path"]) if c_info.get("subtitled_path") else None
                        reframed_file = Path(c_info["reframed_path"]) if c_info.get("reframed_path") else None
                        reframed_sub_file = Path(c_info["reframed_subtitled_path"]) if c_info.get("reframed_subtitled_path") else None
                        srt_file = Path(c_info["srt_path"]) if c_info.get("srt_path") else None

                        st.markdown(
                            f"""
                            <div class="studio-card">
                                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:10px;">
                                    <div style="font-size:1.15rem; font-weight:700; color:#F8FAFC;">
                                        <span style="color:var(--cf-primary);">Clip #{c_idx:02d}:</span> {c_info.get('title')}
                                    </div>
                                    <div class="telemetry-chip">
                                        ⏱️ DURATION: {c_info.get('duration'):.1f}s
                                    </div>
                                </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Video player preview options
                        preview_dict = {}
                        if reframed_sub_file and reframed_sub_file.is_file():
                            preview_dict["📱 Reframed + Subtitled"] = reframed_sub_file
                        if reframed_file and reframed_file.is_file():
                            preview_dict["📱 Reframed Clean"] = reframed_file
                        if subtitled_file and subtitled_file.is_file():
                            preview_dict["💬 Original Subtitled"] = subtitled_file
                        if clip_file and clip_file.is_file():
                            preview_dict["🎬 Original Clean"] = clip_file

                        if preview_dict:
                            if len(preview_dict) > 1:
                                prev_choice = st.radio(
                                    f"Monitor Format Switcher (Clip {c_idx})",
                                    options=list(preview_dict.keys()),
                                    horizontal=True,
                                    key=f"rad_prev_{c_idx}_{clip_file.stem}",
                                )
                                st.video(str(preview_dict[prev_choice]))
                            else:
                                only_key = list(preview_dict.keys())[0]
                                st.video(str(preview_dict[only_key]))

                        # Download action bar
                        st.markdown(
                            """
                            <div style="margin: 12px 0 6px 0; font-size:0.75rem; font-weight:700; color:var(--cf-text-3); text-transform:uppercase; letter-spacing:0.06em;">DOWNLOAD EXPORT ASSETS</div>
                            """,
                            unsafe_allow_html=True,
                        )
                        dl_cols = st.columns(4)
                        with dl_cols[0]:
                            if clip_file and clip_file.is_file():
                                with open(clip_file, "rb") as f:
                                    st.download_button(
                                        label="📥 Clean (Original)",
                                        data=f.read(),
                                        file_name=clip_file.name,
                                        mime="video/mp4",
                                        key=f"dl_clean_{c_idx}_{clip_file.stem}",
                                    )
                        with dl_cols[1]:
                            if reframed_file and reframed_file.is_file():
                                with open(reframed_file, "rb") as f:
                                    st.download_button(
                                        label="📥 Clean (Reframed)",
                                        data=f.read(),
                                        file_name=reframed_file.name,
                                        mime="video/mp4",
                                        key=f"dl_reframe_{c_idx}_{reframed_file.stem}",
                                    )
                        with dl_cols[2]:
                            if reframed_sub_file and reframed_sub_file.is_file():
                                with open(reframed_sub_file, "rb") as f:
                                    st.download_button(
                                        label="📥 Subtitled (Reframed)",
                                        data=f.read(),
                                        file_name=reframed_sub_file.name,
                                        mime="video/mp4",
                                        key=f"dl_ref_sub_{c_idx}_{reframed_sub_file.stem}",
                                    )
                            elif subtitled_file and subtitled_file.is_file():
                                with open(subtitled_file, "rb") as f:
                                    st.download_button(
                                        label="📥 Subtitled (Original)",
                                        data=f.read(),
                                        file_name=subtitled_file.name,
                                        mime="video/mp4",
                                        key=f"dl_sub_{c_idx}_{subtitled_file.stem}",
                                    )
                        with dl_cols[3]:
                            if srt_file and srt_file.is_file():
                                with open(srt_file, "rb") as f:
                                    st.download_button(
                                        label="📄 Subtitles (.SRT)",
                                        data=f.read(),
                                        file_name=srt_file.name,
                                        mime="text/plain",
                                        key=f"dl_srt_{c_idx}_{srt_file.stem}",
                                    )

                        if c_info.get("transcript"):
                            with st.expander(f"💬 Audio Dialogue Transcript (Clip #{c_idx:02d})"):
                                st.text(c_info["transcript"])

                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error(f"Clip {c_idx} failed: {c_info.get('error')}", icon="❌")

        st.divider()

        # Section 5: Manual Precision Video Trimmer (Supporting NLE Feature)
        st.markdown(
            """
            <div class="studio-section-header">
                <span class="studio-step-num">05</span>
                <h2 class="studio-section-title">Precision Manual Trimmer (NLE Tool)</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="studio-card">
                <div style="font-size:0.85rem; color:var(--cf-text-2); margin-bottom:12px;">
                    Define frame-accurate in/out timestamp offsets to export an exact manual subclip using local FFmpeg.
                </div>
            """,
            unsafe_allow_html=True,
        )

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            start_time = st.number_input(
                "In-Point (Start Time in Seconds)",
                min_value=0.0,
                max_value=float(max(0.0, metadata.duration_seconds)),
                value=0.0,
                step=0.5,
                format="%.2f",
                help="Start offset of the clip in seconds (must be >= 0.0).",
            )
        with col_t2:
            default_end = min(
                metadata.duration_seconds,
                15.0 if metadata.duration_seconds >= 15.0 else metadata.duration_seconds,
            )
            end_time = st.number_input(
                "Out-Point (End Time in Seconds)",
                min_value=0.0,
                max_value=float(max(0.0, metadata.duration_seconds)),
                value=float(default_end),
                step=0.5,
                format="%.2f",
                help="End offset of the clip in seconds (must be > start time and <= duration).",
            )

        # Calculate and show clip duration
        clip_duration = end_time - start_time
        if clip_duration > 0:
            st.markdown(
                f"""
                <div class="telemetry-chip" style="margin-bottom:12px;">
                    ⏱️ SELECTED DURATION: <span style="color:var(--cf-cyan-bright);">{clip_duration:.2f}s</span> ({format_duration(clip_duration)})
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Validate timestamps
        validation_error = None
        try:
            validate_timestamps(start_time, end_time, metadata.duration_seconds)
        except TimestampValidationError as err:
            validation_error = str(err)

        if validation_error:
            st.warning(f"⚠️ **Validation Notice**: {validation_error}")

        # Subtitle and Aspect Ratio options for manual clip
        col_ms1, col_ms2 = st.columns(2)
        with col_ms1:
            manual_gen_subs = st.checkbox("Generate Subtitles for Manual Clip", value=False, key="chk_manual_subs")
        with col_ms2:
            manual_burn_subs = st.checkbox(
                "Burn Subtitles into Video",
                value=True,
                disabled=not manual_gen_subs,
                key="chk_manual_burn",
            )

        col_mar1, col_mar2, col_mar3 = st.columns(3)
        with col_mar1:
            manual_ar_choice = st.selectbox(
                "Canvas Format",
                options=[
                    "Original (Source Ratio)",
                    "Vertical 9:16 (Shorts / Reels / TikTok)",
                    "Square 1:1 (Feed / Post)",
                ],
                index=0,
                key="sel_manual_ar",
            )

        m_target_w, m_target_h = 1080, 1920
        m_framing_style = FramingStyle.BLURRED_BACKGROUND

        if "Vertical 9:16" in manual_ar_choice:
            with col_mar2:
                m_res_choice = st.selectbox(
                    "Vertical Resolution",
                    options=["1080x1920 (Full HD)", "720x1280 (HD)"],
                    index=0,
                    key="sel_manual_vert_res",
                )
                m_target_w, m_target_h = (1080, 1920) if "1080x1920" in m_res_choice else (720, 1280)
            with col_mar3:
                m_style_choice = st.selectbox(
                    "Framing Mode",
                    options=["Blurred Background (Fit Entire Frame)", "Center Crop (Fill Canvas)"],
                    index=0,
                    key="sel_manual_vert_style",
                )
                m_framing_style = FramingStyle.CENTER_CROP if "Center Crop" in m_style_choice else FramingStyle.BLURRED_BACKGROUND
        elif "Square 1:1" in manual_ar_choice:
            with col_mar2:
                m_res_choice = st.selectbox(
                    "Square Resolution",
                    options=["1080x1080 (Full HD)", "720x720 (Standard)"],
                    index=0,
                    key="sel_manual_sq_res",
                )
                m_target_w, m_target_h = (1080, 1080) if "1080x1080" in m_res_choice else (720, 720)
            with col_mar3:
                m_style_choice = st.selectbox(
                    "Framing Mode",
                    options=["Blurred Background (Fit Entire Frame)", "Center Crop (Fill Canvas)"],
                    index=0,
                    key="sel_manual_sq_style",
                )
                m_framing_style = FramingStyle.CENTER_CROP if "Center Crop" in m_style_choice else FramingStyle.BLURRED_BACKGROUND

        if "Vertical 9:16" in manual_ar_choice:
            manual_framing_config = FramingConfig(
                aspect_ratio=AspectRatioMode.VERTICAL_9_16,
                style=m_framing_style,
                target_width=m_target_w,
                target_height=m_target_h,
            )
        elif "Square 1:1" in manual_ar_choice:
            manual_framing_config = FramingConfig(
                aspect_ratio=AspectRatioMode.SQUARE_1_1,
                style=m_framing_style,
                target_width=m_target_w,
                target_height=m_target_h,
            )
        else:
            manual_framing_config = FramingConfig(aspect_ratio=AspectRatioMode.ORIGINAL)

        # Trigger clipping
        if st.button("✂️ Render Manual Subclip", type="secondary", disabled=bool(validation_error)):
            clip_stamp = int(time.time())
            stem = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(metadata.filename).stem)
            export_path = EXPORTS_DIR / f"clip_{stem}_{start_time:.1f}s_to_{end_time:.1f}s_{clip_stamp}.mp4"

            with st.spinner("Processing clip with FFmpeg..."):
                try:
                    created_clip = create_video_clip(
                        input_path=source_temp_path,
                        output_path=export_path,
                        start_time=start_time,
                        end_time=end_time,
                        max_duration=metadata.duration_seconds,
                    )
                    st.session_state["latest_clip"] = str(created_clip)
                    st.session_state["latest_clip_srt"] = None
                    st.session_state["latest_clip_subtitled"] = None
                    st.session_state["latest_clip_reframed"] = None
                    st.session_state["latest_clip_reframed_subtitled"] = None

                    srt_path = None
                    if manual_gen_subs:
                        from src.subtitles import burn_subtitles_to_video, transcription_registry
                        transcriber = transcription_registry.get_provider()
                        if transcriber and transcriber.is_available():
                            trans_res = transcriber.transcribe(created_clip)
                            if trans_res.has_speech:
                                srt_path = export_path.with_suffix(".srt")
                                srt_path.write_text(trans_res.to_srt(offset_secs=0.0), encoding="utf-8")
                                st.session_state["latest_clip_srt"] = str(srt_path)

                                if manual_burn_subs:
                                    sub_path = export_path.with_name(f"{export_path.stem}_subtitled.mp4")
                                    burn_subtitles_to_video(created_clip, srt_path, sub_path)
                                    st.session_state["latest_clip_subtitled"] = str(sub_path)

                    if manual_framing_config.is_reframe_needed():
                        suffix = manual_framing_config.get_suffix()
                        m_reframed = export_path.with_name(f"{export_path.stem}{suffix}.mp4")
                        reframe_video(created_clip, m_reframed, manual_framing_config)
                        st.session_state["latest_clip_reframed"] = str(m_reframed)

                        if manual_burn_subs and srt_path and srt_path.is_file():
                            m_ref_sub = export_path.with_name(f"{export_path.stem}{suffix}_subtitled.mp4")
                            reframe_video(created_clip, m_ref_sub, manual_framing_config, srt_path=srt_path)
                            st.session_state["latest_clip_reframed_subtitled"] = str(m_ref_sub)

                    st.success(f"Clip created successfully: **{export_path.name}**", icon="🎉")
                except (TimestampValidationError, VideoClipperError, Exception) as err:
                    st.error(f"Clipping failed: {err}", icon="🚨")

        # Preview and download latest exported clip
        latest_clip_path = st.session_state.get("latest_clip")
        latest_sub_path = st.session_state.get("latest_clip_subtitled")
        latest_ref_path = st.session_state.get("latest_clip_reframed")
        latest_ref_sub_path = st.session_state.get("latest_clip_reframed_subtitled")
        latest_srt_path = st.session_state.get("latest_clip_srt")

        if latest_clip_path and Path(latest_clip_path).is_file():
            st.markdown("#### 🎬 Exported Manual Clip Monitor")
            m_prev_options = {}
            if latest_ref_sub_path and Path(latest_ref_sub_path).is_file():
                m_prev_options["📱 Reframed + Subtitled"] = latest_ref_sub_path
            if latest_ref_path and Path(latest_ref_path).is_file():
                m_prev_options["📱 Reframed Clean"] = latest_ref_path
            if latest_sub_path and Path(latest_sub_path).is_file():
                m_prev_options["💬 Original Subtitled"] = latest_sub_path
            if Path(latest_clip_path).is_file():
                m_prev_options["🎬 Original Clean"] = latest_clip_path

            if len(m_prev_options) > 1:
                m_choice = st.radio(
                    "Preview Version",
                    options=list(m_prev_options.keys()),
                    horizontal=True,
                    key="rad_prev_manual",
                )
                st.video(m_prev_options[m_choice])
            elif m_prev_options:
                only_m_key = list(m_prev_options.keys())[0]
                st.video(m_prev_options[only_m_key])

            m_dl_cols = st.columns(4)
            with m_dl_cols[0]:
                with open(latest_clip_path, "rb") as clip_file:
                    st.download_button(
                        label="📥 Clean (Original)",
                        data=clip_file.read(),
                        file_name=Path(latest_clip_path).name,
                        mime="video/mp4",
                        key="dl_manual_clean",
                    )
            with m_dl_cols[1]:
                if latest_ref_path and Path(latest_ref_path).is_file():
                    with open(latest_ref_path, "rb") as ref_file:
                        st.download_button(
                            label="📥 Clean (Reframed)",
                            data=ref_file.read(),
                            file_name=Path(latest_ref_path).name,
                            mime="video/mp4",
                            key="dl_manual_ref",
                        )
            with m_dl_cols[2]:
                if latest_ref_sub_path and Path(latest_ref_sub_path).is_file():
                    with open(latest_ref_sub_path, "rb") as ref_sub_file:
                        st.download_button(
                            label="📥 Subtitled (Reframed)",
                            data=ref_sub_file.read(),
                            file_name=Path(latest_ref_sub_path).name,
                            mime="video/mp4",
                            key="dl_manual_ref_sub",
                        )
                elif latest_sub_path and Path(latest_sub_path).is_file():
                    with open(latest_sub_path, "rb") as sub_file:
                        st.download_button(
                            label="📥 Subtitled (Original)",
                            data=sub_file.read(),
                            file_name=Path(latest_sub_path).name,
                            mime="video/mp4",
                            key="dl_manual_sub",
                        )
            with m_dl_cols[3]:
                if latest_srt_path and Path(latest_srt_path).is_file():
                    with open(latest_srt_path, "rb") as srt_file:
                        st.download_button(
                            label="📄 Subtitles (.SRT)",
                            data=srt_file.read(),
                            file_name=Path(latest_srt_path).name,
                            mime="text/plain",
                            key="dl_manual_srt",
                        )

        st.markdown("</div>", unsafe_allow_html=True)

    except FFmpegNotFoundError as err:
        st.error(f"FFmpeg Error: {err}", icon="🚨")
    except InvalidVideoError as err:
        st.error(f"Invalid Video File: {err}", icon="❌")
    except Exception as err:
        st.error(f"Failed to inspect video: {err}", icon="⚠️")

st.divider()

# Section 6: Future Studio Pipeline Extensions
st.markdown(
    """
    <div class="studio-section-header">
        <span class="studio-step-num">06</span>
        <h2 class="studio-section-title">Studio Roadmap & Future Engines</h2>
    </div>
    <div class="roadmap-grid">
        <div class="roadmap-card">
            <div class="roadmap-title">🌐 Multi-Provider AI Architecture</div>
            <div class="roadmap-desc">Plug-and-play adapter endpoints for OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet, and local offline LLMs via Ollama.</div>
        </div>
        <div class="roadmap-card">
            <div class="roadmap-title">🎯 Active Speaker CV Auto-Tracking</div>
            <div class="roadmap-desc">Computer-vision active speaker detection and smooth dynamic face-centering panning across multi-camera podcast setups.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
