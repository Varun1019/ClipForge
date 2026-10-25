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
    acquire_job_slot,
    calculate_dir_size_str,
    check_disk_space_available,
    cleanup_stale_files,
    init_session_workspace,
    CapacityExceededError,
    DEFAULT_MAX_UPLOAD_MB,
    DEFAULT_MAX_VIDEO_DURATION_SECONDS,
    DEFAULT_MIN_FREE_DISK_BYTES,
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
    # Run stale cleanup once on session start, protecting current session
    cleanup_stale_files(
        base_dirs=[BASE_TEMP_DIR, BASE_EXPORTS_DIR],
        max_age_seconds=3600,
        active_session_ids=[st.session_state["session_id"]],
    )

session_id = st.session_state["session_id"]
TEMP_DIR, EXPORTS_DIR = init_session_workspace(BASE_TEMP_DIR, BASE_EXPORTS_DIR, session_id)

# Premium Liquid Glass Design System CSS
st.markdown(
    """
    <style>
    /* Google Fonts: Instrument Serif + Inter + JetBrains Mono */
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --cf-pure-black: #000000;
        --cf-deep-black: #050505;
        --cf-elevated-black: #080808;
        --cf-glass-bg: rgba(255, 255, 255, 0.01);
        --cf-glass-border: rgba(255, 255, 255, 0.05);
        --cf-glass-border-hover: rgba(255, 255, 255, 0.12);
        --cf-glass-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.05);
        --cf-text-primary: #FFFFFF;
        --cf-text-secondary: rgba(255, 255, 255, 0.60);
        --cf-text-muted: rgba(255, 255, 255, 0.40);
        --cf-ease: cubic-bezier(0.34, 1.56, 0.64, 1);
        --cf-font-serif: 'Instrument Serif', Georgia, serif;
        --cf-font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        --cf-font-mono: 'JetBrains Mono', monospace;

        /* Monochromatic compatibility aliases */
        --cf-text-1: var(--cf-text-primary);
        --cf-text-2: var(--cf-text-secondary);
        --cf-text-3: var(--cf-text-muted);
        --cf-border: var(--cf-glass-border);
        --cf-border-subtle: rgba(255, 255, 255, 0.03);
        --cf-cyan: rgba(255, 255, 255, 0.85);
        --cf-cyan-bright: #FFFFFF;
        --cf-primary: #FFFFFF;
        --cf-card: var(--cf-elevated-black);
        --cf-card-hover: rgba(255, 255, 255, 0.03);
        --cf-panel: var(--cf-glass-bg);
    }

    /* Core Application Background */
    .stApp {
        background-color: var(--cf-pure-black) !important;
        color: var(--cf-text-primary) !important;
        font-family: var(--cf-font-sans) !important;
        -webkit-font-smoothing: antialiased;
        letter-spacing: -0.01em;
    }

    /* Subtle Monochromatic Grain / Vignette Overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle 800px at 50% -200px, rgba(255, 255, 255, 0.025), transparent 80%),
                    radial-gradient(circle 600px at 80% 60%, rgba(255, 255, 255, 0.015), transparent 70%);
        pointer-events: none;
        z-index: 0;
    }

    /* Browser Fallback for Browsers Without backdrop-filter */
    @supports not (backdrop-filter: blur(8px)) {
        .liquid-nav, .liquid-card, .studio-card, .editorial-graphic-panel, .hero-glass-btn, div[data-testid="stFileUploader"] {
            background-color: var(--cf-elevated-black) !important;
        }
    }

    /* Responsive Top Navigation */
    .liquid-nav {
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(5, 5, 5, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid var(--cf-glass-border);
        padding: 16px 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: -6rem -4rem 2rem -4rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.6);
    }
    @media (max-width: 768px) {
        .liquid-nav {
            margin: -4rem -1rem 1.5rem -1rem;
            padding: 12px 16px;
        }
    }
    .liquid-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        text-decoration: none !important;
        color: var(--cf-text-primary);
    }
    .liquid-brand-title {
        font-family: var(--cf-font-serif);
        font-size: 1.75rem;
        font-weight: 400;
        letter-spacing: -0.02em;
        color: var(--cf-text-primary);
    }
    .liquid-badge {
        font-family: var(--cf-font-mono);
        font-size: 0.70rem;
        color: var(--cf-text-secondary);
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--cf-glass-border);
        padding: 2px 8px;
        border-radius: 9999px;
        letter-spacing: 0.05em;
    }
    .liquid-nav-links {
        display: flex;
        align-items: center;
        gap: 28px;
    }
    @media (max-width: 680px) {
        .liquid-nav-links { display: none; }
    }
    .liquid-nav-link {
        color: var(--cf-text-secondary) !important;
        text-decoration: none !important;
        font-size: 0.88rem;
        font-weight: 400;
        position: relative;
        padding: 4px 0;
        transition: color 0.25s var(--cf-ease), transform 0.25s var(--cf-ease);
    }
    .liquid-nav-link:hover {
        color: var(--cf-text-primary) !important;
        transform: scale(1.04);
    }
    .liquid-nav-link::after {
        content: '';
        position: absolute;
        width: 0;
        height: 1px;
        bottom: 0;
        left: 50%;
        background-color: var(--cf-text-primary);
        transition: all 0.3s var(--cf-ease);
        transform: translateX(-50%);
    }
    .liquid-nav-link:hover::after {
        width: 100%;
    }
    .liquid-nav-cta {
        background: var(--cf-text-primary) !important;
        color: var(--cf-pure-black) !important;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 8px 18px;
        border-radius: 9999px;
        text-decoration: none !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.25s var(--cf-ease);
    }
    .liquid-nav-cta:hover {
        background: rgba(255, 255, 255, 0.9) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(255, 255, 255, 0.2);
    }

    /* HERO SECTION */
    .liquid-hero {
        padding: 42px 0 32px 0;
        text-align: left;
        max-width: 960px;
        margin: 0 auto 36px auto;
    }
    .hero-headline {
        font-family: var(--cf-font-serif);
        font-size: clamp(2.8rem, 7vw, 5.4rem);
        line-height: 1.05;
        font-weight: 400;
        color: var(--cf-text-primary);
        margin: 0 0 20px 0;
        letter-spacing: -0.03em;
    }
    .hero-headline-indent {
        display: block;
        padding-left: clamp(1.5rem, 5vw, 4rem);
        font-style: italic;
        color: rgba(255, 255, 255, 0.85);
    }
    .hero-tagline-wrapper {
        min-height: 32px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
    }
    .hero-tagline-rotating {
        font-family: var(--cf-font-serif);
        font-size: clamp(1.1rem, 2.4vw, 1.45rem);
        font-style: italic;
        color: var(--cf-text-secondary);
        display: inline-block;
    }
    .hero-tagline-rotating::after {
        content: "AI-guided highlight discovery for long podcasts.";
        animation: taglineShuffle 14s infinite;
        transition: opacity 0.7s var(--cf-ease);
    }
    @keyframes taglineShuffle {
        0%, 20% { content: "AI-guided highlight discovery for long podcasts."; opacity: 1; }
        22.5% { opacity: 0; }
        25%, 45% { content: "Precision local clipping without middleman cloud fees."; opacity: 1; }
        47.5% { opacity: 0; }
        50%, 70% { content: "Automatic synchronized subtitles powered by Whisper."; opacity: 1; }
        72.5% { opacity: 0; }
        75%, 95% { content: "Studio-grade 9:16 and 1:1 social video reframing."; opacity: 1; }
        97.5% { opacity: 0; }
        100% { opacity: 1; content: "AI-guided highlight discovery for long podcasts."; }
    }
    .hero-subcopy {
        font-size: clamp(0.95rem, 1.8vw, 1.15rem);
        color: var(--cf-text-secondary);
        line-height: 1.6;
        max-width: 640px;
        margin-bottom: 28px;
        font-weight: 300;
    }
    .hero-actions {
        display: flex;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
    }
    .hero-glass-btn {
        background: var(--cf-glass-bg);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid var(--cf-glass-border);
        box-shadow: var(--cf-glass-shadow);
        border-radius: 9999px;
        padding: 12px 28px;
        color: var(--cf-text-primary) !important;
        font-size: 0.92rem;
        font-weight: 500;
        text-decoration: none !important;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        transition: all 0.3s var(--cf-ease);
    }
    .hero-glass-btn:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: var(--cf-glass-border-hover);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6), inset 0 1px 1px rgba(255, 255, 255, 0.1);
    }

    /* FEATURE CARDS GRID */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 18px;
        margin-bottom: 48px;
    }
    .liquid-card {
        background: var(--cf-glass-bg);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid var(--cf-glass-border);
        box-shadow: var(--cf-glass-shadow);
        border-radius: 24px;
        padding: 24px 22px;
        transition: all 0.35s var(--cf-ease);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .liquid-card:hover {
        background: rgba(255, 255, 255, 0.025);
        border-color: var(--cf-glass-border-hover);
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.8), inset 0 1px 2px rgba(255, 255, 255, 0.08);
    }
    .liquid-card-icon {
        width: 32px;
        height: 32px;
        opacity: 0.40;
        margin-bottom: 16px;
        transition: opacity 0.3s var(--cf-ease), transform 0.3s var(--cf-ease);
        color: var(--cf-text-primary);
    }
    .liquid-card:hover .liquid-card-icon {
        opacity: 1.0;
        transform: scale(1.08);
    }
    .liquid-card-title {
        font-family: var(--cf-font-serif);
        font-size: 1.35rem;
        font-weight: 400;
        color: var(--cf-text-primary);
        margin: 0 0 8px 0;
    }
    .liquid-card-desc {
        font-size: 0.84rem;
        color: var(--cf-text-secondary);
        line-height: 1.5;
        margin: 0;
    }

    /* FEATURE EXPLANATION SECTION (Two Column Editorial) */
    .editorial-section {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 36px;
        align-items: center;
        margin-bottom: 56px;
        padding: 16px 0;
    }
    @media (max-width: 860px) {
        .editorial-section {
            grid-template-columns: 1fr;
            gap: 24px;
        }
    }
    .editorial-left {
        display: flex;
        flex-direction: column;
        gap: 16px;
    }
    .editorial-heading {
        font-family: var(--cf-font-serif);
        font-size: clamp(2rem, 3.8vw, 2.9rem);
        line-height: 1.15;
        font-weight: 400;
        color: var(--cf-text-primary);
        margin: 0;
    }
    .editorial-desc {
        font-size: 0.95rem;
        color: var(--cf-text-secondary);
        line-height: 1.6;
    }
    .editorial-list {
        list-style: none;
        padding: 0;
        margin: 8px 0 0 0;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .editorial-item {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.88rem;
        color: var(--cf-text-primary);
    }
    .editorial-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.4);
    }
    .editorial-muted {
        font-size: 0.76rem;
        color: var(--cf-text-muted);
        font-family: var(--cf-font-mono);
        margin-top: 8px;
        letter-spacing: 0.05em;
    }

    /* Decorative Waveform & Timeline Graphic */
    .editorial-graphic-panel {
        background: var(--cf-glass-bg);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid var(--cf-glass-border);
        box-shadow: var(--cf-glass-shadow);
        border-radius: 24px;
        padding: 28px;
        display: flex;
        flex-direction: column;
        gap: 18px;
        position: relative;
        overflow: hidden;
    }
    .timeline-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: var(--cf-font-mono);
        font-size: 0.72rem;
        color: var(--cf-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .timeline-track {
        height: 68px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        display: flex;
        align-items: center;
        padding: 0 16px;
        gap: 5px;
        position: relative;
        overflow: hidden;
    }
    .waveform-bar {
        flex: 1;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 2px;
        transition: all 0.3s var(--cf-ease);
    }
    .waveform-bar:hover {
        background: #FFFFFF;
        transform: scaleY(1.3);
    }
    .timeline-marker-region {
        position: absolute;
        left: 28%;
        width: 44%;
        top: 0; bottom: 0;
        background: rgba(255, 255, 255, 0.05);
        border-left: 2px solid #FFFFFF;
        border-right: 2px solid #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: var(--cf-font-mono);
        font-size: 0.70rem;
        color: var(--cf-text-primary);
        letter-spacing: 0.04em;
    }
    .timeline-meta-row {
        display: flex;
        justify-content: space-between;
        font-family: var(--cf-font-mono);
        font-size: 0.72rem;
        color: var(--cf-text-secondary);
    }

    /* SECTION TITLES & HEADERS */
    .studio-section-header {
        display: flex;
        align-items: baseline;
        gap: 14px;
        margin: 32px 0 16px 0;
    }
    .studio-step-num {
        font-family: var(--cf-font-mono);
        font-size: 0.72rem;
        color: var(--cf-text-muted);
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--cf-glass-border);
        padding: 2px 8px;
        border-radius: 6px;
        letter-spacing: 0.06em;
    }
    .studio-section-title {
        font-family: var(--cf-font-serif);
        font-size: 1.85rem;
        font-weight: 400;
        color: var(--cf-text-primary);
        margin: 0;
        letter-spacing: -0.01em;
    }

    /* WORKSPACE GLASS PANELS */
    .studio-card {
        background: var(--cf-glass-bg);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid var(--cf-glass-border);
        box-shadow: var(--cf-glass-shadow);
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 24px;
        transition: border-color 0.25s var(--cf-ease);
    }
    .studio-card:hover {
        border-color: var(--cf-glass-border-hover);
    }

    /* MEDIA INSPECTOR METRICS GRID */
    .inspector-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin: 18px 0;
    }
    .inspector-card {
        background: var(--cf-elevated-black);
        border: 1px solid var(--cf-glass-border);
        border-radius: 16px;
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .inspector-label {
        color: var(--cf-text-muted);
        font-size: 0.68rem;
        font-family: var(--cf-font-mono);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .inspector-val {
        color: var(--cf-text-primary);
        font-family: var(--cf-font-mono);
        font-size: 1.10rem;
        font-weight: 500;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .inspector-val-cyan {
        color: var(--cf-text-primary);
    }

    /* PROPOSAL CARDS */
    .proposal-card {
        background: var(--cf-elevated-black);
        border: 1px solid var(--cf-glass-border);
        border-radius: 20px;
        padding: 18px 20px;
        margin-bottom: 14px;
        transition: all 0.25s var(--cf-ease);
    }
    .proposal-card:hover {
        background: rgba(255, 255, 255, 0.02);
        border-color: var(--cf-glass-border-hover);
        transform: translateY(-1px);
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
        color: var(--cf-text-primary);
        font-family: var(--cf-font-serif);
        font-size: 1.25rem;
        font-weight: 400;
    }
    .proposal-time-chip {
        background: rgba(255, 255, 255, 0.04);
        color: var(--cf-text-secondary);
        border: 1px solid var(--cf-glass-border);
        font-family: var(--cf-font-mono);
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 6px;
    }
    .proposal-hook {
        color: var(--cf-text-primary);
        font-style: italic;
        font-family: var(--cf-font-serif);
        font-size: 1.05rem;
        background: rgba(255, 255, 255, 0.02);
        border-left: 2px solid rgba(255, 255, 255, 0.4);
        padding: 8px 14px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
        line-height: 1.4;
    }
    .proposal-explanation {
        color: var(--cf-text-secondary);
        font-size: 0.86rem;
        line-height: 1.5;
    }

    /* Notice and Security Pills */
    .studio-notice {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--cf-glass-border);
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: var(--cf-text-secondary);
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 14px 0;
        line-height: 1.4;
    }

    /* STREAMLIT CONTROLS OVERRIDES */
    /* Sidebar */
    div[data-testid="stSidebar"] {
        background-color: var(--cf-deep-black) !important;
        border-right: 1px solid var(--cf-glass-border) !important;
    }
    div[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
    }

    /* Primary Action Buttons */
    div.stButton > button[kind="primary"] {
        background: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: #000000 !important;
        font-family: var(--cf-font-sans) !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        border-radius: 9999px !important;
        padding: 0.60rem 1.6rem !important;
        box-shadow: 0 4px 16px rgba(255, 255, 255, 0.15) !important;
        transition: all 0.25s var(--cf-ease) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: rgba(255, 255, 255, 0.92) !important;
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 6px 24px rgba(255, 255, 255, 0.3) !important;
    }

    /* Secondary Action Buttons */
    div.stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid var(--cf-glass-border) !important;
        color: var(--cf-text-primary) !important;
        font-family: var(--cf-font-sans) !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        border-radius: 9999px !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.25s var(--cf-ease) !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: var(--cf-glass-border-hover) !important;
        transform: translateY(-1px) !important;
    }

    /* Download Buttons */
    div.stDownloadButton > button {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid var(--cf-glass-border) !important;
        color: var(--cf-text-primary) !important;
        font-family: var(--cf-font-sans) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        border-radius: 9999px !important;
        padding: 0.50rem 1.0rem !important;
        transition: all 0.25s var(--cf-ease) !important;
    }
    div.stDownloadButton > button:hover {
        background: #FFFFFF !important;
        color: #000000 !important;
        border-color: #FFFFFF !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(255, 255, 255, 0.2) !important;
    }

    /* Form Inputs, Selectboxes, Textareas */
    div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="textarea"] {
        background-color: var(--cf-elevated-black) !important;
        border-color: var(--cf-glass-border) !important;
        border-radius: 12px !important;
        color: var(--cf-text-primary) !important;
        font-family: var(--cf-font-sans) !important;
    }
    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within, div[data-baseweb="textarea"]:focus-within {
        border-color: rgba(255, 255, 255, 0.35) !important;
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.2) !important;
    }

    /* Status and Alert Containers */
    div[data-testid="stStatusWidget"] {
        background-color: var(--cf-elevated-black) !important;
        border: 1px solid var(--cf-glass-border) !important;
        border-radius: 16px !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] [role="slider"] {
        background-color: #FFFFFF !important;
        border: 2px solid var(--cf-pure-black) !important;
    }

    /* File Uploader Container */
    div[data-testid="stFileUploader"] {
        background: var(--cf-glass-bg);
        border: 1px dashed var(--cf-glass-border);
        border-radius: 24px;
        padding: 16px;
        transition: border-color 0.25s var(--cf-ease);
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: var(--cf-glass-border-hover);
    }

    /* Video Player Styling */
    div[data-testid="stVideo"] video {
        border-radius: 18px !important;
        border: 1px solid var(--cf-glass-border) !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.8) !important;
    }

    /* Reduced Motion */
    @media (prefers-reduced-motion: reduce) {
        *, ::before, ::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
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
        <div class="studio-card" style="padding: 16px; margin-bottom: 16px; border-radius: 18px;">
            <div style="font-size:0.68rem; color:var(--cf-text-muted); text-transform:uppercase; letter-spacing:0.08em; font-family:var(--cf-font-mono); margin-bottom:8px;">RUNTIME ENGINES</div>
            <div style="font-size:0.82rem; margin-bottom:6px; color:var(--cf-text-primary);">
                <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#FFFFFF; margin-right:6px; box-shadow:0 0 6px #FFFFFF;"></span>
                <b>FFmpeg</b>: <span style="font-family:var(--cf-font-mono); color:var(--cf-text-secondary);">{'Active (Local)' if ffprobe_ok else 'Missing'}</span>
            </div>
            <div style="font-size:0.82rem; margin-bottom:6px; color:var(--cf-text-primary);">
                <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:rgba(255,255,255,0.6); margin-right:6px;"></span>
                <b>Highlight AI</b>: <span style="font-family:var(--cf-font-mono); color:var(--cf-text-secondary);">Gemini BYOK</span>
            </div>
            <div style="font-size:0.82rem; color:var(--cf-text-primary);">
                <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:rgba(255,255,255,0.6); margin-right:6px;"></span>
                <b>Speech ASR</b>: <span style="font-family:var(--cf-font-mono); color:var(--cf-text-secondary);">{speech_device_str}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 💾 Storage & Isolation")
    st.markdown(
        f"""
        <div style="font-size:0.75rem; color:var(--cf-text-muted); font-family:var(--cf-font-mono); margin-bottom:8px;">
            SESSION: <span style="color:var(--cf-text-primary);">{session_id}</span> (ISOLATED)
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

# Liquid Glass Top Navigation
st.markdown(
    """
    <nav class="liquid-nav">
        <a href="#" class="liquid-brand">
            <span class="liquid-brand-title">ClipForge</span>
            <span class="liquid-badge">v0.5.0-pro</span>
        </a>
        <div class="liquid-nav-links">
            <a href="#capabilities" class="liquid-nav-link">Capabilities</a>
            <a href="#architecture" class="liquid-nav-link">Architecture</a>
            <a href="#workspace" class="liquid-nav-link">Studio Workspace</a>
        </div>
        <a href="#workspace" class="liquid-nav-cta">Launch Studio ↓</a>
    </nav>
    """,
    unsafe_allow_html=True,
)

# Hero Section
st.markdown(
    """
    <section class="liquid-hero">
        <h1 class="hero-headline">
            Every moment.<br>
            <span class="hero-headline-indent">Worth watching.</span>
        </h1>
        <div class="hero-tagline-wrapper">
            <span class="hero-tagline-rotating"></span>
        </div>
        <p class="hero-subcopy">
            Turn long videos into compelling short clips with AI-assisted highlight detection and local video processing.
        </p>
        <div class="hero-actions">
            <a href="#workspace" class="hero-glass-btn">
                Open Video Workspace
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><polyline points="19 12 12 19 5 12"></polyline></svg>
            </a>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# Feature Cards Grid
st.markdown(
    """
    <div id="capabilities"></div>
    <section class="features-grid">
        <div class="liquid-card">
            <div>
                <svg class="liquid-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                    <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
                    <path d="M5 3v4"/>
                    <path d="M19 17v4"/>
                    <path d="M3 5h4"/>
                    <path d="M17 19h4"/>
                </svg>
                <h3 class="liquid-card-title">AI Highlight Detection</h3>
                <p class="liquid-card-desc">Multimodal Gemini analysis pinpoints viral moments, dialogue hooks, and key topics from podcasts or streams.</p>
            </div>
        </div>
        <div class="liquid-card">
            <div>
                <svg class="liquid-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="6" cy="6" r="3"/>
                    <circle cx="6" cy="18" r="3"/>
                    <line x1="20" y1="4" x2="8.12" y2="15.88"/>
                    <line x1="14.47" y1="14.48" x2="20" y2="20"/>
                    <line x1="8.12" y1="8.12" x2="12" y2="12"/>
                </svg>
                <h3 class="liquid-card-title">Precision Clipping</h3>
                <p class="liquid-card-desc">Local FFmpeg engine extracts subclips with millisecond accuracy, preserving source quality without cloud transcoding.</p>
            </div>
        </div>
        <div class="liquid-card">
            <div>
                <svg class="liquid-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                    <rect width="18" height="14" x="3" y="5" rx="2" ry="2"/>
                    <path d="M7 15h4M15 15h2M7 11h2M13 11h4"/>
                </svg>
                <h3 class="liquid-card-title">Automatic Subtitles</h3>
                <p class="liquid-card-desc">Hardware-accelerated Faster-Whisper ASR generates aligned SRT captions and optional burned-in hardsubs.</p>
            </div>
        </div>
        <div class="liquid-card">
            <div>
                <svg class="liquid-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                    <rect width="14" height="20" x="5" y="2" rx="2" ry="2"/>
                    <path d="M12 18h.01"/>
                </svg>
                <h3 class="liquid-card-title">Vertical & Square Exports</h3>
                <p class="liquid-card-desc">Social reframing for 9:16 Shorts/TikTok and 1:1 Feed posts with Center Crop or stylized Blurred Background fill.</p>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# Feature Explanation Section (Two-Column Editorial)
st.markdown(
    """
    <div id="architecture"></div>
    <section class="editorial-section">
        <div class="editorial-left">
            <h2 class="editorial-heading">Designed for creators. Built for precision.</h2>
            <p class="editorial-desc">
                ClipForge pairs Google Gemini's multimodal reasoning with local FFmpeg media processing. You connect your own Gemini API key—ensuring complete control over model selection, privacy, and quotas without platform markups.
            </p>
            <ul class="editorial-list">
                <li class="editorial-item">
                    <span class="editorial-dot"></span>
                    <span><b>BYOK Architecture</b>: Direct Google API pricing, zero monthly middleman markups.</span>
                </li>
                <li class="editorial-item">
                    <span class="editorial-dot"></span>
                    <span><b>Local Media Pipeline</b>: Sub-clipping, subtitles, and reframing run 100% locally.</span>
                </li>
                <li class="editorial-item">
                    <span class="editorial-dot"></span>
                    <span><b>Immediate Cloud Purge</b>: Media sent to Google is deleted immediately after parsing.</span>
                </li>
                <li class="editorial-item">
                    <span class="editorial-dot"></span>
                    <span><b>Session Isolation</b>: Isolated per-user workspaces with automated TTL garbage collection.</span>
                </li>
            </ul>
            <div class="editorial-muted">
                PIPELINE: FFMPEG 9 · FASTER-WHISPER · GEMINI 3.6 FLASH · BYOK ENGINE
            </div>
        </div>
        <div class="editorial-graphic-panel">
            <div class="timeline-header">
                <span>TIMELINE MONITOR</span>
                <span>CH-1 / CH-2 STEREO</span>
            </div>
            <div class="timeline-track">
                <div class="waveform-bar" style="height: 35%;"></div>
                <div class="waveform-bar" style="height: 60%;"></div>
                <div class="waveform-bar" style="height: 45%;"></div>
                <div class="waveform-bar" style="height: 80%;"></div>
                <div class="waveform-bar" style="height: 95%;"></div>
                <div class="waveform-bar" style="height: 70%;"></div>
                <div class="waveform-bar" style="height: 85%;"></div>
                <div class="waveform-bar" style="height: 100%;"></div>
                <div class="waveform-bar" style="height: 75%;"></div>
                <div class="waveform-bar" style="height: 90%;"></div>
                <div class="waveform-bar" style="height: 65%;"></div>
                <div class="waveform-bar" style="height: 85%;"></div>
                <div class="waveform-bar" style="height: 50%;"></div>
                <div class="waveform-bar" style="height: 90%;"></div>
                <div class="waveform-bar" style="height: 100%;"></div>
                <div class="waveform-bar" style="height: 75%;"></div>
                <div class="waveform-bar" style="height: 60%;"></div>
                <div class="waveform-bar" style="height: 40%;"></div>
                <div class="waveform-bar" style="height: 70%;"></div>
                <div class="waveform-bar" style="height: 55%;"></div>
                <div class="waveform-bar" style="height: 85%;"></div>
                <div class="waveform-bar" style="height: 40%;"></div>
                <div class="waveform-bar" style="height: 25%;"></div>
                <div class="timeline-marker-region">
                    <span>CLIP HIGHLIGHT [00:42 — 01:18]</span>
                </div>
            </div>
            <div class="timeline-meta-row">
                <span>IN: 00:00:42.000</span>
                <span>DURATION: 36.00s</span>
                <span>OUT: 00:01:18.000</span>
            </div>
        </div>
    </section>
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

# Video Workspace Anchor
st.markdown('<div id="workspace"></div>', unsafe_allow_html=True)

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
        <div class="studio-card" style="text-align: center; padding: 48px 24px;">
            <div style="font-family: var(--cf-font-serif); font-size: 1.8rem; color: var(--cf-text-primary); margin-bottom: 8px;">
                Ready for Media Ingestion
            </div>
            <div style="color: var(--cf-text-secondary); font-size: 0.95rem; max-width: 540px; margin: 0 auto 24px auto; line-height: 1.6;">
                Drop an MP4, MOV, or MKV podcast or livestream file above to inspect audio/video tracks, discover viral moments with Gemini, and reframe for social platforms.
            </div>
            <div style="font-family: var(--cf-font-mono); font-size: 0.75rem; color: var(--cf-text-muted); letter-spacing: 0.06em;">
                ACCEPTED CODECS: H.264 · HEVC · VP9 · AV1 · AAC · OPUS · MAX 1024 MB
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Validate upload size bounds
    if uploaded_file.size <= 0:
        st.error("Uploaded video file is empty (0 bytes). Please upload a valid media file.", icon="⚠️")
        st.stop()

    max_upload_bytes = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
    if uploaded_file.size > max_upload_bytes:
        st.error(
            f"File size ({uploaded_file.size / (1024 * 1024):.1f} MB) exceeds maximum upload limit of {DEFAULT_MAX_UPLOAD_MB} MB.",
            icon="🚨",
        )
        st.stop()

    # Validate file extension
    file_suffix = Path(uploaded_file.name).suffix.lower()
    if file_suffix not in (".mp4", ".mov", ".mkv"):
        st.error(f"Unsupported file extension '{file_suffix}'. ClipForge supports MP4, MOV, and MKV files.", icon="🚨")
        st.stop()

    # Pre-flight disk space safeguard: ensure sufficient disk space before writing
    required_space = uploaded_file.size + DEFAULT_MIN_FREE_DISK_BYTES
    is_disk_ok, free_bytes, _ = check_disk_space_available(BASE_TEMP_DIR, min_free_bytes=required_space)
    if not is_disk_ok:
        # Run aggressive cleanup on older sessions to reclaim disk space
        cleanup_stale_files([BASE_TEMP_DIR, BASE_EXPORTS_DIR], max_age_seconds=1800, active_session_ids=[session_id])
        is_disk_ok, free_bytes, _ = check_disk_space_available(BASE_TEMP_DIR, min_free_bytes=required_space)
        if not is_disk_ok:
            st.error(
                f"Server storage threshold reached ({free_bytes / (1024*1024*1024):.1f} GB available, "
                f"need at least {required_space / (1024*1024*1024):.1f} GB). "
                "Please wait for background cleanup or upload a smaller video.",
                icon="⚠️",
            )
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

        # Video duration limit safeguard
        if metadata.duration_seconds > DEFAULT_MAX_VIDEO_DURATION_SECONDS:
            st.error(
                f"Video duration ({metadata.duration_formatted}) exceeds maximum studio limit of "
                f"{format_duration(DEFAULT_MAX_VIDEO_DURATION_SECONDS)}. "
                "Please upload a shorter clip or episode.",
                icon="⚠️",
            )
            st.stop()

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
                        try:
                            with acquire_job_slot(timeout=10.0):
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
                        except CapacityExceededError as err:
                            st.warning(f"⏳ {err}", icon="⏳")

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
                                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:14px;">
                                    <div style="font-family:var(--cf-font-serif); font-size:1.35rem; font-weight:400; color:var(--cf-text-primary);">
                                        <span style="opacity:0.6;">#{c_idx:02d}</span> {c_info.get('title')}
                                    </div>
                                    <div class="liquid-badge">
                                        DURATION: {c_info.get('duration'):.1f}s
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
                <div class="liquid-badge" style="display:inline-block; margin-bottom:12px;">
                    SELECTED DURATION: {clip_duration:.2f}s ({format_duration(clip_duration)})
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
                    with acquire_job_slot(timeout=10.0):
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
                except CapacityExceededError as err:
                    st.warning(f"⏳ {err}", icon="⏳")
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
