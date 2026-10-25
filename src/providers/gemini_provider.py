"""Google Gemini AI highlight provider adapter for ClipForge using official google-genai SDK."""

import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Optional

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

from src.providers.base import (
    BaseAIProvider,
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

RECOMMENDED_MODELS_ORDER = [
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-8b",
    "gemini-2.5-pro",
    "gemini-2.0-pro",
    "gemini-1.5-pro",
]

# Regular expression pattern to detect and redact API keys and query tokens
_SECRET_PATTERN = re.compile(
    r"(?:AIza[0-9A-Za-z_-]{30,}|AQ\.[0-9A-Za-z_-]{15,}|(?:key|api_key|token)=[0-9A-Za-z_-]{15,})"
)


def sanitize_secret_string(text: str, user_key: Optional[str] = None) -> str:
    """Ensure no API keys, query tokens, or credentials appear in logs or errors."""
    if not text:
        return ""
    sanitized = str(text)
    if user_key and len(user_key.strip()) >= 6:
        sanitized = sanitized.replace(user_key.strip(), "[REDACTED_API_KEY]")
    sanitized = _SECRET_PATTERN.sub("[REDACTED_KEY]", sanitized)
    return sanitized


def normalize_model_name(model_name: str) -> str:
    """Normalize model name by stripping 'models/' prefix and whitespace."""
    if not model_name:
        return ""
    return model_name.strip().removeprefix("models/").strip()


def _log_stage(msg: str, callback: Optional[Callable[[str], None]] = None) -> None:
    """Print flushed progress message to terminal and trigger UI callback if provided."""
    safe_msg = sanitize_secret_string(msg)
    print(f"[ClipForge] {safe_msg}", flush=True)
    if callback:
        try:
            callback(safe_msg)
        except Exception:
            pass


class GeminiHighlightItem(BaseModel):
    """Pydantic schema for single highlight item expected from Gemini."""
    title: str = Field(description="Short descriptive title of the moment (3-6 words)")
    start_time: float = Field(description="Start timestamp in seconds from the beginning")
    end_time: float = Field(description="End timestamp in seconds from the beginning")
    hook: str = Field(description="The opening spoken hook or compelling opening phrase")
    explanation: str = Field(description="Why this segment is an engaging, self-contained moment")
    confidence_score: float = Field(default=0.9, description="Confidence score from 0.0 to 1.0")


class GeminiHighlightCollection(BaseModel):
    """Pydantic schema for structured output collection."""
    highlights: list[GeminiHighlightItem] = Field(description="List of detected highlight moments")


class GoogleGeminiProvider(BaseAIProvider):
    """
    Adapter for Google Gemini API via the official Google Gen AI SDK.
    """

    @property
    def provider_id(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Google Gemini"

    @property
    def supported_models(self) -> list[str]:
        """Return fallback list of known models when dynamic discovery has not occurred."""
        return list(RECOMMENDED_MODELS_ORDER)

    @property
    def default_model(self) -> str:
        """Default fallback model if discovery has not yet executed."""
        return "gemini-3.6-flash"

    def choose_default_model(self, available_models: list[str]) -> str:
        """
        Pick a sensible, cost-effective default model from the available models list.
        Prefers fast, economical models (flash series) over large/costly ones.
        Does not assume any specific model exists.
        """
        if not available_models:
            return self.default_model

        avail_map = {normalize_model_name(m).lower(): normalize_model_name(m) for m in available_models}

        for pref in RECOMMENDED_MODELS_ORDER:
            pref_l = pref.lower()
            if pref_l in avail_map:
                return avail_map[pref_l]
            for m_l, orig in avail_map.items():
                if m_l.startswith(pref_l):
                    return orig

        return normalize_model_name(available_models[0])

    def list_available_models(
        self,
        api_key: str,
        timeout_secs: float = 10.0,
    ) -> list[str]:
        """
        Dynamically query Google Gemini API to discover models available to this API key.

        Filters models to only include those supporting:
        1. Content generation ('generateContent' in supported_actions)
        2. Multimodal input (audio/video) - i.e. Gemini series models.
        Excludes embedding models, text-only legacy PaLM/Bison models, image-only Imagen models.
        Strips 'models/' prefix for clean UI presentation and SDK compatibility.
        """
        valid_syntax, syntax_msg = self.validate_api_key_format(api_key)
        if not valid_syntax:
            raise InvalidAPIKeyError(syntax_msg)

        clean = api_key.strip()
        timeout_ms = int(max(1.0, timeout_secs) * 1000)

        try:
            client = genai.Client(
                api_key=clean,
                http_options=types.HttpOptions(
                    timeout=timeout_ms,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            discovered_models: list[str] = []
            pager = client.models.list()
            for m in pager:
                raw_name = getattr(m, "name", "") or ""
                clean_name = normalize_model_name(raw_name)
                if not clean_name:
                    continue

                supported_actions = getattr(m, "supported_actions", None) or []
                if supported_actions and "generateContent" not in supported_actions:
                    continue

                name_lower = clean_name.lower()
                if not name_lower.startswith("gemini"):
                    continue

                if "embedding" in name_lower or "imagen" in name_lower or "aqa" in name_lower:
                    continue

                if clean_name not in discovered_models:
                    discovered_models.append(clean_name)

            if not discovered_models:
                raise ProviderError("No compatible multimodal Gemini models found for this API key.")

            # Sort/order: prioritized models first, then alphabetical for the rest
            def _sort_key(name: str):
                name_l = name.lower()
                for rank, pref in enumerate(RECOMMENDED_MODELS_ORDER):
                    if name_l == pref or name_l.startswith(pref):
                        return (0, rank)
                return (1, name_l)

            discovered_models.sort(key=_sort_key)
            return discovered_models

        except (httpx.TimeoutException, TimeoutError) as exc:
            raise ProviderError(
                f"Connection timed out: Unable to retrieve Gemini models from Google within {timeout_secs:.0f}s. "
                "Please check your internet connection."
            ) from exc
        except httpx.NetworkError as exc:
            raise ProviderError(
                "Network connection error: Unable to reach Google servers to list models. "
                "Please check your network connection, proxy, or DNS settings."
            ) from exc
        except errors.APIError as exc:
            msg_lower = str(exc).lower()
            code = getattr(exc, "code", 0)
            if (
                code in (400, 401, 403)
                or "api_key_invalid" in msg_lower
                or "api key not valid" in msg_lower
                or "invalid_argument" in msg_lower
            ):
                raise InvalidAPIKeyError(
                    "Invalid API key or unauthorized credentials. Verify your key at https://aistudio.google.com/apikey"
                ) from exc
            elif code == 429 or "resource_exhausted" in msg_lower or "quota" in msg_lower:
                raise QuotaExceededError("Rate limit or quota exhausted for this API key while querying models.") from exc
            safe_detail = sanitize_secret_string(exc.message if hasattr(exc, 'message') else str(exc), api_key)
            raise ProviderError(f"Google Gemini API error when retrieving models (Code {code}): {safe_detail}") from exc
        except Exception as exc:
            if isinstance(exc, (InvalidAPIKeyError, QuotaExceededError, ProviderError)):
                raise
            raise ProviderError(f"Failed to list Gemini models: {sanitize_secret_string(str(exc), api_key)}") from exc

    def verify_model_access(
        self,
        api_key: str,
        model_name: str,
        timeout_secs: float = 10.0,
    ) -> tuple[bool, str]:
        """
        Verify that a specific chosen model is accessible and usable with this API key.
        Handles both 'models/name' and 'name' formats.
        """
        valid_syntax, syntax_msg = self.validate_api_key_format(api_key)
        if not valid_syntax:
            return False, syntax_msg

        clean_key = api_key.strip()
        clean_model = normalize_model_name(model_name)
        if not clean_model:
            return False, "Model name cannot be empty."

        timeout_ms = int(max(1.0, timeout_secs) * 1000)

        try:
            client = genai.Client(
                api_key=clean_key,
                http_options=types.HttpOptions(
                    timeout=timeout_ms,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            _ = client.models.get(model=clean_model)
            return True, f"Model '{clean_model}' is accessible."
        except errors.APIError as exc:
            code = getattr(exc, "code", 0)
            if code == 404:
                return False, f"Model '{clean_model}' was not found or is not enabled for your account."
            elif code in (400, 401, 403):
                return False, f"Unauthorized access to model '{clean_model}'."
            elif code == 429:
                return False, f"Rate limit or quota exhausted for model '{clean_model}'."
            return False, f"Google Gemini API error for model '{clean_model}' (Code {code})."
        except (httpx.TimeoutException, TimeoutError):
            return False, f"Verification of model '{clean_model}' timed out after {timeout_secs:.0f} seconds."
        except httpx.NetworkError:
            return False, f"Network error when verifying model '{clean_model}'."
        except Exception as exc:
            return False, f"Failed to verify model '{clean_model}': {exc}"

    def validate_api_key_format(self, api_key: str) -> tuple[bool, str]:
        """
        Verify syntactic format of a Google Gemini API key.

        Supports both traditional formats (e.g. 'AIza...') and newer key formats
        (e.g. 'AQ...'), as well as general Google Developer API credentials.
        Does not enforce arbitrary prefix restrictions. Never logs or prints the key.
        """
        import re

        if not api_key:
            return False, "API key cannot be empty."

        clean = api_key.strip()
        if not clean:
            return False, "API key cannot consist solely of whitespace."

        if re.search(r"\s", clean):
            return False, "API key must not contain spaces, tabs, or newlines."

        if len(clean) < 20:
            return False, f"API key is too short ({len(clean)} characters). Valid Gemini keys are typically 30+ characters."

        # Valid Google API keys are alphanumeric with standard URL-safe symbols (- _ .)
        if not re.match(r"^[A-Za-z0-9_.-]+$", clean):
            return False, "API key contains invalid characters. Only alphanumeric characters, hyphens, underscores, and dots are permitted."

        return True, "API key syntax is valid."

    def verify_api_key(
        self,
        api_key: str,
        check_live: bool = True,
        timeout_secs: float = 10.0,
    ) -> tuple[bool, str]:
        """
        Validate and optionally authenticate an API key against Google's API endpoint.

        Performs a lightweight, read-only metadata probe (client.models.get) with an explicit
        timeout (converted to milliseconds for Google Gen AI SDK) and retry_options=HttpRetryOptions(attempts=1)
        to prevent infinite or unwanted retries. Never logs or prints the API key.
        """
        valid_syntax, syntax_msg = self.validate_api_key_format(api_key)
        if not valid_syntax:
            _log_stage(f"API key syntax check failed: {syntax_msg}")
            return False, syntax_msg

        if not check_live:
            return True, syntax_msg

        _log_stage("Verifying live API key authentication with Google servers...")
        clean = api_key.strip()
        timeout_ms = int(max(1.0, timeout_secs) * 1000)

        try:
            client = genai.Client(
                api_key=clean,
                http_options=types.HttpOptions(
                    timeout=timeout_ms,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            # Lightweight probe: list 1 model to verify credentials without assuming any model name exists
            pager = client.models.list(config=types.ListModelsConfig(page_size=1))
            for _ in pager:
                break
            _log_stage("API key authenticated successfully with Google Gemini.")
            return True, "API key verified successfully with Google Gemini."
        except (httpx.TimeoutException, TimeoutError) as exc:
            err_msg = (
                f"Connection timed out: Google servers did not respond within {timeout_secs:.0f} seconds. "
                "Please check your internet connection and try again."
            )
            _log_stage(f"Authentication timeout: {err_msg}")
            return False, err_msg
        except httpx.NetworkError as exc:
            err_msg = (
                "Network connection error: Unable to reach Google servers. "
                "Please check your network connection, proxy, or DNS settings."
            )
            _log_stage(f"Authentication network error: {err_msg}")
            return False, err_msg
        except errors.APIError as exc:
            msg_lower = str(exc).lower()
            code = getattr(exc, "code", 0)
            if (
                code in (400, 401, 403)
                or "api_key_invalid" in msg_lower
                or "api key not valid" in msg_lower
                or "invalid_argument" in msg_lower
            ):
                err_msg = (
                    "Google Gemini authentication failed: Invalid API key or unauthorized credentials. "
                    "Verify your key at https://aistudio.google.com/apikey"
                )
            elif code == 429 or "resource_exhausted" in msg_lower or "quota" in msg_lower:
                err_msg = "Google Gemini authentication failed: Rate limit or quota exhausted for this API key."
            elif code == 404:
                err_msg = f"Default model '{self.default_model}' is not accessible or not found for this account."
            else:
                safe_api_msg = sanitize_secret_string(exc.message if hasattr(exc, 'message') else 'Authentication failed', api_key)
                err_msg = f"Google Gemini API error (Code {code}): {safe_api_msg}"
            _log_stage(f"Authentication failed: {err_msg}")
            return False, err_msg
        except Exception as exc:
            err_str = sanitize_secret_string(str(exc), api_key)
            if "timeout" in err_str.lower() or "timed out" in err_str.lower():
                err_msg = (
                    f"Connection timed out: Google servers did not respond within {timeout_secs:.0f} seconds. "
                    "Please check your internet connection and try again."
                )
            else:
                err_msg = f"Failed to authenticate API key with Google Gemini: {err_str}"
            _log_stage(f"Authentication error: {err_msg}")
            return False, err_msg

    def get_model_capabilities(self, model_name: str) -> set[ModelCapability]:
        """Return capabilities for the selected Gemini model."""
        clean_name = normalize_model_name(model_name).lower()
        if not clean_name.startswith("gemini"):
            raise ModelNotFoundError(f"Model '{model_name}' is not recognized as a Gemini multimodal model.")

        capabilities = {
            ModelCapability.VIDEO_UNDERSTANDING,
            ModelCapability.AUDIO_UNDERSTANDING,
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.LARGE_CONTEXT_WINDOW,
        }
        if "flash" in clean_name or "lite" in clean_name:
            capabilities.add(ModelCapability.LOW_LATENCY)
        return capabilities

    def resolve_input_strategy(
        self,
        model_name: str,
        file_size_bytes: int,
        duration_secs: float,
    ) -> InputModality:
        """
        Choose between native video and extracted audio.

        For large files (> 30 MB or > 5 min), extracting audio via FFmpeg significantly reduces
        upload latency, token cost, and network bandwidth while retaining full verbal context.
        """
        mb = file_size_bytes / (1024 * 1024)
        if mb <= 30.0 and duration_secs <= 300.0:
            return InputModality.NATIVE_VIDEO
        return InputModality.AUDIO_EXTRACTED

    def build_analysis_prompt(
        self,
        target_clip_count: int,
        target_clip_duration_secs: float,
        video_duration_secs: float,
        custom_instructions: Optional[str] = None,
    ) -> str:
        """Construct the prompt instructing Gemini to identify self-contained engaging moments."""
        extra = f"\nUser Focus: {custom_instructions}" if custom_instructions else ""
        return f"""You are an expert video editor and podcast producer.
Analyze this media content and identify exactly {target_clip_count} engaging, high-value highlight moments suitable for standalone short vertical clips.

Constraints:
1. Target clip duration: around {target_clip_duration_secs:.0f} seconds (acceptable range: 20 to 90 seconds).
2. Total source video length: {video_duration_secs:.2f} seconds.
3. Every moment MUST be self-contained: start at the beginning of an idea/question and end after the payoff/conclusion. Never cut mid-sentence.
4. Prioritize:
   - Engaging stories and personal anecdotes
   - Practical, actionable advice or takeaways
   - Humorous or surprising revelations
   - Strong conversational hooks
5. Do NOT promise or score based on 'virality' predictions. Score purely on narrative completeness and interest.
{extra}
"""

    def _extract_audio_for_upload(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Path:
        """Extract optimized audio using local FFmpeg for faster upload and lower token consumption."""
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        out_audio = temp_dir / f"gemini_audio_{video_path.stem}_{int(time.time())}.m4a"

        _log_stage(f"Extracting 16kHz audio from '{video_path.name}' via FFmpeg...", progress_callback)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "16000",
            str(out_audio),
        ]
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=60,
            )
            if res.returncode != 0:
                raise ProviderError(f"Local audio extraction failed: {res.stderr.strip()}")

            size_mb = out_audio.stat().st_size / (1024 * 1024)
            _log_stage(f"Audio extracted successfully: {size_mb:.2f} MB", progress_callback)
            return out_audio
        except subprocess.TimeoutExpired as exc:
            raise ProviderError("FFmpeg audio extraction timed out after 60 seconds.") from exc
        except Exception as exc:
            raise ProviderError(f"Error launching FFmpeg for audio extraction: {exc}") from exc

    def analyze_highlights(
        self,
        request: HighlightRequest,
        api_key: str,
        model_name: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> HighlightResponse:
        """
        Execute AI analysis against Google Gemini API using official Google Gen AI SDK.

        - Flushed terminal logging and UI callback updates at every step.
        - Discloses media upload to user.
        - Uses structured Pydantic schema for guaranteed format.
        - Deletes provider-side uploaded file immediately after generation.
        """
        chosen_model = normalize_model_name(model_name or self.default_model)
        if not chosen_model.lower().startswith("gemini"):
            raise ModelNotFoundError(f"Model '{chosen_model}' is not recognized as a Gemini multimodal model.")

        is_valid, msg = self.validate_api_key_format(api_key)
        if not is_valid:
            raise InvalidAPIKeyError(msg)

        strategy = request.preferred_modality or self.resolve_input_strategy(
            chosen_model,
            request.file_size_bytes,
            request.video_duration,
        )

        _log_stage(f"Starting highlight analysis with Gemini '{chosen_model}' (Strategy: {strategy.value.upper()})", progress_callback)

        media_path = request.video_path
        temp_audio_path: Optional[Path] = None

        if strategy == InputModality.AUDIO_EXTRACTED:
            temp_audio_path = self._extract_audio_for_upload(media_path, progress_callback)
            upload_target = temp_audio_path
        else:
            upload_target = media_path
            _log_stage(f"Using native video for upload: {upload_target.name} ({request.file_size_bytes / (1024*1024):.2f} MB)", progress_callback)

        if not upload_target.is_file():
            raise FileNotFoundError(f"Media file to upload not found: {upload_target}")

        uploaded_file = None
        client = None

        try:
            upload_size_mb = upload_target.stat().st_size / (1024 * 1024)
            _log_stage(f"Connecting to Google GenAI Client (timeout=180s)...", progress_callback)

            client = genai.Client(
                api_key=api_key.strip(),
                http_options=types.HttpOptions(
                    timeout=180000,
                    retry_options=types.HttpRetryOptions(attempts=2),
                ),
            )

            _log_stage(f"Uploading '{upload_target.name}' ({upload_size_mb:.2f} MB) to Google GenAI Files API...", progress_callback)
            uploaded_file = client.files.upload(file=str(upload_target))
            _log_stage(f"Upload completed. Remote file name: {uploaded_file.name}", progress_callback)

            # Poll until file is processed if it enters PROCESSING state
            poll_seconds = 0
            while getattr(uploaded_file, "state", None) and uploaded_file.state.name == "PROCESSING":
                _log_stage(f"Waiting for Google servers to process media... ({poll_seconds}s elapsed)", progress_callback)
                if poll_seconds >= 120:
                    raise ProviderError("Media processing timed out on Google servers after 120 seconds.")
                time.sleep(4)
                poll_seconds += 4
                uploaded_file = client.files.get(name=uploaded_file.name)

            if getattr(uploaded_file, "state", None) and uploaded_file.state.name == "FAILED":
                fail_err = getattr(uploaded_file, "error", "Unknown processing failure")
                raise ProviderError(f"Google processing failed for uploaded media: {fail_err}")

            _log_stage(f"Media file ready on Google servers (state={getattr(uploaded_file.state, 'name', 'ACTIVE')}).", progress_callback)

            prompt = self.build_analysis_prompt(
                target_clip_count=request.target_clip_count,
                target_clip_duration_secs=request.target_clip_duration_secs,
                video_duration_secs=request.video_duration,
                custom_instructions=request.custom_instructions,
            )

            # Generate structured content
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiHighlightCollection,
                temperature=0.2,
            )

            _log_stage(f"Sending prompt to Gemini {chosen_model} (requesting {request.target_clip_count} clips)...", progress_callback)
            response = client.models.generate_content(
                model=chosen_model,
                contents=[uploaded_file, prompt],
                config=config,
            )
            _log_stage("Gemini response received. Parsing structured output...", progress_callback)

            candidates: list[HighlightCandidate] = []
            raw_text = getattr(response, "text", "") or ""

            # Parse structured output from response.parsed or response.text
            parsed_data = getattr(response, "parsed", None)
            if parsed_data and hasattr(parsed_data, "highlights"):
                raw_items = parsed_data.highlights
            else:
                try:
                    data_dict = json.loads(raw_text)
                    raw_items = [
                        GeminiHighlightItem(**item)
                        for item in data_dict.get("highlights", [])
                    ]
                except Exception as exc:
                    raise ProviderError(f"Failed to parse structured JSON from Gemini: {exc}") from exc

            for item in raw_items:
                start_val = round(float(item.start_time), 2)
                end_val = round(float(item.end_time), 2)
                dur_val = round(end_val - start_val, 2)

                cand = HighlightCandidate(
                    title=item.title.strip() or "Untitled Highlight",
                    start_time=start_val,
                    end_time=end_val,
                    duration=dur_val,
                    explanation=item.explanation.strip(),
                    hook=item.hook.strip(),
                    confidence_score=getattr(item, "confidence_score", 0.9),
                )

                # Validate timestamps against source duration and minimum 5.0s threshold
                is_valid_cand, err_msg = cand.validate(max_duration=request.video_duration)
                if is_valid_cand:
                    candidates.append(cand)
                else:
                    _log_stage(f"Filtered out invalid candidate '{cand.title}' ({cand.start_time:.1f}s - {cand.end_time:.1f}s): {err_msg}", progress_callback)

            _log_stage(f"Analysis completed: {len(candidates)} valid highlight moments retained.", progress_callback)

            usage_meta = {}
            if getattr(response, "usage_metadata", None):
                usage_meta = {
                    "prompt_token_count": getattr(response.usage_metadata, "prompt_token_count", None),
                    "candidates_token_count": getattr(response.usage_metadata, "candidates_token_count", None),
                    "total_token_count": getattr(response.usage_metadata, "total_token_count", None),
                }

            return HighlightResponse(
                provider_id=self.provider_id,
                model_name=chosen_model,
                highlights=candidates,
                input_modality_used=strategy,
                usage_metadata=usage_meta,
                raw_response=raw_text,
            )

        except errors.APIError as exc:
            msg_lower = str(exc).lower()
            code = getattr(exc, "code", 0)

            if code in (400, 401, 403) or "api_key_invalid" in msg_lower or "api key not valid" in msg_lower:
                safe_detail = sanitize_secret_string(exc.message if hasattr(exc, "message") else str(exc), api_key)
                err_msg = f"Google Gemini rejected the provided API key (Code {code}): {safe_detail}"
                _log_stage(f"Error: {err_msg}", progress_callback)
                raise InvalidAPIKeyError(err_msg) from exc
            elif code == 429 or "resource_exhausted" in msg_lower or "quota" in msg_lower:
                safe_detail = sanitize_secret_string(exc.message if hasattr(exc, "message") else str(exc), api_key)
                err_msg = f"Gemini API quota or rate limit exceeded: {safe_detail}"
                _log_stage(f"Error: {err_msg}", progress_callback)
                raise QuotaExceededError(err_msg) from exc
            elif code == 404 or "models/" in msg_lower:
                safe_detail = sanitize_secret_string(str(exc), api_key)
                err_msg = f"Selected Gemini model '{chosen_model}' was not found or is unavailable: {safe_detail}"
                _log_stage(f"Error: {err_msg}", progress_callback)
                raise ModelNotFoundError(err_msg) from exc
            else:
                safe_detail = sanitize_secret_string(str(exc), api_key)
                err_msg = f"Google Gemini API error (Code {code}): {safe_detail}"
                _log_stage(f"Error: {err_msg}", progress_callback)
                raise ProviderError(err_msg) from exc

        except (InvalidAPIKeyError, QuotaExceededError, ModelNotFoundError, ProviderError):
            raise

        except Exception as exc:
            safe_detail = sanitize_secret_string(str(exc), api_key)
            err_msg = f"Unexpected error communicating with Gemini API: {safe_detail}"
            _log_stage(f"Error: {err_msg}", progress_callback)
            raise ProviderError(err_msg) from exc

        finally:
            # Responsibly clean up the provider-side uploaded file
            if client is not None and uploaded_file is not None and hasattr(uploaded_file, "name"):
                try:
                    _log_stage(f"Cleaning up remote file on Google servers ({uploaded_file.name})...", progress_callback)
                    client.files.delete(name=uploaded_file.name)
                    _log_stage("Remote cleanup complete.", progress_callback)
                except Exception as del_exc:
                    _log_stage(f"Notice: Failed to delete remote file: {del_exc}", progress_callback)

            # Clean up local scratch audio
            if temp_audio_path and temp_audio_path.exists():
                try:
                    temp_audio_path.unlink()
                except Exception:
                    pass
