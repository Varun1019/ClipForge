"""Unit tests for ClipForge AI provider interface, Gemini adapter, and mocked API responses."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from google.genai import errors

from src.providers.base import (
    HighlightCandidate,
    HighlightRequest,
    InputModality,
    InvalidAPIKeyError,
    ModelCapability,
    ModelNotFoundError,
    ProviderError,
    QuotaExceededError,
)
from src.providers.gemini_provider import (
    GeminiHighlightCollection,
    GeminiHighlightItem,
    GoogleGeminiProvider,
)
from src.providers.registry import ProviderRegistry


class TestProviders(unittest.TestCase):
    def setUp(self):
        self.provider = GoogleGeminiProvider()
        self.registry = ProviderRegistry()
        self.test_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.test_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_highlight_candidate_validation(self):
        """Test timestamp bounds checking on AI proposed moments."""
        # Valid moment
        valid = HighlightCandidate(
            title="Great Story",
            start_time=10.0,
            end_time=45.0,
            duration=35.0,
            explanation="Cohesive story with punchline",
            hook="You won't believe what happened next",
        )
        is_ok, err = valid.validate(max_duration=60.0)
        self.assertTrue(is_ok)
        self.assertIsNone(err)

        # Negative start
        neg_start = HighlightCandidate("Bad", -1.0, 10.0, 11.0, "exp", "hook")
        is_ok, err = neg_start.validate(max_duration=60.0)
        self.assertFalse(is_ok)
        self.assertIn("cannot be negative", err)

        # End before start
        end_before = HighlightCandidate("Bad", 30.0, 20.0, 10.0, "exp", "hook")
        is_ok, err = end_before.validate(max_duration=60.0)
        self.assertFalse(is_ok)
        self.assertIn("must be after start time", err)

        # Too short (< 5s)
        too_short = HighlightCandidate("Short", 10.0, 13.0, 3.0, "exp", "hook")
        is_ok, err = too_short.validate(max_duration=60.0)
        self.assertFalse(is_ok)
        self.assertIn("too short", err)

        # Exceeds max duration
        out_of_bounds = HighlightCandidate("Bad", 10.0, 75.0, 65.0, "exp", "hook")
        is_ok, err = out_of_bounds.validate(max_duration=60.0)
        self.assertFalse(is_ok)
        self.assertIn("exceeds video duration", err)

    def test_gemini_provider_metadata(self):
        """Verify Gemini provider properties and supported modern models."""
        self.assertEqual(self.provider.provider_id, "gemini")
        self.assertEqual(self.provider.display_name, "Google Gemini")
        self.assertEqual(self.provider.default_model, "gemini-3.6-flash")
        self.assertIn("gemini-3.6-flash", self.provider.supported_models)
        self.assertIn("gemini-2.5-flash", self.provider.supported_models)
        self.assertIn("gemini-2.5-pro", self.provider.supported_models)
        self.assertIn("gemini-2.0-flash", self.provider.supported_models)

    def test_gemini_capabilities(self):
        """Verify model capabilities query."""
        caps = self.provider.get_model_capabilities("gemini-2.5-flash")
        self.assertIn(ModelCapability.VIDEO_UNDERSTANDING, caps)
        self.assertIn(ModelCapability.AUDIO_UNDERSTANDING, caps)
        self.assertIn(ModelCapability.STRUCTURED_OUTPUT, caps)

        with self.assertRaises(ModelNotFoundError):
            self.provider.get_model_capabilities("obsolete-model-v1")

    def test_gemini_api_key_syntax_validation(self):
        """Test API key format verification without arbitrary prefix lock."""
        # Empty and whitespace
        valid, msg = self.provider.validate_api_key_format("")
        self.assertFalse(valid)
        self.assertIn("empty", msg)

        valid, msg = self.provider.validate_api_key_format("   ")
        self.assertFalse(valid)
        self.assertIn("whitespace", msg)

        # Embedded whitespace / newlines
        valid, msg = self.provider.validate_api_key_format("AIzaSy 1234567890 1234567890")
        self.assertFalse(valid)
        self.assertIn("spaces", msg)

        valid, msg = self.provider.validate_api_key_format("AQ.1234567890\n1234567890")
        self.assertFalse(valid)
        self.assertIn("newlines", msg)

        # Too short (< 20 chars)
        valid, msg = self.provider.validate_api_key_format("short_key_123")
        self.assertFalse(valid)
        self.assertIn("too short", msg)

        # Invalid characters
        valid, msg = self.provider.validate_api_key_format("AIzaSy!@#$%^&*()_+1234567890")
        self.assertFalse(valid)
        self.assertIn("invalid characters", msg)

        # Traditional AIza prefix format
        valid, msg = self.provider.validate_api_key_format("AIzaSyD-abc12345678901234567890123456")
        self.assertTrue(valid)

        # Newer AQ. prefix format
        valid, msg = self.provider.validate_api_key_format("AQ.ABcdEfG12345-XyZ_9876543210.sampleKey")
        self.assertTrue(valid)

        # General valid key format without AIza
        valid, msg = self.provider.validate_api_key_format("genai_key_format_1234567890_abcdefg")
        self.assertTrue(valid)

    @patch("src.providers.gemini_provider.genai.Client")
    def test_gemini_verify_api_key_live_mock(self, mock_client_cls):
        """Test live verification using mocked client without real network calls."""
        import httpx

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.list.return_value = [MagicMock(name="models/gemini-2.0-flash")]

        # Success with newer AQ format
        valid, msg = self.provider.verify_api_key("AQ.TestKey12345678901234567890", check_live=True)
        self.assertTrue(valid)
        self.assertIn("verified successfully", msg)
        mock_client.models.list.assert_called()

        # Verify client initialized with timeout in ms (10000ms = 10s) and retry_options.attempts = 1
        _, kwargs = mock_client_cls.call_args
        self.assertEqual(kwargs["http_options"].timeout, 10000)
        self.assertEqual(kwargs["http_options"].retry_options.attempts, 1)

        # Custom timeout sets milliseconds accordingly
        self.provider.verify_api_key("AQ.TestKey12345678901234567890", check_live=True, timeout_secs=5.0)
        _, kwargs = mock_client_cls.call_args
        self.assertEqual(kwargs["http_options"].timeout, 5000)

        # Failure: 400 Invalid argument / API_KEY_INVALID (rejected credentials)
        mock_client.models.list.side_effect = errors.ClientError(400, {"error": {"message": "API key not valid. Please pass a valid API key.", "reason": "API_KEY_INVALID"}})
        valid, msg = self.provider.verify_api_key("AQ.BadKey12345678901234567890", check_live=True)
        self.assertFalse(valid)
        self.assertIn("Invalid API key", msg)

        # Failure: 401 unauthenticated
        mock_client.models.list.side_effect = errors.APIError(401, {"error": {"message": "Unauthorized"}})
        valid, msg = self.provider.verify_api_key("AQ.BadKey12345678901234567890", check_live=True)
        self.assertFalse(valid)
        self.assertIn("Invalid API key", msg)

        # Failure: 429 quota exhausted
        mock_client.models.list.side_effect = errors.APIError(429, {"error": {"message": "Quota exceeded"}})
        valid, msg = self.provider.verify_api_key("AIzaSyValidKey12345678901234567890", check_live=True)
        self.assertFalse(valid)
        self.assertIn("quota", msg.lower())

        # Failure: Connection timeout (server cannot be reached within limit)
        mock_client.models.list.side_effect = httpx.ConnectTimeout("Connection timed out")
        valid, msg = self.provider.verify_api_key("AIzaSyValidKey12345678901234567890", check_live=True, timeout_secs=10.0)
        self.assertFalse(valid)
        self.assertIn("timed out", msg.lower())
        self.assertIn("check your internet connection", msg.lower())

        # Failure: Network connectivity exception
        mock_client.models.list.side_effect = httpx.NetworkError("Failed to resolve host")
        valid, msg = self.provider.verify_api_key("AIzaSyValidKey12345678901234567890", check_live=True)
        self.assertFalse(valid)
        self.assertIn("network connection error", msg.lower())

    @patch("src.providers.gemini_provider.genai.Client")
    def test_list_available_models_success_and_filtering(self, mock_client_cls):
        """Test dynamic model discovery filters out non-generative and non-multimodal models."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        m_flash20 = MagicMock()
        m_flash20.name = "models/gemini-2.0-flash"
        m_flash20.supported_actions = ["generateContent", "countTokens"]

        m_pro15 = MagicMock()
        m_pro15.name = "models/gemini-1.5-pro"
        m_pro15.supported_actions = ["generateContent"]

        m_embed = MagicMock()
        m_embed.name = "models/text-embedding-004"
        m_embed.supported_actions = ["embedContent"]

        m_imagen = MagicMock()
        m_imagen.name = "models/imagen-3.0"
        m_imagen.supported_actions = ["imageGeneration"]

        m_bison = MagicMock()
        m_bison.name = "models/chat-bison-001"
        m_bison.supported_actions = ["generateContent"]

        m_aqa = MagicMock()
        m_aqa.name = "models/aqa"
        m_aqa.supported_actions = ["generateAnswer"]

        mock_client.models.list.return_value = [
            m_embed, m_flash20, m_imagen, m_pro15, m_bison, m_aqa
        ]

        models = self.provider.list_available_models("AIzaSyValidKey12345678901234567890")
        # Should only include gemini-2.0-flash and gemini-1.5-pro with models/ stripped
        self.assertIn("gemini-2.0-flash", models)
        self.assertIn("gemini-1.5-pro", models)
        self.assertNotIn("text-embedding-004", models)
        self.assertNotIn("imagen-3.0", models)
        self.assertNotIn("chat-bison-001", models)
        self.assertNotIn("aqa", models)
        # Check ordering: gemini-2.0-flash precedes gemini-1.5-pro
        self.assertEqual(models[0], "gemini-2.0-flash")

    def test_choose_default_model(self):
        """Test default model selection preferences without assuming any specific model exists."""
        # 0. When gemini-3.6-flash is available, it is prioritized
        default0 = self.provider.choose_default_model(["gemini-2.0-flash", "gemini-3.6-flash", "gemini-1.5-pro"])
        self.assertEqual(default0, "gemini-3.6-flash")

        # 1. When 3.6 & 2.5 Flash are missing but 2.0 Flash is present
        default1 = self.provider.choose_default_model(["gemini-1.5-pro", "gemini-2.0-flash"])
        self.assertEqual(default1, "gemini-2.0-flash")

        # 2. When only 1.5 Flash is available
        default2 = self.provider.choose_default_model(["gemini-1.5-flash", "gemini-1.5-pro"])
        self.assertEqual(default2, "gemini-1.5-flash")

        # 3. When custom model is available
        default3 = self.provider.choose_default_model(["gemini-experimental-custom"])
        self.assertEqual(default3, "gemini-experimental-custom")

        # 4. Handles models/ prefix in input list
        default4 = self.provider.choose_default_model(["models/gemini-2.0-flash-lite"])
        self.assertEqual(default4, "gemini-2.0-flash-lite")

    @patch("src.providers.gemini_provider.genai.Client")
    def test_list_available_models_api_error_and_timeout(self, mock_client_cls):
        """Test error handling in dynamic model discovery."""
        import httpx
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Timeout
        mock_client.models.list.side_effect = httpx.ConnectTimeout("Timed out")
        with self.assertRaises(ProviderError) as ctx:
            self.provider.list_available_models("AIzaSyValidKey12345678901234567890")
        self.assertIn("timed out", str(ctx.exception).lower())

        # Invalid key (400)
        mock_client.models.list.side_effect = errors.ClientError(400, {"error": {"message": "API key not valid."}})
        with self.assertRaises(InvalidAPIKeyError):
            self.provider.list_available_models("AIzaSyValidKey12345678901234567890")

        # Quota exceeded (429)
        mock_client.models.list.side_effect = errors.APIError(429, {"error": {"message": "Resource exhausted"}})
        with self.assertRaises(QuotaExceededError):
            self.provider.list_available_models("AIzaSyValidKey12345678901234567890")

        # Empty result
        mock_client.models.list.side_effect = None
        mock_client.models.list.return_value = []
        with self.assertRaises(ProviderError) as ctx:
            self.provider.list_available_models("AIzaSyValidKey12345678901234567890")
        self.assertIn("no compatible multimodal", str(ctx.exception).lower())

    @patch("src.providers.gemini_provider.genai.Client")
    def test_verify_model_access(self, mock_client_cls):
        """Test pre-flight model verification before expensive uploads."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Accessible model
        mock_client.models.get.return_value = MagicMock(name="models/gemini-2.0-flash")
        ok, msg = self.provider.verify_model_access("AIzaSyValidKey12345678901234567890", "gemini-2.0-flash")
        self.assertTrue(ok)
        self.assertIn("accessible", msg)

        # Handles models/ prefix safely
        ok, msg = self.provider.verify_model_access("AIzaSyValidKey12345678901234567890", "models/gemini-2.0-flash")
        self.assertTrue(ok)

        # Inaccessible model (404)
        mock_client.models.get.side_effect = errors.ClientError(404, {"error": {"message": "Model not found."}})
        ok, msg = self.provider.verify_model_access("AIzaSyValidKey12345678901234567890", "gemini-nonexistent")
        self.assertFalse(ok)
        self.assertIn("not found", msg.lower())

    def test_dynamic_model_capabilities(self):
        """Test capability detection works dynamically across models and handles prefixes."""
        caps_flash = self.provider.get_model_capabilities("gemini-2.0-flash")
        self.assertIn(ModelCapability.VIDEO_UNDERSTANDING, caps_flash)
        self.assertIn(ModelCapability.AUDIO_UNDERSTANDING, caps_flash)
        self.assertIn(ModelCapability.LOW_LATENCY, caps_flash)

        caps_pro = self.provider.get_model_capabilities("models/gemini-1.5-pro")
        self.assertIn(ModelCapability.VIDEO_UNDERSTANDING, caps_pro)
        self.assertIn(ModelCapability.AUDIO_UNDERSTANDING, caps_pro)

        with self.assertRaises(ModelNotFoundError):
            self.provider.get_model_capabilities("text-embedding-004")

    def test_resolve_input_strategy(self):
        """Verify model capability and payload size decide input strategy."""
        strategy_small = self.provider.resolve_input_strategy(
            "gemini-2.5-flash",
            file_size_bytes=20 * 1024 * 1024,
            duration_secs=120.0,
        )
        self.assertEqual(strategy_small, InputModality.NATIVE_VIDEO)

        strategy_large = self.provider.resolve_input_strategy(
            "gemini-2.5-flash",
            file_size_bytes=250 * 1024 * 1024,
            duration_secs=3600.0,
        )
        self.assertEqual(strategy_large, InputModality.AUDIO_EXTRACTED)

    def test_build_analysis_prompt(self):
        """Verify prompt contains necessary instructions and formatting constraints."""
        prompt = self.provider.build_analysis_prompt(
            target_clip_count=4,
            target_clip_duration_secs=40.0,
            video_duration_secs=600.0,
            custom_instructions="Focus on coding tips",
        )
        self.assertIn("exactly 4 engaging", prompt)
        self.assertIn("40 seconds", prompt)
        self.assertIn("Focus on coding tips", prompt)
        self.assertIn("Never cut mid-sentence", prompt)

    @patch("src.providers.gemini_provider.genai.Client")
    def test_gemini_analyze_highlights_mock_success(self, mock_client_cls):
        """Test full mock execution: upload -> structured generation -> parsing -> cleanup."""
        dummy_video = self.tmp_path / "dummy.mp4"
        dummy_video.write_bytes(b"dummy video data for mock testing")

        # Setup mock client
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_file = MagicMock()
        mock_file.name = "files/test_uploaded_file_123"
        mock_file.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_file

        # Setup mock response
        mock_response = MagicMock()
        mock_response.parsed = GeminiHighlightCollection(
            highlights=[
                GeminiHighlightItem(
                    title="Engaging Insight",
                    start_time=12.0,
                    end_time=42.0,
                    hook="Here is the secret to scaling...",
                    explanation="Clear self-contained discussion with actionable advice.",
                    confidence_score=0.95,
                ),
                GeminiHighlightItem(
                    title="Funny Story",
                    start_time=50.0,
                    end_time=80.0,
                    hook="Then the server exploded...",
                    explanation="Humorous anecdote with a punchline.",
                    confidence_score=0.88,
                ),
            ]
        )
        mock_response.text = '{"highlights": []}'
        mock_response.usage_metadata.prompt_token_count = 1200
        mock_response.usage_metadata.candidates_token_count = 150
        mock_response.usage_metadata.total_token_count = 1350
        mock_client.models.generate_content.return_value = mock_response

        req = HighlightRequest(
            video_path=dummy_video,
            video_duration=100.0,
            file_size_bytes=dummy_video.stat().st_size,
            target_clip_count=2,
            target_clip_duration_secs=30.0,
            preferred_modality=InputModality.NATIVE_VIDEO,
        )

        resp = self.provider.analyze_highlights(
            request=req,
            api_key="AIzaSyTestKeyValid1234567890123456",
            model_name="gemini-2.5-flash",
        )

        self.assertEqual(resp.provider_id, "gemini")
        self.assertEqual(resp.model_name, "gemini-2.5-flash")
        self.assertEqual(len(resp.highlights), 2)
        self.assertEqual(resp.highlights[0].title, "Engaging Insight")
        self.assertEqual(resp.highlights[0].start_time, 12.0)
        self.assertEqual(resp.highlights[0].end_time, 42.0)
        self.assertEqual(resp.highlights[0].duration, 30.0)

        # Verify responsible provider cleanup was called
        mock_client.files.delete.assert_called_once_with(name="files/test_uploaded_file_123")

    @patch("src.providers.gemini_provider.genai.Client")
    def test_gemini_filter_out_of_bounds_candidates(self, mock_client_cls):
        """Test that proposed moments exceeding video duration are filtered out."""
        dummy_video = self.tmp_path / "dummy.mp4"
        dummy_video.write_bytes(b"dummy video data")

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_file = MagicMock()
        mock_file.name = "files/test_file"
        mock_file.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.parsed = GeminiHighlightCollection(
            highlights=[
                # Valid (10s to 40s in a 50s video)
                GeminiHighlightItem(
                    title="Valid Clip",
                    start_time=10.0,
                    end_time=40.0,
                    hook="Valid hook",
                    explanation="Within bounds",
                    confidence_score=0.9,
                ),
                # Invalid: exceeds video duration (50s)
                GeminiHighlightItem(
                    title="Out of Bounds",
                    start_time=30.0,
                    end_time=85.0,
                    hook="Invalid hook",
                    explanation="Exceeds duration",
                    confidence_score=0.9,
                ),
            ]
        )
        mock_client.models.generate_content.return_value = mock_response

        req = HighlightRequest(
            video_path=dummy_video,
            video_duration=50.0,
            file_size_bytes=dummy_video.stat().st_size,
            preferred_modality=InputModality.NATIVE_VIDEO,
        )

        resp = self.provider.analyze_highlights(req, api_key="AIzaSyTestKeyValid1234567890123456")
        self.assertEqual(len(resp.highlights), 1)
        self.assertEqual(resp.highlights[0].title, "Valid Clip")

    @patch("src.providers.gemini_provider.genai.Client")
    def test_gemini_api_error_handling_quota_exceeded(self, mock_client_cls):
        """Mock Gemini 429 quota exhaustion and verify QuotaExceededError."""
        dummy_video = self.tmp_path / "dummy.mp4"
        dummy_video.write_bytes(b"dummy")

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_file = MagicMock()
        mock_file.name = "files/quota_test"
        mock_file.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_file

        # Raise 429 APIError
        api_err = errors.APIError(429, {"error": {"message": "Resource has been exhausted (e.g. check quota)."}})
        mock_client.models.generate_content.side_effect = api_err

        req = HighlightRequest(
            video_path=dummy_video,
            video_duration=60.0,
            file_size_bytes=dummy_video.stat().st_size,
            preferred_modality=InputModality.NATIVE_VIDEO,
        )

        with self.assertRaises(QuotaExceededError):
            self.provider.analyze_highlights(req, api_key="AIzaSyTestKeyValid1234567890123456")

        # Provider cleanup must still occur
        mock_client.files.delete.assert_called_once_with(name="files/quota_test")

    @patch("src.providers.gemini_provider.genai.Client")
    def test_gemini_api_error_handling_invalid_key(self, mock_client_cls):
        """Mock Gemini 400/403 invalid key error and verify InvalidAPIKeyError."""
        dummy_video = self.tmp_path / "dummy.mp4"
        dummy_video.write_bytes(b"dummy")

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_file = MagicMock()
        mock_file.name = "files/auth_test"
        mock_file.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_file

        api_err = errors.APIError(400, {"error": {"message": "API_KEY_INVALID: API key not valid. Please pass a valid API key."}})
        mock_client.models.generate_content.side_effect = api_err

        req = HighlightRequest(
            video_path=dummy_video,
            video_duration=60.0,
            file_size_bytes=dummy_video.stat().st_size,
            preferred_modality=InputModality.NATIVE_VIDEO,
        )

        with self.assertRaises(InvalidAPIKeyError):
            self.provider.analyze_highlights(req, api_key="AIzaSyTestKeyValid1234567890123456")

    @patch("src.providers.gemini_provider.genai.Client")
    def test_gemini_malformed_json_response(self, mock_client_cls):
        """Mock unparseable text response and verify ProviderError."""
        dummy_video = self.tmp_path / "dummy.mp4"
        dummy_video.write_bytes(b"dummy")

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_file = MagicMock()
        mock_file.name = "files/json_test"
        mock_file.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_file

        mock_resp = MagicMock()
        mock_resp.parsed = None
        mock_resp.text = "Error: internal server format malformed {not valid json"
        mock_client.models.generate_content.return_value = mock_resp

        req = HighlightRequest(
            video_path=dummy_video,
            video_duration=60.0,
            file_size_bytes=dummy_video.stat().st_size,
            preferred_modality=InputModality.NATIVE_VIDEO,
        )

        with self.assertRaises(ProviderError):
            self.provider.analyze_highlights(req, api_key="AIzaSyTestKeyValid1234567890123456")


if __name__ == "__main__":
    unittest.main()
