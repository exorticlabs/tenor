"""
Standard-Library HTTP LLM Client (llm_client.py)
================================================

Enterprise-grade, zero-third-party-dependency HTTP client for calling LLM inference APIs:
- Ollama Cloud / Local Ollama (native /api/chat with format="json")
- OpenAI-compatible endpoints (/v1/chat/completions)
- Configurable environment variable overrides (OLLAMA_HOST, OLLAMA_API_KEY, LLM_BASE_URL, LLM_API_KEY)
- Exponential backoff with uniform jitter and robust JSON fence stripping
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# Auto-load .env configuration if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Look for .env in project root
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
    else:
        load_dotenv()
except ImportError:
    # If dotenv is not installed, parse .env manually if exists
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(_env_path):
        try:
            with open(_env_path, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _k, _v = _line.split("=", 1)
                        _v = _v.strip().strip("'\"")
                        if _k.strip() not in os.environ:
                            os.environ[_k.strip()] = _v
        except Exception:
            pass


class LLMClientError(Exception):
    """Raised when an LLM inference call fails or returns unparseable content."""
    pass


class MemoryLLMClient:
    """
    Standard-library HTTP client for Ollama Cloud, local Ollama, and OpenAI-compatible inference APIs.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ):
        self.api_key = (
            api_key
            or os.environ.get("OLLAMA_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or ""
        )

        default_base = "https://ollama.com" if self.api_key else "http://localhost:11434"
        raw_url = (
            base_url
            or os.environ.get("OLLAMA_HOST")
            or os.environ.get("OLLAMA_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or default_base
        ).rstrip("/")

        if raw_url.endswith("/v1"):
            self.root_url = raw_url[:-3]
            self.v1_url = raw_url
        else:
            self.root_url = raw_url
            self.v1_url = f"{raw_url}/v1"

        self.model = (
            model
            or os.environ.get("OLLAMA_MODEL")
            or os.environ.get("LLM_MODEL")
            or ("gpt-oss:20b" if self.api_key else "qwen2.5:7b")
        )
        self.prefer_openai_v1 = bool(
            raw_url.endswith("/v1")
            or "20128" in raw_url
            or "openai" in raw_url
            or os.environ.get("LLM_BASE_URL")
        )
        self.timeout = max(timeout, 120.0)
        self.max_retries = max(max_retries, 1)
        self.backoff_base = backoff_base
        self._available: Optional[bool] = None

    def is_available(self, force_check: bool = False) -> bool:
        """Checks if the configured endpoint is reachable."""
        if self._available is not None and not force_check:
            return self._available

        if self.api_key:
            self._available = True
            return True

        # Fast socket probe before full HTTP handshake
        try:
            import socket
            from urllib.parse import urlparse
            parsed = urlparse(self.root_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else (11434 if "11434" in self.root_url else 80))
            with socket.create_connection((host, port), timeout=2.0):
                self._available = True
                return True
        except Exception:
            self._available = False
            return False

    @staticmethod
    def clean_json_response(content: str) -> Dict[str, Any]:
        """
        Strips markdown code fences (```json ... ```) or extracts the outermost
        JSON object {...} from raw LLM output, with resilient character normalization.
        """
        cleaned = content.strip()

        # Strip model thinking blocks (<think>...</think>) if present
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned).strip()

        # Normalize unicode quotes and dashes that can break json.loads
        cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
        cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
        cleaned = cleaned.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")

        # Strip markdown fences if present
        cleaned_no_fence = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned_no_fence = re.sub(r"\s*```$", "", cleaned_no_fence)

        def _try_parse(text: str) -> Optional[Dict[str, Any]]:
            # Try strict=False first (allows literal newlines and control chars in strings)
            try:
                res = json.loads(text, strict=False)
                if isinstance(res, dict):
                    return res
            except Exception:
                pass
            # Try fixing trailing commas before closing braces/brackets
            fixed = re.sub(r",\s*([\}\]])", r"\1", text)
            try:
                res = json.loads(fixed, strict=False)
                if isinstance(res, dict):
                    return res
            except Exception:
                pass
            return None

        # 1. Regex match for ```json {...} ``` (greedy across entire fence block)
        fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", cleaned)
        if fence_match:
            parsed = _try_parse(fence_match.group(1))
            if parsed is not None:
                return parsed

        # 2. Match outermost { ... }
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            substring = cleaned[start_idx : end_idx + 1]
            parsed = _try_parse(substring)
            if parsed is not None:
                return parsed

        # 3. Cleaned no fence attempt
        parsed = _try_parse(cleaned_no_fence)
        if parsed is not None:
            return parsed

        # 4. Partial/truncated recovery: search backwards for valid closing brace
        if start_idx != -1:
            all_braces = [i for i, c in enumerate(cleaned) if c == "}"]
            for last_brace in reversed(all_braces):
                if last_brace > start_idx:
                    sub = cleaned[start_idx : last_brace + 1]
                    parsed = _try_parse(sub)
                    if parsed is not None:
                        return parsed

        raise LLMClientError(f"Failed to parse JSON from LLM response: {cleaned[:300]}...")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        json_output: bool = True,
    ) -> str:
        """
        Executes a chat completion request with exponential backoff and protocol fallback.
        Returns the raw string output of the assistant's message.
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "EEKG-Memory/2.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error = None

        # Guard against context overflow (default 256k characters, configurable)
        max_chars = int(os.environ.get("LLM_MAX_CHARS", "256000"))
        safe_messages = []
        for m in messages:
            c_text = str(m.get("content", ""))
            if len(c_text) > max_chars:
                half = max_chars // 2 - 100
                c_text = c_text[:half] + "\n\n...[middle context omitted for context window]...\n\n" + c_text[-half:]
            safe_messages.append({"role": m.get("role", "user"), "content": c_text})

        for attempt in range(self.max_retries):
            # If endpoint is OpenAI compatible (e.g. Omniroute, OpenAI, v1 endpoints), call /v1/chat/completions directly
            if self.prefer_openai_v1:
                v1_endpoint = f"{self.v1_url}/chat/completions"
                v1_payload: Dict[str, Any] = {
                    "model": self.model,
                    "messages": safe_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if json_output:
                    v1_payload["response_format"] = {"type": "json_object"}

                try:
                    data = json.dumps(v1_payload).encode("utf-8")
                    req = urllib.request.Request(v1_endpoint, data=data, headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=self.timeout) as response:
                        if response.status == 200:
                            body = json.loads(response.read().decode("utf-8"))
                            choices = body.get("choices", [])
                            if choices:
                                content = choices[0].get("message", {}).get("content")
                                if content is not None:
                                    return str(content)
                                else:
                                    last_error = RuntimeError(f"Null content in choice: {choices[0]}")
                            else:
                                last_error = RuntimeError(f"No choices in response: {body}")
                        else:
                            last_error = RuntimeError(f"HTTP status {response.status}")
                except Exception as e:
                    err_detail = ""
                    if hasattr(e, "read"):
                        try:
                            err_detail = f" | Body: {e.read().decode('utf-8')}"
                        except Exception:
                            pass
                    last_error = f"{e}{err_detail}"

                jitter = random.uniform(0.1, 0.5)
                backoff = min(10.0, self.backoff_base * (2**attempt)) + jitter
                time.sleep(backoff)
                continue

            # Attempt 1: Native Ollama /api/chat
            native_endpoint = f"{self.root_url}/api/chat"
            native_payload: Dict[str, Any] = {
                "model": self.model,
                "messages": safe_messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            if json_output:
                native_payload["format"] = "json"

            try:
                data = json.dumps(native_payload).encode("utf-8")
                req = urllib.request.Request(native_endpoint, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    if response.status == 200:
                        body = json.loads(response.read().decode("utf-8"))
                        content = body.get("message", {}).get("content")
                        if content is not None:
                            return str(content)
                        else:
                            last_error = RuntimeError(f"Null content in native response: {body}")
                    else:
                        last_error = RuntimeError(f"HTTP status {response.status}")
            except Exception as e:
                err_detail = ""
                if hasattr(e, "read"):
                    try:
                        err_detail = f" | Body: {e.read().decode('utf-8')}"
                    except Exception:
                        pass
                last_error = f"{e}{err_detail}"

            # Attempt 2: Fallback to OpenAI /v1/chat/completions
            v1_endpoint = f"{self.v1_url}/chat/completions"
            v1_payload: Dict[str, Any] = {
                "model": self.model,
                "messages": safe_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_output:
                v1_payload["response_format"] = {"type": "json_object"}

            try:
                data = json.dumps(v1_payload).encode("utf-8")
                req = urllib.request.Request(v1_endpoint, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    if response.status == 200:
                        body = json.loads(response.read().decode("utf-8"))
                        choices = body.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content")
                            if content is not None:
                                return str(content)
                            else:
                                last_error = RuntimeError(f"Null content in choice: {choices[0]}")
                        else:
                            last_error = RuntimeError(f"No choices in response: {body}")
                    else:
                        last_error = RuntimeError(f"HTTP status {response.status}")
            except Exception as e:
                err_detail = ""
                if hasattr(e, "read"):
                    try:
                        err_detail = f" | Body: {e.read().decode('utf-8')}"
                    except Exception:
                        pass
                last_error = f"{e}{err_detail}"

            # Exponential backoff with uniform jitter before retry
            jitter = random.uniform(0.1, 0.5)
            backoff = min(10.0, self.backoff_base * (2**attempt)) + jitter
            time.sleep(backoff)

        raise LLMClientError(f"LLM request failed after {self.max_retries} attempts: {last_error}")
