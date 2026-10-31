"""CAE providers: Gemini hybrid analysis and Qwen 3.8 Max structured analysis."""

import os
import json
import httpx
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

LLM_CONCURRENCY = asyncio.Semaphore(5)
DEFAULT_QWEN_MODEL = "qwen3.8-max"


def resolve_provider_name(provider_name: str | None = None) -> str:
    name = (provider_name or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
    if name not in {"gemini", "qwen"}:
        raise ValueError("Supported providers: gemini, qwen. Update LLM_PROVIDER or the saved provider selection.")
    return name



@dataclass
class LLMResponse:
    """Standard response from any LLM provider."""
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_estimate_usd: float


class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    async def analyze(self, messages: list[dict]) -> LLMResponse:
        pass


# ============================================================================
# RETRY HELPER — handles 429 & 50x (server errors) automatically
# ============================================================================

async def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 15.0):
    """
    Retry an async function with exponential backoff.
    Specifically handles 429 (Too Many Requests) and 50x (Server Errors) from LLM providers.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries (default 3)
        base_delay: Base delay in seconds between retries (default 15s)
    
    Returns:
        The result of the function call
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except httpx.HTTPStatusError as e:
            last_exception = e
            status = e.response.status_code
            
            # Cek apakah error terkait limit (429) atau server sedang down (500, 502, 503, 504)
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                # Waktu tunggu eksponensial: 15s, 30s, 60s
                delay = base_delay * (2 ** attempt)  
                
                error_type = "Rate limited (429)" if status == 429 else f"Server error ({status})"
                print(f"  ⏳ {error_type}. API LLM sibuk. Menunggu {delay} detik sebelum percobaan {attempt + 1}/{max_retries}...")
                
                await asyncio.sleep(delay)
            else:
                # Jika error selain 429/50x (misal 400 Bad Request) atau jatah retry habis, hentikan.
                raise
        except Exception as e:
            raise
    
    raise last_exception


# ============================================================================
# PROVIDER 1: GOOGLE GEMINI (HYBRID 3.0 FLASH & 3.1 PRO)
# ============================================================================

class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini API 
    Mendukung Gemini 3 Flash Preview (Kecepatan) dan Gemini 3.1 Pro Preview (Penalaran Mendalam).
    """
    # Default model diperbarui ke gemini-3-flash-preview
    def __init__(self, api_key: str = None, model_name: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required. Get free at https://aistudio.google.com/apikey")
        
        self.model = model_name or os.getenv("GEMINI_FLASH_MODEL", "gemini-3-flash-preview")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def analyze(self, messages: list[dict]) -> LLMResponse:
        system_text = ""
        user_text = ""
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            elif msg["role"] == "user":
                user_text = msg["content"]

        payload = {
            "contents": [{"parts": [{"text": user_text}]}],
            "systemInstruction": {"parts": [{"text": system_text}]},
            "generationConfig": {
                "temperature": 0.1,  
                "topP": 0.95,
                "maxOutputTokens": 65536,  
                "responseMimeType": "application/json"  
            }
        }

        url = f"{self.base_url}/models/{self.model}:generateContent"

        async def _make_request():
            async with LLM_CONCURRENCY:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    response = await client.post(url, json=payload, headers={"x-goog-api-key": self.api_key})
                    response.raise_for_status()
                    return response.json()

        data = await retry_with_backoff(_make_request)

        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError(f"No candidates in response. Keys: {list(data.keys())}")
            
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError(f"No parts in response. Candidate keys: {list(candidates[0].keys())}")
            
            # Thinking models return multiple parts — thinking first, answer last
            # Take the LAST text part (the actual JSON response)
            content = ""
            for part in parts:
                if "text" in part:
                    content = part["text"]
            
            if not content:
                raise ValueError(f"No text found in parts. Parts: {json.dumps(parts)[:500]}")
                
        except (KeyError, IndexError) as e:
            raise ValueError(f"Gemini response parse error: {e}. Response: {json.dumps(data)[:500]}")

        usage = data.get("usageMetadata", {})

        return LLMResponse(
            content=content,
            model=self.model,
            provider="gemini",
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            total_tokens=usage.get("totalTokenCount", 0),
            cost_estimate_usd=0.0
        )


# ============================================================================
# PROVIDER 2: QWEN (BACKUP — DashScope International)
# ============================================================================

class QwenProvider(BaseLLMProvider):
    """Qwen 3.8 Max through Alibaba Cloud Model Studio Chat Completions."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for Qwen. Use a key from the endpoint's region.")
        self.model = os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL)
        workspace = os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()
        if workspace and not all(c.isalnum() or c == "-" for c in workspace):
            raise ValueError("Invalid DASHSCOPE_WORKSPACE_ID")
        default_url = (f"https://{workspace}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
                       if workspace else "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        self.base_url = (os.getenv("QWEN_BASE_URL") or default_url).rstrip("/")
        parsed = httpx.URL(self.base_url)
        if parsed.scheme != "https" or not parsed.host or parsed.userinfo or parsed.query or parsed.fragment:
            raise ValueError("QWEN_BASE_URL must be an HTTPS base URL without credentials, query or fragment")

    async def analyze(self, messages: list[dict]) -> LLMResponse:
        """Send analysis request to Qwen via DashScope OpenAI-compatible API."""

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 32768,
            "response_format": {"type": "json_object"}
            # Qwen 3.8 supports structured output with its default thinking mode.
            # extra_body is an SDK option, not a nested HTTP request field.
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async def _make_request():
            async with LLM_CONCURRENCY:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers
                    )
                    response.raise_for_status()
                    return response.json()

        data = await retry_with_backoff(_make_request)

        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("Qwen output exceeded its limit; retry with fewer target countries or files.")
        content = choice["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Qwen returned no final answer")
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=self.model,
            provider="qwen",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cost_estimate_usd=0.0  # Unknown; result metadata explicitly marks estimates unavailable
        )


# ============================================================================
# PROVIDER FACTORY
# ============================================================================

PROVIDERS = {
    "gemini": GeminiProvider,
    "qwen": QwenProvider,
}


def get_provider(
    provider_name: str = None,
    api_key: str = None
) -> BaseLLMProvider:
    """
    Get an LLM provider instance.

    Priority (if provider_name is None):
      1. Check LLM_PROVIDER env var
      2. Default to "gemini" (best option for large docs)

    Args:
        provider_name: "gemini" or "qwen"
        api_key: Optional API key (otherwise reads from env)

    Returns:
        An initialized LLM provider instance
    """
    provider_name = resolve_provider_name(provider_name)

    provider_class = PROVIDERS[provider_name]

    if api_key:
        return provider_class(api_key=api_key)
    return provider_class()


# ============================================================================
# PROVIDER RECOMMENDATIONS
# ============================================================================

PROVIDER_GUIDE = """
CAE supports two providers:
- Gemini: Flash scan with Pro escalation for high-risk evidence.
- Qwen: qwen3.8-max, JSON output, optional workspace-specific endpoint.
Both are usage-billed services. Configure keys server-side or per request.
See env.example and docs/model-providers.md. Retry 429/5xx with backoff.
"""

if __name__ == "__main__":
    print(PROVIDER_GUIDE)
