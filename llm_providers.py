"""
Corporate Accountability Engine (CAE) — LLM Provider Abstraction
Built for Act For Farmed Animals (AFFA) / Sinergia Animal International

Supports multiple LLM providers, with Hybrid Routing for Gemini:
  1. Google Gemini (Hybrid: gemini-3-flash-preview for fast scan, gemini-3.1-pro-preview for deep analysis)
  2. Qwen (Backup — Qwen 3.5-397B-A17B via DashScope OpenAI-compatible API)
  3. OpenAI (Paid — GPT-4o, if grant money comes through)

All providers use the same interface so you can swap with a single env var.

Updated: March 2026 — Providers: Gemini (primary), Qwen 3.5-235B-A22B (backup), OpenAI (grant-funded)
"""

import os
import json
import httpx
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

LLM_CONCURRENCY = asyncio.Semaphore(5)


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
    def __init__(self, api_key: str = None, model_name: str = "gemini-3-flash-preview"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required. Get free at https://aistudio.google.com/apikey")
        
        self.model = model_name
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

        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        async def _make_request():
            async with LLM_CONCURRENCY:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    response = await client.post(url, json=payload)
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
    """
    Qwen API via Alibaba DashScope International — OpenAI-compatible endpoint.
    - Model: qwen3.5-235b-a22b
    - Context: 131k tokens
    - Supports thinking/reasoning mode (enable_thinking)
    - Best backup for large documents when Gemini is rate-limited.

    Get your API key: https://dashscope.console.aliyun.com/
    Set env var: DASHSCOPE_API_KEY
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY is not configured. "
                "Qwen is listed as a backup provider but DashScope International is not yet available. "
                "Set the DASHSCOPE_API_KEY env var when it launches: https://dashscope.console.aliyun.com/"
            )
        self.model = "qwen3.5-235b-a22b"
        self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    async def analyze(self, messages: list[dict]) -> LLMResponse:
        """Send analysis request to Qwen via DashScope OpenAI-compatible API."""

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 16384,
            "response_format": {"type": "json_object"},
            "extra_body": {"enable_thinking": False}  # Disabled for structured JSON output
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

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=self.model,
            provider="qwen",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cost_estimate_usd=0.0  # Pay-as-you-go, minimal cost
        )


# ============================================================================
# PROVIDER 3: OPENAI (PAID — if grant comes through)
# ============================================================================

class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API — PAID
    - Model: gpt-4o
    - ~$0.15-$0.30 per 200-page report
    - 128k context window
    - Best quality, but costs money.

    Only use if AFFA gets grant funding.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        self.model = "gpt-4o"
        self.base_url = "https://api.openai.com/v1"

    async def analyze(self, messages: list[dict]) -> LLMResponse:
        """Send analysis request to OpenAI API."""

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 8192,
            "response_format": {"type": "json_object"}
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

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        # Estimate cost (GPT-4o pricing as of 2025)
        input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * 2.50
        output_cost = (usage.get("completion_tokens", 0) / 1_000_000) * 10.00
        total_cost = input_cost + output_cost

        return LLMResponse(
            content=content,
            model=self.model,
            provider="openai",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cost_estimate_usd=round(total_cost, 4)
        )


# ============================================================================
# PROVIDER FACTORY
# ============================================================================

PROVIDERS = {
    "gemini": GeminiProvider,
    "qwen": QwenProvider,
    "openai": OpenAIProvider,
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
        provider_name: "gemini", "qwen", or "openai"
        api_key: Optional API key (otherwise reads from env)

    Returns:
        An initialized LLM provider instance
    """
    if provider_name is None:
        provider_name = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider_name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available: {', '.join(PROVIDERS.keys())}"
        )

    provider_class = PROVIDERS[provider_name]

    if api_key:
        return provider_class(api_key=api_key)
    return provider_class()


# ============================================================================
# PROVIDER RECOMMENDATIONS
# ============================================================================

PROVIDER_GUIDE = """
╔══════════════════════════════════════════════════════════════╗
║              CAE — LLM PROVIDER GUIDE (2026)                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🥇 GEMINI (Primary — Recommended)                          ║
║     Stage 1: gemini-3-flash-preview (fast scan)              ║
║     Stage 2: gemini-3.1-pro-preview (escalation for high risk)║                       
║     Context: 1M tokens (handles full 300-page PDFs!)         ║
║     Paid: $300 prepaid balance                                ║
║     Best for: Large documents, single-pass analysis          ║
║     Key: https://aistudio.google.com/apikey                  ║
║                                                              ║
║  🥈 QWEN (Backup)                                           ║
║     Model: qwen3.5-235b-a22b                                 ║
║     Context: 131k tokens                                     ║
║     API: DashScope International (OpenAI-compatible)         ║
║     Best for: Backup when Gemini is rate-limited             ║
║     Key: https://dashscope.console.aliyun.com/               ║
║                                                              ║
║  💰 OPENAI (in the future)                                  ║
║     Model: gpt-5                                            ║
║     Context: 128k tokens                                     ║
║     Cost: ~$0.15-$0.30 per 200-page report                  ║
║     Best for: Highest quality, if budget allows              ║
║                                                              ║
║  ⚡ ALL PROVIDERS: Auto-retry on 429 rate limits             ║
║     Backoff: 15s → 30s → 60s (3 retries max)                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    print(PROVIDER_GUIDE)
