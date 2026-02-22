"""
Corporate Accountability Engine (CAE) — LLM Provider Abstraction
Built for Act For Farmed Animals (AFFA) / Sinergia Animal International

Supports multiple FREE LLM providers:
  1. Google Gemini (Free tier — gemini-2.5-flash, 1M tokens)
  2. Groq (Free tier — Llama 3, 30 RPM)
  3. Mistral (Free tier — mistral-small)
  4. OpenAI (Paid — GPT-4o, if grant money comes through)

All providers use the same interface so you can swap with a single env var.

Updated: Feb 2026 — Gemini 2.5 Flash (gemini-2.0-flash is deprecated)
"""

import os
import json
import httpx
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass


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
# RETRY HELPER — handles 429 (rate limit) automatically
# ============================================================================

async def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 15.0):
    """
    Retry an async function with exponential backoff.
    Specifically handles 429 Too Many Requests from LLM providers.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries (default 3)
        base_delay: Base delay in seconds between retries (default 15s for rate limits)
    
    Returns:
        The result of the function call
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except httpx.HTTPStatusError as e:
            last_exception = e
            if e.response.status_code == 429 and attempt < max_retries:
                # Rate limited — wait and retry
                delay = base_delay * (2 ** attempt)  # 15s, 30s, 60s
                print(f"  ⏳ Rate limited (429). Waiting {delay}s before retry {attempt + 1}/{max_retries}...")
                await asyncio.sleep(delay)
            else:
                raise
        except Exception as e:
            raise
    
    raise last_exception


# ============================================================================
# PROVIDER 1: GOOGLE GEMINI (FREE)
# ============================================================================

class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini API — FREE TIER
    - Model: gemini-2.5-flash (latest as of Feb 2026)
    - Free: 15 requests/min, 1 million tokens/min
    - Context window: 1M tokens (can handle entire 300-page PDFs!)
    - Best free option for large documents.

    Get your free API key: https://aistudio.google.com/apikey
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required. Get free at https://aistudio.google.com/apikey")
        self.model = "gemini-2.5-flash"
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def analyze(self, messages: list[dict]) -> LLMResponse:
        """Send analysis request to Gemini API with automatic retry on rate limit."""

        # Convert OpenAI-style messages to Gemini format
        system_text = ""
        user_text = ""
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            elif msg["role"] == "user":
                user_text = msg["content"]

        payload = {
            "contents": [
                {
                    "parts": [{"text": user_text}]
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system_text}]
            },
            "generationConfig": {
                "temperature": 0.1,  # Low temp for precise, factual analysis
                "topP": 0.95,
                "maxOutputTokens": 65536,  # Large output for detailed 300-page analysis
                "responseMimeType": "application/json"  # Force JSON output
            }
        }

        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        async def _make_request():
            async with httpx.AsyncClient(timeout=180.0) as client:  # 3 min timeout for large docs
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()

        # Retry up to 3 times on 429 errors with 15s/30s/60s backoff
        data = await retry_with_backoff(_make_request)

        # Extract response
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})

        return LLMResponse(
            content=content,
            model=self.model,
            provider="gemini",
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            total_tokens=usage.get("totalTokenCount", 0),
            cost_estimate_usd=0.0  # Free tier!
        )


# ============================================================================
# PROVIDER 2: GROQ (FREE)
# ============================================================================

class GroqProvider(BaseLLMProvider):
    """
    Groq API — FREE TIER
    - Model: llama-3.3-70b-versatile
    - Free: 30 requests/min, 131k context
    - EXTREMELY fast inference (fastest in the world)
    - Good for smaller documents or chunked analysis.

    Get your free API key: https://console.groq.com/keys
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required. Get free at https://console.groq.com/keys")
        self.model = "llama-3.3-70b-versatile"
        self.base_url = "https://api.groq.com/openai/v1"

    async def analyze(self, messages: list[dict]) -> LLMResponse:
        """Send analysis request to Groq API (OpenAI-compatible)."""

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
            async with httpx.AsyncClient(timeout=180.0) as client:
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
            provider="groq",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cost_estimate_usd=0.0  # Free tier!
        )


# ============================================================================
# PROVIDER 3: MISTRAL (FREE)
# ============================================================================

class MistralProvider(BaseLLMProvider):
    """
    Mistral API — FREE TIER
    - Model: mistral-small-latest
    - Free: limited requests (check current limits)
    - 32k context window
    - Good European alternative, strong multilingual.

    Get your free API key: https://console.mistral.ai/api-keys
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is required. Get free at https://console.mistral.ai/api-keys")
        self.model = "mistral-small-latest"
        self.base_url = "https://api.mistral.ai/v1"

    async def analyze(self, messages: list[dict]) -> LLMResponse:
        """Send analysis request to Mistral API."""

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
            async with httpx.AsyncClient(timeout=180.0) as client:
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
            provider="mistral",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cost_estimate_usd=0.0  # Free tier!
        )


# ============================================================================
# PROVIDER 4: OPENAI (PAID — if grant comes through)
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
            async with httpx.AsyncClient(timeout=180.0) as client:
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
    "groq": GroqProvider,
    "mistral": MistralProvider,
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
      2. Default to "gemini" (best free option for large docs)

    Args:
        provider_name: "gemini", "groq", "mistral", or "openai"
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
║  🥇 GEMINI (Recommended — FREE)                             ║
║     Model: gemini-2.5-flash (latest)                         ║
║     Context: 1M tokens (handles full 300-page PDFs!)         ║
║     Free: 15 req/min, 1M tokens/min                          ║
║     Best for: Large documents, single-pass analysis          ║
║     Key: https://aistudio.google.com/apikey                  ║
║                                                              ║
║  🥈 GROQ (Runner-up — FREE)                                 ║
║     Model: llama-3.3-70b-versatile                           ║
║     Context: 131k tokens                                     ║
║     Free: 30 req/min                                         ║
║     Best for: Speed, smaller docs, chunked analysis          ║
║     Key: https://console.groq.com/keys                       ║
║                                                              ║
║  🥉 MISTRAL (Backup — FREE)                                 ║
║     Model: mistral-small-latest                              ║
║     Context: 32k tokens                                      ║
║     Free: Limited requests                                   ║
║     Best for: Backup when others are rate-limited            ║
║     Key: https://console.mistral.ai/api-keys                ║
║                                                              ║
║  💰 OPENAI (If grant funded)                                ║
║     Model: gpt-4o                                            ║
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
