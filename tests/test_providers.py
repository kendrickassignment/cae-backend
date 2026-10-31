import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
import main
import llm_providers as providers


def test_two_providers_only_and_configured_models(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "account-flash")
    monkeypatch.delenv("QWEN_MODEL", raising=False)
    assert set(providers.PROVIDERS) == {"gemini", "qwen"}
    assert providers.get_provider("gemini", "test-key").model == "account-flash"
    assert providers.get_provider("qwen", "test-key").model == "qwen3.8-max"
    with TestClient(main.app) as client:
        data = client.get("/providers").json()
        assert set(data["available"]) == {"gemini", "qwen"}
        assert data["available"]["qwen"]["model"] == "qwen3.8-max"
        assert client.post("/providers/test",json={"provider":"openai","api_key":"old-key"}).status_code == 400


@pytest.mark.parametrize("selection", ["openai","groq","mistral"])
def test_removed_provider_rejected_before_job_reservation(monkeypatch,selection):
    monkeypatch.setitem(main.reports_store,"provider-test", {"status":"uploaded"})
    with TestClient(main.app) as client:
        for endpoint,field in [("/analyze",{"report_id":"provider-test"}),("/analyze-multi",{"report_ids":["provider-test"]})]:
            assert client.post(endpoint,json={**field,"provider":selection}).status_code == 400
    assert main.reports_store["provider-test"]["status"] == "uploaded"
    with pytest.raises(ValueError):
        providers.get_provider(selection,"key")


def test_invalid_server_default_fails_clearly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER","openai")
    monkeypatch.setitem(main.reports_store,"provider-test", {"status":"uploaded"})
    with TestClient(main.app) as client:
        assert client.get("/providers").status_code == 503
        assert client.post("/analyze",json={"report_id":"provider-test"}).status_code == 503
    assert main.reports_store["provider-test"]["status"] == "uploaded"


async def test_qwen_http_payload_endpoint_and_final_answer(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID","workspace-test")
    monkeypatch.delenv("QWEN_BASE_URL",raising=False)
    monkeypatch.delenv("QWEN_MODEL",raising=False)
    def handle(req):
        assert str(req.url) == "https://workspace-test.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
        assert req.headers["Authorization"] == "Bearer qwen-key"
        body=json.loads(req.content)
        assert body["model"] == "qwen3.8-max"
        assert body["response_format"] == {"type":"json_object"}
        assert "extra_body" not in body and "enable_thinking" not in body
        assert body["max_tokens"] == 32768
        return httpx.Response(200,json={"choices":[{"finish_reason":"stop","message":{
            "content":'{"status":"ok"}',"reasoning_content":"not the final answer"}}],
            "usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}})
    original=httpx.AsyncClient
    monkeypatch.setattr(httpx,"AsyncClient",lambda **kw: original(transport=httpx.MockTransport(handle),**kw))
    monkeypatch.setattr(providers,"LLM_CONCURRENCY",asyncio.Semaphore(5))
    result=await providers.QwenProvider("qwen-key").analyze([{"role":"user","content":"Return JSON"}])
    assert json.loads(result.content)=={"status":"ok"}
    assert result.provider=="qwen" and result.total_tokens==30


def test_qwen_endpoint_configuration(monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL","https://custom.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1/")
    assert providers.QwenProvider("key").base_url.endswith("/v1")
    for url in ["http://host/v1","https://key:secret@host/v1","https://host/v1?key=secret"]:
        monkeypatch.setenv("QWEN_BASE_URL",url)
        with pytest.raises(ValueError):
            providers.QwenProvider("key")


@pytest.mark.parametrize("choice",[{"finish_reason":"length","message":{"content":"partial"}},
                                    {"finish_reason":"stop","message":{"content":None}}])
async def test_qwen_rejects_truncated_or_empty_answers(monkeypatch,choice):
    original=httpx.AsyncClient
    monkeypatch.setattr(httpx,"AsyncClient",lambda **kw: original(transport=httpx.MockTransport(
        lambda req:httpx.Response(200,json={"choices":[choice]})),**kw))
    monkeypatch.setattr(providers,"LLM_CONCURRENCY",asyncio.Semaphore(5))
    with pytest.raises(ValueError):
        await providers.QwenProvider("key").analyze([{"role":"user","content":"Return JSON"}])


def test_qwen_context_budget_and_probe_prompt(monkeypatch):
    monkeypatch.delenv("QWEN_MODEL",raising=False)
    assert main._context_budget("qwen")==900_000
    monkeypatch.setenv("QWEN_MODEL","different-model")
    assert main._context_budget("qwen")==220_000
    class Probe:
        async def analyze(self,messages):
            assert "json" in str(messages).lower()
            return providers.LLMResponse('{}','qwen3.8-max','qwen',0,0,0,0)
    monkeypatch.setattr(main,"get_provider",lambda **kw:Probe())
    with TestClient(main.app) as client:
        assert client.post("/providers/test",json={"provider":"qwen","api_key":"key"}).status_code == 200
