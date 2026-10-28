import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fitz
import httpx
import pytest
from fastapi.testclient import TestClient
import main
import llm_providers
from llm_providers import LLMResponse
from pdf_parser import chunk_document

QUOTE = "Japan is excluded from our cage-free egg program."


def response(raw=None, model="test-model"):
    raw = raw or {"company_name":"Fixture Foods", "overall_risk_score":20, "indonesia_mentioned":True,
                  "indonesia_status":"partial", "findings":[],
                  "country_assessments":[{"country_code":"JP", "status":"excluded", "evidence":[{
                      "source_document":"document-1", "page_number":1, "exact_quote":QUOTE}]}]}
    return LLMResponse(json.dumps(raw), model, "openai", 10, 5, 15, 0.01)


def pdf_bytes(text=QUOTE, blank_page=False):
    with fitz.open() as doc:
        page = doc.new_page()
        if text:
            page.insert_text((72,72),text)
        if blank_page:
            doc.new_page()
        return doc.tobytes()


@pytest.fixture
def client(tmp_path, monkeypatch):
    main.reports_store.clear()
    main.analysis_store.clear()
    monkeypatch.setattr(main,"UPLOAD_DIR",tmp_path)
    monkeypatch.setattr(main,"RESULTS_DIR",tmp_path)
    monkeypatch.setattr(main,"JOB_CONCURRENCY",asyncio.Semaphore(5))
    monkeypatch.setattr(llm_providers,"LLM_CONCURRENCY",asyncio.Semaphore(5))
    monkeypatch.setattr(main,"get_provider",lambda *a, **kw: SimpleNamespace(analyze=AsyncMock(return_value=response())))
    with TestClient(main.app) as test_client:
        yield test_client


def upload(client, data=None, filename="sustainability.pdf"):
    res = client.post("/upload", files={"file":(filename, data if data is not None else pdf_bytes(), "application/pdf")})
    assert res.status_code == 200
    return res.json()["report_id"]


def analyze(client, rid, **options):
    return client.post("/analyze",json={"report_id":rid,"provider":"openai","target_countries":["JP","ID"],**options})


def test_single_upload_analysis_poll_export_and_disk_reload(client):
    rid = upload(client)
    assert analyze(client,rid).status_code == 200
    state = client.get(f"/reports/{rid}").json()
    assert state["status"] == "completed"
    main.analysis_store.clear()
    data = client.get(state["analysis_url"]).json()
    assert data["schema_version"] == "3.0"
    assert data["regional_risk_score"] == 30
    assert data["overall_risk_score"] == 20
    assert data["indonesia_status"] == "partial"
    assert data["sea_countries_mentioned"] == []
    assert data["source_documents"][0]["file_name"] == "sustainability.pdf"
    assert client.get(state["analysis_url"] + "/export").json()["findings_count"] == 0


def test_legacy_request_defaults_to_all_asia(client):
    rid = upload(client)
    assert client.post("/analyze",json={"report_id":rid,"provider":"openai"}).status_code == 200
    state = client.get(f"/reports/{rid}").json()
    data = client.get(state["analysis_url"]).json()
    assert data["coverage"]["requested"] == 51
    assert "ID" in data["target_countries"]


def test_multi_file_preserves_source_ids_and_local_pages(client,monkeypatch):
    rids = [upload(client,filename="same-name.pdf"), upload(client,pdf_bytes("Indonesia has a binding commitment."),filename="same-name.pdf")]
    raw = {"country_assessments":[{"country_code":"ID","status":"compliant","evidence":[{
        "source_document":"document-2","page_number":1,"exact_quote":"Indonesia has a binding commitment."}]}]}
    monkeypatch.setattr(main,"get_provider",lambda *a,**kw:SimpleNamespace(analyze=AsyncMock(return_value=response(raw))))
    res = client.post("/analyze-multi",json={"report_ids":rids,"provider":"openai","target_countries":["ID"]})
    assert res.status_code == 200
    states = [client.get(f"/reports/{rid}").json() for rid in rids]
    assert states[0]["analysis_id"] == states[1]["analysis_id"]
    data = client.get(states[0]["analysis_url"]).json()
    assert data["merged_file_count"] == 2
    assert data["country_assessments"][0]["evidence"][0]["source_document"] == "document-2"


@pytest.mark.parametrize("countries",[[],["US"],["ASIA"],None])
def test_invalid_country_requests(client,countries):
    rid = upload(client)
    assert analyze(client,rid,target_countries=countries).status_code == 422


def test_invalid_uploads_and_size_limit(client,monkeypatch):
    for name,data in [("report.pdf",b""),("report.pdf",b"not a PDF"),("report.txt",pdf_bytes())]:
        assert client.post("/upload",files={"file":(name,data)}).status_code == 400
    monkeypatch.setattr(main,"MAX_FILE_SIZE_BYTES",10)
    assert client.post("/upload",files={"file":("large.pdf",pdf_bytes())}).status_code == 413


def test_duplicate_multi_and_already_processing(client):
    rid = upload(client)
    assert client.post("/analyze-multi",json={"report_ids":[rid,rid]}).status_code == 422
    main.reports_store[rid]["status"] = "processing"
    assert analyze(client,rid).status_code == 409


def test_scanned_pdf_fails_without_calling_model(client,monkeypatch):
    call = AsyncMock(return_value=response())
    monkeypatch.setattr(main,"get_provider",lambda *a,**kw:SimpleNamespace(analyze=call))
    rid = upload(client,pdf_bytes(""))
    analyze(client,rid)
    state = client.get(f"/reports/{rid}").json()
    assert state["status"] == "failed"
    assert "OCR" in state["error"]
    call.assert_not_awaited()


def test_partial_extraction_and_context_limit(client,monkeypatch):
    rid = upload(client,pdf_bytes(blank_page=True))
    analyze(client,rid)
    data = client.get(client.get(f"/reports/{rid}").json()["analysis_url"]).json()
    assert data["coverage"]["complete_text"] is False
    assert data["document_confidence"] == "low"
    monkeypatch.setattr(main,"_context_budget",lambda p:10)
    analyze(client,rid)
    state = client.get(f"/reports/{rid}").json()
    assert state["status"] == "failed"
    assert "context budget" in state["error"]


def test_provider_errors_do_not_leak_keys(client,monkeypatch):
    call = AsyncMock(side_effect=RuntimeError("https://provider.test?key=SECRET_API_KEY"))
    monkeypatch.setattr(main,"get_provider",lambda *a,**kw:SimpleNamespace(analyze=call))
    rid = upload(client)
    analyze(client,rid)
    assert "SECRET" not in client.get(f"/reports/{rid}").text
    assert "SECRET" not in client.post("/providers/test",json={"provider":"openai","api_key":"SECRET"}).text


def test_regional_hybrid_escalation_and_fallback(client,monkeypatch):
    calls = []
    class FakeGemini:
        def __init__(self,api_key,model_name):
            self.model = model_name
        async def analyze(self,messages):
            calls.append(self.model)
            if "pro" in self.model:
                raise RuntimeError("provider unavailable")
            return response(model=self.model)
    monkeypatch.setattr(llm_providers,"GeminiProvider",FakeGemini)
    rid = upload(client)
    analyze(client,rid,provider="gemini")
    data = client.get(client.get(f"/reports/{rid}").json()["analysis_url"]).json()
    assert len(calls) == 2  # Japan excluded even though legacy Indonesia score is low
    assert data["hybrid_fallback"] is True
    assert data["document_confidence"] == "low"


async def test_five_simultaneous_jobs_do_not_deadlock(tmp_path,monkeypatch):
    monkeypatch.setattr(main,"JOB_CONCURRENCY",asyncio.Semaphore(5))
    monkeypatch.setattr(llm_providers,"LLM_CONCURRENCY",asyncio.Semaphore(5))
    monkeypatch.setattr(main,"RESULTS_DIR",tmp_path)
    class FakeProvider:
        async def analyze(self,messages):
            async with llm_providers.LLM_CONCURRENCY:
                await asyncio.sleep(0.01)
                return response()
    monkeypatch.setattr(main,"get_provider",lambda *a,**kw:FakeProvider())
    rids = [f"job-{i}" for i in range(5)]
    for rid in rids:
        path = tmp_path / f"{rid}.pdf"
        path.write_bytes(pdf_bytes())
        main.reports_store[rid] = {"file_path":str(path),"file_name":path.name}
    await asyncio.wait_for(asyncio.gather(*[main._run_reports([rid],None,None,"openai",None,["JP"]) for rid in rids]),timeout=5)
    assert all(main.reports_store[rid]["status"] == "completed" for rid in rids)


def test_dedup_does_not_merge_countries_or_documents():
    rows = [{"finding_type":"hedging_language","title":"Same title","country_affected":code,"source_document":doc}
            for code in ["JP","ID"] for doc in ["document-1","document-2"]]
    assert len(main.deduplicate_findings(rows + [rows[0]])) == 4


def test_chunking_terminates_and_validates_overlap():
    parsed = SimpleNamespace(full_text_with_markers="a"*50,page_count=1)
    chunks = chunk_document(parsed,20,5)
    assert len(chunks) == 3
    with pytest.raises(ValueError):
        chunk_document(parsed,20,20)


def test_capabilities_health_and_cors(client):
    assert client.get("/countries").json()["version"] == "asia-2026-10"
    assert client.get("/capabilities").json()["ocr"] is False
    assert client.get("/health").json()["version"] == "3.0.0"
    assert client.head("/").status_code == 200
    res = client.options("/analyze",headers={"Origin":"https://cae-animals.com","Access-Control-Request-Method":"POST"})
    assert res.headers["access-control-allow-origin"] == "https://cae-animals.com"
    assert client.get("/analysis/invalid-id").status_code == 404


def test_gemini_key_uses_header(monkeypatch):
    def handle(req):
        assert "key=" not in str(req.url)
        assert req.headers["x-goog-api-key"] == "test-secret"
        return httpx.Response(200,json={"candidates":[{"content":{"parts":[{"text":"{}"}]}}]})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx,"AsyncClient",lambda **kw: original(transport=httpx.MockTransport(handle),**kw))
    monkeypatch.setattr(llm_providers,"LLM_CONCURRENCY",asyncio.Semaphore(5))
    asyncio.run(llm_providers.GeminiProvider(api_key="test-secret").analyze([{"role":"user","content":"Test"}]))
