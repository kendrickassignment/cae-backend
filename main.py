"""
Corporate Accountability Engine (CAE) — FastAPI Backend
Built for Act For Farmed Animals (AFFA) / Sinergia Animal International

Main application file. Provides REST API endpoints for:
 - PDF upload and parsing
 - AI-powered greenwashing analysis
 - Hybrid Gemini routing (3 Flash → 3.1 Pro escalation)
 - Multi-file merge analysis (/analyze-multi)
 - Results retrieval
 - Health checks and provider testing

Updated: Mar 2026 — Hybrid Gemini + Multi-file merge

Deploy on: Google Cloud Run (GCP)

Run locally: uvicorn main:app --reload --port 8000
"""

import os
import json
import uuid
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()

from geography import COUNTRIES, catalog, normalize_countries
from asia_analysis import normalize_asia_result, needs_regional_escalation

from pdf_parser import parse_pdf
from system_prompt import build_analysis_prompt
from llm_providers import get_provider, PROVIDERS

# ============================================================================
# CONFIG
# ============================================================================

MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_BATCH_FILES = 3
MAX_BATCH_TOTAL_MB = 75
MAX_BATCH_TOTAL_BYTES = MAX_BATCH_TOTAL_MB * 1024 * 1024

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# In-memory store for demo/prototype. In production, use Supabase.
reports_store: dict = {}
analysis_store: dict = {}

# ============================================================================
# APP SETUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("\n" + "=" * 60)
    print(" CORPORATE ACCOUNTABILITY ENGINE (CAE)")
    print(" Built for Sinergia Animal International / AFFA")
    print(" 9 Evasion Patterns + AI Confidence Check")
    print(" Hybrid Gemini: 3 Flash Preview → 3.1 Pro Preview Escalation")
    print(" Asia evidence scoring v1 + legacy Indonesia compatibility")
    print("=" * 60)
    print(f" LLM Provider: {os.getenv('LLM_PROVIDER', 'gemini')} (default)")
    print(f" Upload Dir: {UPLOAD_DIR.absolute()}")
    print(f" Results Dir: {RESULTS_DIR.absolute()}")
    print("=" * 60 + "\n")
    yield
    print("\nCAE shutting down.")

app = FastAPI(
    title="Corporate Accountability Engine (CAE)",
    description=(
        "Adversarial AI auditor for corporate sustainability reports. "
        "Detects greenwashing in cage-free egg commitments across Asia, with country-specific evidence and legacy Indonesia fields. "
        "9 evasion patterns + AI document confidence check. "
        "Hybrid Gemini routing: 3 Flash → 3.1 Pro escalation. "
        "Built for Sinergia Animal International / AFFA."
    ),
    version="3.0.0",
    lifespan=lifespan
)

# CORS — allow Lovable frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://id-preview--66fa6b17-db0f-4366-ba7e-e60ac0ceec2b.lovable.app",
        "https://cae-animals.lovable.app",
        "https://66fa6b17-db0f-4366-ba7e-e60ac0ceec2b.lovableproject.com",
        "https://cae-animals.com",
        "https://www.cae-animals.com",
        "https://preview--cae-animals.lovable.app",
        os.getenv("FRONTEND_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ReportStatus(BaseModel):
    id: str
    file_name: str
    company_name: str | None = None
    report_year: int | None = None
    status: str # uploading | uploaded | processing | analyzing | completed | failed
    page_count: int | None = None
    created_at: str
    error: str | None = None

class AnalysisResult(BaseModel):
    id: str
    report_id: str
    company_name: str | None = None
    report_year: int | None = None
    overall_risk_level: str | None = None
    overall_risk_score: int | None = None
    global_claim: str | None = None
    indonesia_mentioned: bool | None = None
    indonesia_status: str | None = None
    sea_countries_mentioned: list[str] = []
    sea_countries_excluded: list[str] = []
    binding_language_count: int = 0
    hedging_language_count: int = 0
    summary: str | None = None
    findings: list[dict] = []
    document_confidence: str | None = None
    document_confidence_reason: str | None = None
    scoring_breakdown: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_estimate_usd: float = 0.0
    analyzed_at: str | None = None

class UploadResponse(BaseModel):
    report_id: str
    file_name: str
    status: str
    message: str

class RegionalOptions(BaseModel):
    target_countries: list[str] = Field(default_factory=lambda: list(COUNTRIES), min_length=1, max_length=len(COUNTRIES))

    @field_validator("target_countries")
    @classmethod
    def check_countries(cls, values):
        return normalize_countries(values)


class AnalyzeRequest(RegionalOptions):
    report_id: str
    company_name: str | None = None
    report_year: int | None = None
    provider: str | None = None # gemini, qwen, openai
    api_key: str | None = None # Optional override

class AnalyzeMultiRequest(RegionalOptions):
    """Request to analyze multiple uploaded PDFs as one merged analysis."""
    report_ids: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_FILES)
    company_name: str | None = None
    report_year: int | None = None
    provider: str | None = None
    api_key: str | None = None

class ProviderTestRequest(BaseModel):
    provider: str
    api_key: str | None = None

# ============================================================================
# ROBUST JSON PARSER — handles all LLM output quirks
# ============================================================================

import re

def _unwrap_list(parsed) -> dict:
    """
    If the LLM returned a JSON array like [{...}] instead of {...},
    unwrap it to the first dict element. Common with Gemini preview models.
    """
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                return item
        raise ValueError(f"JSON array contained no dict objects. Got: {str(parsed)[:200]}")
    if isinstance(parsed, dict):
        return parsed
    raise ValueError(f"Expected dict or list, got {type(parsed).__name__}: {str(parsed)[:200]}")


def parse_llm_json(raw_content: str) -> dict:
    """
    Parse JSON from LLM output, handling common issues:
    1. Clean JSON (best case)
    2. JSON array unwrapping ([{...}] → {...})
    3. JSON wrapped in markdown code blocks (```json ... ```)
    4. JSON with leading/trailing text
    5. JSON with trailing commas
    6. JSON with comments
    """
    content = raw_content.strip()
    
    # Attempt 1: Direct parse (best case — Gemini responseMimeType: json)
    try:
        return _unwrap_list(json.loads(content))
    except json.JSONDecodeError:
        pass
    
    # Attempt 2: Strip markdown code blocks (```json ... ``` or ``` ... ```)
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?\s*```'
    code_match = re.search(code_block_pattern, content, re.DOTALL)
    if code_match:
        try:
            return _unwrap_list(json.loads(code_match.group(1).strip()))
        except json.JSONDecodeError:
            pass
    
    # Attempt 3: Find the outermost { ... } JSON object
    brace_depth = 0
    json_start = -1
    json_end = -1
    
    for i, char in enumerate(content):
        if char == '{':
            if brace_depth == 0:
                json_start = i
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
            if brace_depth == 0 and json_start >= 0:
                json_end = i + 1
                break
    
    if json_start >= 0 and json_end > json_start:
        json_str = content[json_start:json_end]
        try:
            return _unwrap_list(json.loads(json_str))
        except json.JSONDecodeError:
            # Attempt 4: Fix trailing commas (common LLM mistake)
            fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
            try:
                return _unwrap_list(json.loads(fixed))
            except json.JSONDecodeError:
                pass
    
    # Do not persist raw model output or expose document contents in errors.
    raise ValueError("LLM did not return valid JSON.")

# ============================================================================
# HARD VALIDATION — Python code that AI CANNOT override
# ============================================================================

VALID_FINDING_TYPES = {
    "hedging_language", "geographic_exclusion", "strategic_silence",
    "franchise_firewall", "availability_clause", "timeline_deferral",
    "silent_delisting", "corporate_ghosting", "commitment_downgrade",
    "binding_commitment"
}
VALID_SEVERITIES = {"critical", "high", "medium", "low", "info"}
VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_INDO_STATUS = {"compliant", "excluded", "silent", "partial", "deferred"}


def score_to_level(score: int) -> str:
    """Convert risk score to risk level. This is LAW — no exceptions."""
    if score <= 30:
        return "low"
    elif score <= 55:
        return "medium"
    elif score <= 79:
        return "high"
    else:
        return "critical"


def deduplicate_findings(findings: list[dict]) -> list[dict]:
    """Remove exact duplicate evidence only; never collapse countries or documents."""
    seen, merged = set(), []
    for finding in findings:
        key = json.dumps({k: v for k, v in finding.items() if k != "id"}, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            merged.append(dict(finding, id=f"f-{len(merged) + 1}"))
    return merged


def validate_analysis_result(result_data: dict, fallback_company: str | None = None) -> dict:
    """
    Validate and sanitize all AI output. This function enforces:
    1. Score is integer 0-100
    2. Level STRICTLY matches score thresholds (AI cannot override)
    3. All enums are valid values
    4. All findings have required fields and valid types
    5. Counts are non-negative integers
    
    This is Python code, not an AI instruction. It is DETERMINISTIC.
    """
    # --- Score validation ---
    raw_score = result_data.get("overall_risk_score", 0)
    try:
        score = int(raw_score)
    except (TypeError, ValueError):
        score = 50
    score = max(0, min(100, score))  # Clamp 0-100

    # --- Level enforcement (THE LAW) ---
    enforced_level = score_to_level(score)
    ai_level = str(result_data.get("overall_risk_level", "")).lower()
    if ai_level != enforced_level:
        print(
            f"\u26a0\ufe0f  SCORE-LEVEL OVERRIDE: AI said score={score} level='{ai_level}' "
            f"\u2192 FORCED to '{enforced_level}' "
            f"(company: {result_data.get('company_name', 'unknown')})"
        )

    # --- Document confidence ---
    raw_conf = str(result_data.get("document_confidence", "medium")).lower()
    confidence = raw_conf if raw_conf in VALID_CONFIDENCE else "medium"

    # --- Indonesia status ---
    raw_indo = str(result_data.get("indonesia_status", "silent")).lower()
    indo_status = raw_indo if raw_indo in VALID_INDO_STATUS else "silent"

    # --- Integer counts ---
    def safe_int(val, default=0):
        try:
            return max(0, int(val))
        except (TypeError, ValueError):
            return default

    binding_count = safe_int(result_data.get("binding_language_count"))
    hedging_count = safe_int(result_data.get("hedging_language_count"))

    # --- Findings validation ---
    raw_findings = result_data.get("findings", [])
    if not isinstance(raw_findings, list):
        raw_findings = []

    validated_findings = []
    for i, f in enumerate(raw_findings):
        if not isinstance(f, dict):
            continue

        ftype = str(f.get("finding_type", "hedging_language")).lower()
        if ftype not in VALID_FINDING_TYPES:
            ftype = "hedging_language"

        sev = str(f.get("severity", "medium")).lower()
        if sev not in VALID_SEVERITIES:
            sev = "medium"

        page = 0
        raw_page = f.get("page_number", 0)
        try:
            page = max(0, int(raw_page))
        except (TypeError, ValueError):
            page = 0

        validated_findings.append({
            "id": f"f-{i + 1}",
            "finding_type": ftype,
            "severity": sev,
            "title": str(f.get("title", "Untitled Finding")),
            "description": str(f.get("description", "")),
            "exact_quote": str(f.get("exact_quote", "N/A")),
            "page_number": page,
            "section": f.get("section"),
            "paragraph": f.get("paragraph"),
            "country_affected": f.get("country_affected"),
            "source_document": f.get("source_document")
        })

    # --- Deduplicate findings (safety net for AI over-reporting) ---
    pre_dedup_count = len(validated_findings)
    validated_findings = deduplicate_findings(validated_findings)
    if len(validated_findings) != pre_dedup_count:
        print(
            f"\U0001f9f9 DEDUP: {pre_dedup_count} findings \u2192 {len(validated_findings)} findings "
            f"(company: {result_data.get('company_name', 'unknown')})"
        )

    # --- Boolean validation ---
    indo_mentioned = result_data.get("indonesia_mentioned")
    if not isinstance(indo_mentioned, bool):
        indo_mentioned = False

    # --- SEA countries validation ---
    sea_mentioned = result_data.get("sea_countries_mentioned", [])
    if not isinstance(sea_mentioned, list):
        sea_mentioned = []
    sea_excluded = result_data.get("sea_countries_excluded", [])
    if not isinstance(sea_excluded, list):
        sea_excluded = []

    print(
        f"\u2705 VALIDATED: {result_data.get('company_name', 'unknown')} "
        f"\u2014 score={score} level={enforced_level} "
        f"confidence={confidence} indo={indo_status} "
        f"findings={len(validated_findings)}"
    )

    return {
        "company_name": result_data.get("company_name") or fallback_company or "Unknown Company",
        "report_year": result_data.get("report_year"),
        "overall_risk_score": score,
        "overall_risk_level": enforced_level,
        "global_claim": result_data.get("global_claim", "No cage-free commitment found."),
        "indonesia_mentioned": indo_mentioned,
        "indonesia_status": indo_status,
        "sea_countries_mentioned": sea_mentioned,
        "sea_countries_excluded": sea_excluded,
        "binding_language_count": binding_count,
        "hedging_language_count": hedging_count,
        "summary": result_data.get("summary", "Analysis completed but no summary generated."),
        "findings": validated_findings,
        "document_confidence": confidence,
        "document_confidence_reason": result_data.get("document_confidence_reason", ""),
        "scoring_breakdown": result_data.get("scoring_breakdown", "No breakdown provided by AI."),
    }


# ============================================================================
# BACKGROUND ANALYSIS TASK
# ============================================================================

# Jobs and HTTP calls have DISTINCT limits: acquiring the same semaphore twice
# deadlocks when five jobs all hold a slot before calling the provider.
JOB_CONCURRENCY = asyncio.Semaphore(5)


def _context_budget(provider):
    return {"gemini": 900_000, "qwen": 220_000, "openai": 110_000}.get(provider, 110_000)


async def _run_reports(report_ids, company_name, report_year, provider_name, api_key, target_countries):
    async with JOB_CONCURRENCY:
        analysis_id = str(uuid.uuid4())
        try:
            sources, documents, texts = {}, [], []
            complete_text = True
            for number, rid in enumerate(report_ids, 1):
                report = reports_store[rid]
                report["status"] = "processing"
                parsed = parse_pdf(report["file_path"])
                if not parsed.pages or not parsed.full_text_with_markers.strip():
                    raise ValueError("No extractable text. OCR the PDF before uploading.")
                source_id = f"document-{number}"
                sources[source_id] = {page.page_number: page.full_text for page in parsed.pages}
                complete_text = complete_text and len(parsed.pages) == parsed.page_count
                documents.append({"id": source_id, "report_id": rid, "file_name": report["file_name"],
                                  "page_count": parsed.page_count, "extracted_pages": len(parsed.pages)})
                # Machine source IDs are unambiguous even for duplicate filenames.
                texts.append(f"\n=== DOCUMENT {source_id} ===\n" + parsed.full_text_with_markers)
                report["page_count"] = parsed.page_count
                report["status"] = "analyzing"
            effective_provider = provider_name or os.getenv("LLM_PROVIDER", "gemini")
            document_text = "\n".join(texts)
            messages = build_analysis_prompt(document_text, "Uploaded report bundle",
                                             sum(d["page_count"] for d in documents),
                                             target_countries, complete_text)
            # UTF-8 byte count is a conservative bound for multilingual inputs.
            # Never silently discard middle pages and then claim a country is absent.
            if sum(len(m["content"].encode("utf-8")) for m in messages) > _context_budget(effective_provider):
                raise ValueError("Document exceeds the conservative provider context budget. Use fewer files or a larger-context provider.")

            from llm_providers import GeminiProvider
            responses = []
            fallback = False
            if effective_provider == "gemini":
                flash_model = os.getenv("GEMINI_FLASH_MODEL", "gemini-3-flash-preview")
                pro_model = os.getenv("GEMINI_PRO_MODEL", "gemini-3.1-pro-preview")
                response = await GeminiProvider(api_key=api_key, model_name=flash_model).analyze(messages)
                responses.append(response)
                raw = parse_llm_json(response.content)
                regional = normalize_asia_result(raw, target_countries, sources, complete_text)
                legacy = validate_analysis_result(raw, company_name)
                escalate = needs_regional_escalation(regional) or legacy["overall_risk_score"] >= 56
                if escalate:
                    try:
                        pro = await GeminiProvider(api_key=api_key, model_name=pro_model).analyze(messages)
                        responses.append(pro)
                        pro_raw = parse_llm_json(pro.content)
                        pro_regional = normalize_asia_result(pro_raw, target_countries, sources, complete_text)
                        raw, regional, response = pro_raw, pro_regional, pro
                    except Exception:
                        fallback = True
            else:
                response = await get_provider(effective_provider, api_key).analyze(messages)
                responses.append(response)
                raw = parse_llm_json(response.content)
                regional = normalize_asia_result(raw, target_countries, sources, complete_text)

            validated = validate_analysis_result(raw, company_name)
            if fallback:
                regional["validation_warnings"].append("Deep analysis unavailable; fast-scan result retained for human review.")
                validated["document_confidence"] = "low"
            if not complete_text:
                regional["validation_warnings"].append("Some PDF pages have no extractable text; OCR review required. Silence scoring disabled.")
                validated["document_confidence"] = "low"
            analysis_result = {
                **validated, **regional, "id": analysis_id, "report_id": report_ids[0],
                "report_year": validated["report_year"] or report_year,
                "source_documents": documents, "llm_provider": effective_provider,
                "llm_model": response.model, "hybrid_fallback": fallback,
                "input_tokens": sum(r.input_tokens for r in responses),
                "output_tokens": sum(r.output_tokens for r in responses),
                "cost_estimate_usd": sum(r.cost_estimate_usd for r in responses),
                "cost_estimate_available": effective_provider == "openai",
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
            }
            if len(report_ids) > 1:
                analysis_result.update(merged_report_ids=report_ids, merged_file_count=len(report_ids))
            result_path = RESULTS_DIR / f"{analysis_id}.json"
            temporary = result_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(analysis_result, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(result_path)
            analysis_store[analysis_id] = analysis_result
            for rid in report_ids:
                reports_store[rid].update(status="completed", analysis_id=analysis_id,
                                          company_name=validated["company_name"])
        except Exception as exc:
            # Never return provider URLs, API keys, or raw document text in errors.
            safe_message = "Analysis failed. Check PDF text, provider access and context limits, then retry."
            if isinstance(exc, ValueError) and str(exc).startswith(("No extractable text.", "Document exceeds", "Provider returned no")):
                safe_message = str(exc)
            for rid in report_ids:
                reports_store[rid].update(status="failed", error=safe_message)


async def run_analysis(report_id, file_path, company_name, report_year, provider_name, api_key, target_countries=None):
    await _run_reports([report_id], company_name, report_year, provider_name, api_key, target_countries or list(COUNTRIES))


async def run_multi_analysis(primary_report_id, report_ids, company_name, report_year, provider_name, api_key, target_countries=None):
    await _run_reports(report_ids, company_name, report_year, provider_name, api_key, target_countries or list(COUNTRIES))


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    """Health check and API info. Supports both GET and HEAD for Cloud Run health checks."""
    body = {
        "name": "Corporate Accountability Engine (CAE)",
        "version": "3.0.0",
        "organization": "Sinergia Animal International / AFFA",
        "status": "operational",
        "evasion_patterns": 9,
        "features": [
            "9 evasion patterns",
            "AI confidence check",
            "Hybrid Gemini: 3 Flash → 3.1 Pro escalation",
            "Multi-file merge analysis",
        ],
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /upload",
            "analyze": "POST /analyze",
            "analyze_multi": "POST /analyze-multi",
            "status": "GET /reports/{report_id}",
            "result": "GET /analysis/{analysis_id}",
            "all_reports": "GET /reports",
            "providers": "GET /providers",
            "test_provider": "POST /providers/test"
        }
    }

    if request.method == "HEAD":
        return Response(status_code=200)

    return body

@app.api_route("/health", methods=["GET", "HEAD"])
async def health(request: Request):
    """Simple health check for uptime monitoring."""
    body = {"status": "ok", "version": "3.0.0", "timestamp": datetime.now(timezone.utc).isoformat()}

    if request.method == "HEAD":
        return Response(status_code=200)

    return body

@app.get("/countries")
async def list_countries():
    """Supported country/area codes and transparent geographic scope."""
    return catalog()


@app.get("/capabilities")
async def capabilities():
    return {"schema_version": "3.0", "default_scope": "asia",
            "country_count": len(COUNTRIES), "country_selection": "target_countries",
            "legacy_overall_risk_scope": "Indonesia", "regional_scoring_version": "asia-evidence-v1",
            "output_language": "en", "source_quotes": "original language",
            "ocr": False, "legal_compliance_checks": False,
            "storage": "local files and in-memory report status; single instance only"}


# --- UPLOAD ---

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload dan simpan satu file PDF dengan perlindungan memori (Max 25MB)."""
    
    # Validasi 1: Format File
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang diperbolehkan.")
        
    # Validasi 2: Ukuran File (Baca per 1MB agar RAM server tidak jebol)
    file_size = 0
    while chunk := await file.read(1024 * 1024): 
        file_size += len(chunk)
        if file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413, 
                detail=f"File terlalu besar! Maksimal ukuran file adalah {MAX_FILE_SIZE_MB} MB."
            )
            
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty PDF.")
    await file.seek(0)
    if not (await file.read(1024)).lstrip().startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Invalid PDF signature.")

    # Kembalikan pointer pembacaan ke awal file
    await file.seek(0)
    
    # --- LOGIKA PENYIMPANAN YANG AKTIF ---
    report_id = f"rep_{uuid.uuid4().hex}"
    file_path = UPLOAD_DIR / f"{report_id}.pdf"
    
    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    # Simpan status ke memory store
    reports_store[report_id] = {
        "id": report_id,
        "file_name": file.filename,
        "file_path": str(file_path),
        "company_name": None,
        "report_year": None,
        "status": "uploaded",
        "page_count": 0,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    return UploadResponse(
        report_id=report_id,
        file_name=file.filename,
        status="success",
        message="File valid dan berhasil diunggah."
    )

# --- ANALYZE (single file) ---

@app.post("/analyze")
async def analyze_report(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Trigger AI analysis on an uploaded report.
    Uses hybrid Gemini routing: 3 Flash → 3.1 Pro escalation.

    The analysis runs in the background. Poll GET /reports/{report_id}
    to check status, then GET /analysis/{analysis_id} for results.
    """
    report_id = request.report_id

    if report_id not in reports_store:
        raise HTTPException(status_code=404, detail="Report not found. Upload first via POST /upload.")

    report = reports_store[report_id]

    if report["status"] in ("processing", "analyzing"):
        raise HTTPException(status_code=409, detail="Analysis already in progress.")

    # Validate provider if specified
    if request.provider and request.provider not in PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider: {request.provider}. Available: {', '.join(PROVIDERS.keys())}"
        )

    # Update metadata if provided
    if request.company_name:
        report["company_name"] = request.company_name
    if request.report_year:
        report["report_year"] = request.report_year

    report["status"] = "processing"
    report.pop("error", None)
    # Launch background analysis
    background_tasks.add_task(
        run_analysis,
        report_id=report_id,
        file_path=report["file_path"],
        company_name=request.company_name,
        report_year=request.report_year,
        provider_name=request.provider,
        api_key=request.api_key,
        target_countries=request.target_countries
    )

    return {
        "report_id": report_id,
        "status": "processing",
        "message": "Analysis started (Hybrid Gemini). Poll GET /reports/{report_id} for status updates.",
        "provider": request.provider or os.getenv("LLM_PROVIDER", "gemini")
    }

# --- ANALYZE MULTI (merge multiple files) ---

@app.post("/analyze-multi")
async def analyze_multi_report(request: AnalyzeMultiRequest, background_tasks: BackgroundTasks):
    """
    Trigger AI analysis on MULTIPLE uploaded reports, merged into ONE analysis.

    Upload each PDF separately via POST /upload, then call this endpoint
    with all report_ids. The system will:
    1. Parse all PDFs
    2. Merge extracted text with document separators
    3. Run ONE hybrid Gemini analysis on the combined text
    4. Store ONE analysis result

    Poll GET /reports/{primary_report_id} for status (first report_id in the list).
    Max: 3 files per request.
    """
    if len(set(request.report_ids)) != len(request.report_ids):
        raise HTTPException(status_code=422, detail="Duplicate report_ids are not allowed.")
    # Validate all report_ids exist
    for rid in request.report_ids:
        if rid not in reports_store:
            raise HTTPException(
                status_code=404,
                detail=f"Report {rid} not found. Upload all files first via POST /upload."
            )

    # Check none are already processing
    for rid in request.report_ids:
        if reports_store[rid]["status"] in ("processing", "analyzing"):
            raise HTTPException(
                status_code=409,
                detail=f"Report {rid} is already being analyzed."
            )

    if request.provider and request.provider not in PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider: {request.provider}. Available: {', '.join(PROVIDERS.keys())}"
        )

    # Update metadata on all reports
    for rid in request.report_ids:
        if request.company_name:
            reports_store[rid]["company_name"] = request.company_name
        if request.report_year:
            reports_store[rid]["report_year"] = request.report_year

    for rid in request.report_ids:
        reports_store[rid]["status"] = "processing"
        reports_store[rid].pop("error", None)
    primary_report_id = request.report_ids[0]

    background_tasks.add_task(
        run_multi_analysis,
        primary_report_id=primary_report_id,
        report_ids=request.report_ids,
        company_name=request.company_name,
        report_year=request.report_year,
        provider_name=request.provider,
        api_key=request.api_key,
        target_countries=request.target_countries
    )

    return {
        "report_id": primary_report_id,
        "status": "processing",
        "merged_files": len(request.report_ids),
        "message": (
            f"Multi-analysis started ({len(request.report_ids)} files merged). "
            f"Poll GET /reports/{primary_report_id} for status updates."
        ),
        "provider": request.provider or os.getenv("LLM_PROVIDER", "gemini")
    }

# --- REPORT STATUS ---

@app.get("/reports/{report_id}")
async def get_report_status(report_id: str):
    """Get the current status of a report (uploading/processing/completed/failed)."""
    if report_id not in reports_store:
        raise HTTPException(status_code=404, detail="Report not found.")

    report = reports_store[report_id]
    response = {
        "id": report["id"],
        "file_name": report["file_name"],
        "company_name": report["company_name"],
        "report_year": report["report_year"],
        "status": report["status"],
        "page_count": report["page_count"],
        "created_at": report["created_at"],
    }

    if report["status"] == "completed" and report.get("analysis_id"):
        response["analysis_id"] = report["analysis_id"]
        response["analysis_url"] = f"/analysis/{report['analysis_id']}"

    if report["status"] == "failed":
        response["error"] = report.get("error")

    return response

# --- ALL REPORTS ---

@app.get("/reports")
async def list_reports():
    """List all uploaded reports and their statuses."""
    reports = list(reports_store.values())
    safe_reports = []
    for r in reports:
        safe = {k: v for k, v in r.items() if k != "file_path"}
        safe_reports.append(safe)

    return {
        "total": len(safe_reports),
        "reports": sorted(safe_reports, key=lambda x: x["created_at"], reverse=True)
    }

# --- ANALYSIS RESULTS ---

@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get the full analysis results including all findings."""
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", analysis_id):
        raise HTTPException(status_code=404, detail="Analysis not found.")
    if analysis_id not in analysis_store:
        # Try loading from disk
        result_path = RESULTS_DIR / f"{analysis_id}.json"
        if result_path.exists():
            with open(result_path, encoding="utf-8") as f:
                analysis_store[analysis_id] = json.load(f)
        else:
            raise HTTPException(status_code=404, detail="Analysis not found.")

    return analysis_store[analysis_id]

# --- EXPORT ---

@app.get("/analysis/{analysis_id}/export")
async def export_analysis_csv(analysis_id: str):
    """Export findings as CSV-formatted data."""
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", analysis_id):
        raise HTTPException(status_code=404, detail="Analysis not found.")
    if analysis_id not in analysis_store:
        result_path = RESULTS_DIR / f"{analysis_id}.json"
        if result_path.exists():
            with open(result_path, encoding="utf-8") as f:
                analysis_store[analysis_id] = json.load(f)
        else:
            raise HTTPException(status_code=404, detail="Analysis not found.")

    analysis = analysis_store[analysis_id]
    findings = analysis.get("findings", [])

    headers = [
        "finding_type", "severity", "title", "description",
        "exact_quote", "page_number", "section", "country_affected", "source_document"
    ]

    csv_lines = [",".join(headers)]
    for f in findings:
        row = []
        for h in headers:
            value = str(f.get(h, "") or "")
            if value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            value = value.replace('"', '""')
            row.append(f'"{value}"')
        csv_lines.append(",".join(row))

    csv_content = "\n".join(csv_lines)

    return JSONResponse(
        content={
            "csv": csv_content,
            "company": analysis.get("company_name"),
            "findings_count": len(findings),
            "risk_level": analysis.get("overall_risk_level"),
            "document_confidence": analysis.get("document_confidence")
        }
    )

# --- PROVIDERS ---

@app.get("/providers")
async def list_providers():
    """List available LLM providers and their details."""
    return {
        "default": os.getenv("LLM_PROVIDER", "gemini"),
        "available": {
            "gemini": {
                "name": "Google Gemini (Hybrid)",
                "models": {
                    "fast": os.getenv("GEMINI_FLASH_MODEL", "gemini-3-flash-preview"),
                    "pro": os.getenv("GEMINI_PRO_MODEL", "gemini-3.1-pro-preview")
                },
                "routing": "Auto-escalates to 3.1 Pro Preview for high-risk reports",
                "free": False,
                "context_window": "1M tokens",
                "best_for": "Large documents (200+ pages), single-pass analysis",
                "get_key": "https://aistudio.google.com/apikey"
            },
            "qwen": {
                "name": "Qwen 3.5-397B-A17B (DashScope)",
                "model": os.getenv("QWEN_MODEL", "qwen3.5-397b-a17b"),
                "free": False,
                "context_window": "256k tokens",
                "best_for": "Backup provider, large reasoning model",
                "get_key": "https://dashscope.console.aliyun.com/"
            },
            "openai": {
                "name": "OpenAI GPT-4o",
                "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
                "free": False,
                "context_window": "128k tokens",
                "cost": "Usage-based; confirm current provider billing",
                "best_for": "Highest quality (if budget allows)",
                "get_key": "https://platform.openai.com/api-keys"
            }
        }
    }

@app.post("/providers/test")
async def test_provider(request: ProviderTestRequest):
    """Test if a provider API key is valid and working."""
    try:
        provider = get_provider(
            provider_name=request.provider,
            api_key=request.api_key
        )

        test_messages = [
            {"role": "system", "content": "You are a test bot. Respond with exactly: {\"status\": \"ok\"}"},
            {"role": "user", "content": "Test connection."}
        ]

        response = await provider.analyze(test_messages)

        return {
            "status": "success",
            "provider": request.provider,
            "model": response.model,
            "message": "Connection successful! Provider is ready."
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "provider": request.provider,
                "error": "Provider connection failed. Check credentials, model access and service availability.",
                "message": "Connection failed. Check your API key."
            }
        )

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
