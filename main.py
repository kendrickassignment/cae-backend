"""
Corporate Accountability Engine (CAE) — FastAPI Backend
Built for Act For Farmed Animals (AFFA) / Sinergia Animal International

Main application file. Provides REST API endpoints for:
 - PDF upload and parsing
 - AI-powered greenwashing analysis
 - Results retrieval
 - Health checks and provider testing

Updated: Feb 2026 — Added document_confidence support, 9 evasion patterns

Deploy FREE on: Render.com, Railway.app, or Fly.io

Run locally: uvicorn main:app --reload --port 8000
"""

import os
import json
import uuid
import shutil
import asyncio
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from pdf_parser import parse_pdf, chunk_document, ParsedDocument
from system_prompt import build_analysis_prompt
from llm_providers import get_provider, LLMResponse, PROVIDERS

# ============================================================================
# CONFIG
# ============================================================================

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
    print(" Strict Scoring Algorithm v2.1")
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
        "Detects greenwashing in cage-free egg commitments with focus on Southeast Asia. "
        "9 evasion patterns + AI document confidence check. "
        "Built for Sinergia Animal International / AFFA."
    ),
    version="2.1.0",
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
        "https://truthextracted.com",
        "https://www.truthextracted.com",
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

class AnalyzeRequest(BaseModel):
    report_id: str
    company_name: str | None = None
    report_year: int | None = None
    provider: str | None = None # gemini, groq, mistral, openai
    api_key: str | None = None # Optional override

class ProviderTestRequest(BaseModel):
    provider: str
    api_key: str | None = None

# ============================================================================
# ROBUST JSON PARSER — handles all LLM output quirks
# ============================================================================

import re

def parse_llm_json(raw_content: str) -> dict:
    """
    Parse JSON from LLM output, handling common issues:
    1. Clean JSON (best case)
    2. JSON wrapped in markdown code blocks (```json ... ```)
    3. JSON with leading/trailing text
    4. JSON with trailing commas
    5. JSON with comments
    """
    content = raw_content.strip()
    
    # Attempt 1: Direct parse (best case — Gemini responseMimeType: json)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Attempt 2: Strip markdown code blocks (```json ... ``` or ``` ... ```)
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?\s*```'
    code_match = re.search(code_block_pattern, content, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1).strip())
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
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Attempt 4: Fix trailing commas (common LLM mistake)
            fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
    
    # Attempt 5: If all else fails, log what we got and raise
    debug_path = RESULTS_DIR / f"debug_raw_response_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(debug_path, 'w') as f:
        f.write(f"RAW LLM RESPONSE ({len(content)} chars):\n")
        f.write(f"First 500 chars: {content[:500]}\n")
        f.write(f"Last 500 chars: {content[-500:]}\n")
        f.write(f"\nFULL CONTENT:\n{content}")
    
    raise ValueError(
        f"LLM did not return valid JSON. Response length: {len(content)} chars. "
        f"First 100 chars: {content[:100]}... "
        f"Debug saved to {debug_path}"
    )

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
            "country_affected": f.get("country_affected")
        })

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

async def run_analysis(
    report_id: str,
    file_path: str,
    company_name: str | None,
    report_year: int | None,
    provider_name: str | None,
    api_key: str | None
):
    """
    Background task: parse PDF → build prompt → call LLM → store results.
    """
    analysis_id = str(uuid.uuid4())

    try:
        # Step 1: Update status to processing
        reports_store[report_id]["status"] = "processing"

        # Step 2: Parse the PDF
        temp_path = file_path + ".processing.pdf"
        shutil.copy2(file_path, temp_path)

        try:
            parsed_doc = parse_pdf(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        reports_store[report_id]["page_count"] = parsed_doc.page_count
        reports_store[report_id]["status"] = "analyzing"

        # Step 3: Prepare text for LLM
        document_text = parsed_doc.full_text_with_markers
        estimated_tokens = len(document_text) // 4
        effective_provider = provider_name or os.getenv("LLM_PROVIDER", "gemini")

        context_limits = {
            "gemini": 900_000,
            "groq": 120_000,
            "mistral": 28_000,
            "openai": 120_000,
        }

        max_tokens = context_limits.get(effective_provider, 120_000)

        if estimated_tokens > max_tokens:
            char_limit = max_tokens * 4
            half = char_limit // 2
            document_text = (
                document_text[:half]
                + "\n\n--- [DOCUMENT TRUNCATED FOR CONTEXT LIMIT — MIDDLE SECTIONS OMITTED] ---\n\n"
                + document_text[-half:]
            )

        # Step 4: Build the adversarial prompt
        messages = build_analysis_prompt(
            document_text=document_text,
            file_name=parsed_doc.file_name,
            page_count=parsed_doc.page_count
        )

        # Step 5: Call the LLM
        provider = get_provider(
            provider_name=effective_provider,
            api_key=api_key
        )

        llm_response: LLMResponse = await provider.analyze(messages)

        # Step 6: Parse LLM response
        result_data = parse_llm_json(llm_response.content)

        # Step 6.5: HARD VALIDATION — Python code enforces rules AI cannot override
        validated = validate_analysis_result(result_data, company_name)

        # Step 7: Store the VALIDATED analysis result
        analysis_result = {
            "id": analysis_id,
            "report_id": report_id,
            "company_name": validated["company_name"],
            "report_year": validated["report_year"] or report_year,
            "overall_risk_level": validated["overall_risk_level"],
            "overall_risk_score": validated["overall_risk_score"],
            "global_claim": validated["global_claim"],
            "indonesia_mentioned": validated["indonesia_mentioned"],
            "indonesia_status": validated["indonesia_status"],
            "sea_countries_mentioned": validated["sea_countries_mentioned"],
            "sea_countries_excluded": validated["sea_countries_excluded"],
            "binding_language_count": validated["binding_language_count"],
            "hedging_language_count": validated["hedging_language_count"],
            "summary": validated["summary"],
            "findings": validated["findings"],
            "document_confidence": validated["document_confidence"],
            "document_confidence_reason": validated["document_confidence_reason"],
            "scoring_breakdown": validated["scoring_breakdown"],
            "llm_provider": llm_response.provider,
            "llm_model": llm_response.model,
            "input_tokens": llm_response.input_tokens,
            "output_tokens": llm_response.output_tokens,
            "cost_estimate_usd": llm_response.cost_estimate_usd,
            "analyzed_at": datetime.utcnow().isoformat()
        }

        analysis_store[analysis_id] = analysis_result

        # Save to disk as well
        result_path = RESULTS_DIR / f"{analysis_id}.json"
        with open(result_path, "w") as f:
            json.dump(analysis_result, f, indent=2)

        # Update report status
        reports_store[report_id]["status"] = "completed"
        reports_store[report_id]["analysis_id"] = analysis_id
        reports_store[report_id]["company_name"] = analysis_result["company_name"]

    except Exception as e:
        reports_store[report_id]["status"] = "failed"
        reports_store[report_id]["error"] = str(e)
        print(f"Analysis failed for report {report_id}: {e}")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    """Health check and API info. Supports both GET and HEAD for Render health checks."""
    body = {
        "name": "Corporate Accountability Engine (CAE)",
        "version": "2.0.0",
        "organization": "Sinergia Animal International / AFFA",
        "status": "operational",
        "evasion_patterns": 9,
        "features": ["9 evasion patterns", "AI confidence check", "document verification"],
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /upload",
            "analyze": "POST /analyze",
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
    body = {"status": "ok", "version": "2.1.0", "timestamp": datetime.utcnow().isoformat()}

    if request.method == "HEAD":
        return Response(status_code=200)

    return body

# --- UPLOAD ---

@app.post("/upload", response_model=UploadResponse)
async def upload_report(file: UploadFile = File(...)):
    """
    Upload a PDF sustainability report for analysis.

    Accepts: PDF files up to 50MB
    Returns: report_id to use for triggering analysis
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Validate file size (50MB limit)
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum 50MB.")

    # Generate unique ID and save file
    report_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{report_id}.pdf"

    with open(file_path, "wb") as f:
        f.write(contents)

    # Store report metadata
    reports_store[report_id] = {
        "id": report_id,
        "file_name": file.filename,
        "file_path": str(file_path),
        "file_size": len(contents),
        "company_name": None,
        "report_year": None,
        "status": "uploaded",
        "page_count": None,
        "analysis_id": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat()
    }

    return UploadResponse(
        report_id=report_id,
        file_name=file.filename,
        status="uploaded",
        message="PDF uploaded successfully. Send POST /analyze to start analysis."
    )

# --- ANALYZE ---

@app.post("/analyze")
async def analyze_report(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Trigger AI analysis on an uploaded report.

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

    # Launch background analysis
    background_tasks.add_task(
        run_analysis,
        report_id=report_id,
        file_path=report["file_path"],
        company_name=request.company_name,
        report_year=request.report_year,
        provider_name=request.provider,
        api_key=request.api_key
    )

    return {
        "report_id": report_id,
        "status": "processing",
        "message": "Analysis started. Poll GET /reports/{report_id} for status updates.",
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
    if analysis_id not in analysis_store:
        # Try loading from disk
        result_path = RESULTS_DIR / f"{analysis_id}.json"
        if result_path.exists():
            with open(result_path) as f:
                analysis_store[analysis_id] = json.load(f)
        else:
            raise HTTPException(status_code=404, detail="Analysis not found.")

    return analysis_store[analysis_id]

# --- EXPORT ---

@app.get("/analysis/{analysis_id}/export")
async def export_analysis_csv(analysis_id: str):
    """Export findings as CSV-formatted data."""
    if analysis_id not in analysis_store:
        result_path = RESULTS_DIR / f"{analysis_id}.json"
        if result_path.exists():
            with open(result_path) as f:
                analysis_store[analysis_id] = json.load(f)
        else:
            raise HTTPException(status_code=404, detail="Analysis not found.")

    analysis = analysis_store[analysis_id]
    findings = analysis.get("findings", [])

    headers = [
        "finding_type", "severity", "title", "description",
        "exact_quote", "page_number", "section", "country_affected"
    ]

    csv_lines = [",".join(headers)]
    for f in findings:
        row = []
        for h in headers:
            value = str(f.get(h, "")).replace('"', '""')
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
                "name": "Google Gemini",
                "model": "gemini-2.5-flash",
                "free": True,
                "context_window": "1M tokens",
                "best_for": "Large documents (200+ pages), single-pass analysis",
                "get_key": "https://aistudio.google.com/apikey"
            },
            "groq": {
                "name": "Groq (Llama 3.3 70B)",
                "model": "llama-3.3-70b-versatile",
                "free": True,
                "context_window": "131k tokens",
                "best_for": "Fast inference, medium documents",
                "get_key": "https://console.groq.com/keys"
            },
            "mistral": {
                "name": "Mistral",
                "model": "mistral-small-latest",
                "free": True,
                "context_window": "32k tokens",
                "best_for": "Backup provider, multilingual",
                "get_key": "https://console.mistral.ai/api-keys"
            },
            "openai": {
                "name": "OpenAI GPT-4o",
                "model": "gpt-4o",
                "free": False,
                "context_window": "128k tokens",
                "cost": "$0.15-$0.30 per 200-page report",
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
                "error": str(e),
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
