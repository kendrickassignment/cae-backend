"""
Corporate Accountability Engine (CAE) — FastAPI Backend

Main application file. Provides REST API endpoints for:
  - PDF upload and parsing
  - AI-powered greenwashing analysis
  - Results retrieval
  - Health checks and provider testing

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
    print("  CORPORATE ACCOUNTABILITY ENGINE (CAE)")
    print("  Built for Sinergia Animal International / AFFA")
    print("=" * 60)
    print(f"  LLM Provider: {os.getenv('LLM_PROVIDER', 'gemini')} (default)")
    print(f"  Upload Dir:   {UPLOAD_DIR.absolute()}")
    print(f"  Results Dir:  {RESULTS_DIR.absolute()}")
    print("=" * 60 + "\n")
    yield
    print("\nCAE shutting down.")


app = FastAPI(
    title="Corporate Accountability Engine (CAE)",
    description=(
        "Adversarial AI auditor for corporate sustainability reports. "
        "Detects greenwashing in cage-free egg commitments with focus on Southeast Asia. "
        "Built for Sinergia Animal International / AFFA."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allow Lovable frontend to connect
# NOTE: FastAPI CORSMiddleware does NOT support wildcard subdomains like https://*.lovable.app
# You must list each specific origin explicitly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://id-preview--66fa6b17-db0f-4366-ba7e-e60ac0ceec2b.lovable.app",
        "https://cae-animals.lovable.app",
        "https://66fa6b17-db0f-4366-ba7e-e60ac0ceec2b.lovableproject.com",
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
    status: str  # uploading | uploaded | processing | analyzing | completed | failed
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
    provider: str | None = None  # gemini, groq, mistral, openai
    api_key: str | None = None   # Optional override


class ProviderTestRequest(BaseModel):
    provider: str
    api_key: str | None = None


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
        # Copy file to a temp path to avoid file handle conflicts on large docs
        temp_path = file_path + ".processing.pdf"
        shutil.copy2(file_path, temp_path)

        try:
            parsed_doc = parse_pdf(temp_path)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        reports_store[report_id]["page_count"] = parsed_doc.page_count
        reports_store[report_id]["status"] = "analyzing"

        # Step 3: Prepare text for LLM
        # Gemini has 1M context — can handle full docs.
        # For smaller context models (Groq 131k, Mistral 32k), we may need to chunk.
        document_text = parsed_doc.full_text_with_markers

        # Rough token estimate (1 token ~ 4 chars)
        estimated_tokens = len(document_text) // 4

        effective_provider = provider_name or os.getenv("LLM_PROVIDER", "gemini")

        # Context window limits (conservative, accounting for system prompt + output)
        context_limits = {
            "gemini": 900_000,    # 1M context, leave room
            "groq": 120_000,      # 131k context
            "mistral": 28_000,    # 32k context
            "openai": 120_000,    # 128k context
        }

        max_tokens = context_limits.get(effective_provider, 120_000)

        if estimated_tokens > max_tokens:
            # Truncate to fit (prioritize beginning + end where exclusions hide)
            char_limit = max_tokens * 4
            half = char_limit // 2

            # Take first half (executive summary, main claims) + last half (appendices, footnotes)
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
        try:
            result_data = json.loads(llm_response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            content = llm_response.content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result_data = json.loads(content[json_start:json_end])
            else:
                raise ValueError("LLM did not return valid JSON")

        # Step 7: Store the analysis result
        analysis_result = {
            "id": analysis_id,
            "report_id": report_id,
            "company_name": result_data.get("company_name", company_name),
            "report_year": result_data.get("report_year", report_year),
            "overall_risk_level": result_data.get("overall_risk_level"),
            "overall_risk_score": result_data.get("overall_risk_score"),
            "global_claim": result_data.get("global_claim"),
            "indonesia_mentioned": result_data.get("indonesia_mentioned"),
            "indonesia_status": result_data.get("indonesia_status"),
            "sea_countries_mentioned": result_data.get("sea_countries_mentioned", []),
            "sea_countries_excluded": result_data.get("sea_countries_excluded", []),
            "binding_language_count": result_data.get("binding_language_count", 0),
            "hedging_language_count": result_data.get("hedging_language_count", 0),
            "summary": result_data.get("summary"),
            "findings": result_data.get("findings", []),
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
        "version": "1.0.0",
        "organization": "Sinergia Animal International / AFFA",
        "status": "operational",
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
    body = {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

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
    # Remove file_path from response (internal only)
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

    # Build CSV lines
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
            "risk_level": analysis.get("overall_risk_level")
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
                "model": "gemini-2.0-flash",
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

        # Send a minimal test request
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
