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
from llm_providers import get_provider, LLMResponse, PROVIDERS, LLM_CONCURRENCY

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
        "Detects greenwashing in cage-free egg commitments with focus on Indonesia. "
        "9 evasion patterns + AI document confidence check. "
        "Hybrid Gemini routing: 3 Flash → 3.1 Pro escalation. "
        "Built for Sinergia Animal International / AFFA."
    ),
    version="2.2.0",
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

class AnalyzeRequest(BaseModel):
    report_id: str
    company_name: str | None = None
    report_year: int | None = None
    provider: str | None = None # gemini, qwen, openai
    api_key: str | None = None # Optional override

class AnalyzeMultiRequest(BaseModel):
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


def deduplicate_findings(findings: list[dict]) -> list[dict]:
    """
    Merge findings with identical finding_type + similar titles.
    Safety net — the AI prompt should already group them,
    but if it doesn't, Python enforces it.
    """
    from collections import defaultdict

    # Group by finding_type
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in findings:
        groups[f["finding_type"]].append(f)

    merged = []
    for ftype, group_findings in groups.items():
        if len(group_findings) <= 3:
            # 3 or fewer findings of this type — keep all
            merged.extend(group_findings)
            continue

        # Check if titles are identical/very similar
        titles = [f["title"].lower().strip() for f in group_findings]
        unique_titles = set(titles)

        if len(unique_titles) <= 2:
            # All findings have same/similar titles — MERGE into one
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
            sorted_findings = sorted(
                group_findings,
                key=lambda f: severity_order.get(f["severity"], 5)
            )

            main_finding = sorted_findings[0].copy()

            # Collect all page numbers and quotes
            all_pages = []
            all_quotes = []
            for f in group_findings:
                if f.get("page_number") and f["page_number"] > 0:
                    all_pages.append(f["page_number"])
                if f.get("exact_quote") and f["exact_quote"] != "N/A":
                    all_quotes.append(f["exact_quote"])

            all_pages = sorted(set(all_pages))

            # Build page reference string
            page_list = ", ".join(f"p.{p}" for p in all_pages[:10])
            if len(all_pages) > 10:
                page_list += f" (+{len(all_pages) - 10} more)"

            instance_count = len(group_findings)
            main_finding["title"] = f"{main_finding['title']} ({instance_count} instances)"
            main_finding["description"] = (
                f"{main_finding.get('description', '')} "
                f"[Detected {instance_count} instances across the report. "
                f"Found on: {page_list}. "
                f"Most significant occurrence on p.{all_pages[0] if all_pages else 'N/A'}.]"
            ).strip()

            # Keep up to 3 most distinct quotes
            unique_quotes = list(dict.fromkeys(all_quotes))[:3]
            if unique_quotes:
                main_finding["exact_quote"] = " | ".join(unique_quotes)

            merged.append(main_finding)

            print(
                f"\U0001f504 MERGED {instance_count} '{ftype}' findings into 1 "
                f"(pages: {page_list})"
            )
        else:
            # Different titles — keep up to 5 most severe
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
            sorted_findings = sorted(
                group_findings,
                key=lambda f: severity_order.get(f["severity"], 5)
            )
            kept = sorted_findings[:5]
            merged.extend(kept)

            if len(group_findings) > 5:
                print(
                    f"\u2702\ufe0f TRIMMED '{ftype}' findings from {len(group_findings)} to 5 "
                    f"(kept highest severity)"
                )

    # Re-assign IDs
    for i, f in enumerate(merged):
        f["id"] = f"f-{i + 1}"

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
    # Bungkus seluruh logika di dalam Semaphore agar antrean terjaga
    async with LLM_CONCURRENCY:
        analysis_id = str(uuid.uuid4())

        try:
            # Step 1: Update status to processing
            reports_store[report_id]["status"] = "processing"

            # Step 2: Parse the PDF
            # (Pastikan semua baris ini ada di dalam blok 'async with')
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
                "qwen": 240_000,
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

            # ====================================================================
            # STEP 5 & 6: CALL LLM & PARSE (HYBRID ROUTING FOR GEMINI)
            # ====================================================================
            final_model = ""
            final_input_tokens = 0
            final_output_tokens = 0
            final_cost = 0.0

            if effective_provider == "gemini":
                # --- TAHAP 1: FAST SCAN DENGAN GEMINI 3.0 FLASH ---
                print(f"⚡ [TAHAP 1] Memulai Fast Scan dengan Gemini 3 Flash Preview...")
                from llm_providers import GeminiProvider
            
                flash_provider = GeminiProvider(api_key=api_key, model_name="gemini-3-flash-preview")
                flash_response = await flash_provider.analyze(messages)
                result_data = parse_llm_json(flash_response.content)
            
                raw_score = int(result_data.get("overall_risk_score", 0))
                indo_mentioned = result_data.get("indonesia_mentioned")
                indo_status = str(result_data.get("indonesia_status", "")).lower()

                # --- TAHAP 2: EVALUASI UNTUK ESCALATE KE PRO ---
                needs_pro = False
                routing_reason = ""

                if raw_score >= 56:
                    needs_pro = True
                    routing_reason = f"Skor risiko tinggi ({raw_score} - High/Critical)"
                elif indo_mentioned is False or indo_status == "silent":
                    needs_pro = True
                    routing_reason = "Terdeteksi Strategic Silence (Indonesia tidak dibahas)"

                if needs_pro:
                    print(f"🚩 [TAHAP 2] Trigger Pro aktif: {routing_reason}. Merutekan ke Gemini 3.1 Pro Preview...")
                    try:
                        pro_provider = GeminiProvider(api_key=api_key, model_name="gemini-3.1-pro-preview")
                        pro_response = await pro_provider.analyze(messages)
                    
                        result_data = parse_llm_json(pro_response.content)
                        final_model = "gemini-3.1-pro-preview (Hybrid Escalate)"
                        final_input_tokens = flash_response.input_tokens + pro_response.input_tokens
                        final_output_tokens = flash_response.output_tokens + pro_response.output_tokens
                
                    except Exception as pro_error:
                        print(f"⚠️ [TAHAP 2 GAGAL] Gemini Pro bermasalah ({pro_error}). Menggunakan fallback hasil Flash.")
                    
                        # Kembalikan ke hasil Flash (Tahap 1)
                        result_data = parse_llm_json(flash_response.content)
                        catatan = " [CATATAN SISTEM: Eskalasi deep analysis gagal karena server LLM sibuk. Ini adalah hasil Fast Scan.]"
                        result_data["document_confidence_reason"] = str(result_data.get("document_confidence_reason", "")) + catatan
                    
                        final_model = "gemini-3-flash-preview (Pro Fallback)"
                        final_input_tokens = flash_response.input_tokens
                        final_output_tokens = flash_response.output_tokens
                else:
                    print("✅ Laporan tampak aman. Menyelesaikan dengan hasil Flash saja.")
                    final_model = "gemini-3-flash-preview"
                    final_input_tokens = flash_response.input_tokens
                    final_output_tokens = flash_response.output_tokens
            
                final_cost = 0.0 # Free/Kredit

            else:
                # --- JIKA MENGGUNAKAN PROVIDER LAIN (Qwen, OpenAI, dll) ---
                provider = get_provider(provider_name=effective_provider, api_key=api_key)
                llm_response = await provider.analyze(messages)
                result_data = parse_llm_json(llm_response.content)
            
                final_model = llm_response.model
                final_input_tokens = llm_response.input_tokens
                final_output_tokens = llm_response.output_tokens
                final_cost = llm_response.cost_estimate_usd


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
                "llm_provider": effective_provider,
                "llm_model": final_model,
                "input_tokens": final_input_tokens,
                "output_tokens": final_output_tokens,
                "cost_estimate_usd": final_cost,
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
async def run_multi_analysis(
    primary_report_id: str,
    report_ids: list[str],
    company_name: str | None,
    report_year: int | None,
    provider_name: str | None,
    api_key: str | None
):
    """
    Background task: parse MULTIPLE PDFs → merge text → hybrid analysis → ONE result.
    """
    # 1. Masuk ke antrean (Semaphore)
    async with LLM_CONCURRENCY:
        analysis_id = str(uuid.uuid4())

        try:
            # 2. Update status awal
            reports_store[primary_report_id]["status"] = "processing"

            # Parse all PDFs and collect text
            all_texts = []
            total_pages = 0
            file_names = []

            for rid in report_ids:
                report = reports_store[rid]
                file_path = report["file_path"]

                temp_path = file_path + ".processing.pdf"
                shutil.copy2(file_path, temp_path)

                try:
                    parsed_doc = parse_pdf(temp_path)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                page_offset = total_pages
                total_pages += parsed_doc.page_count
                file_names.append(parsed_doc.file_name)

                # Add document separator with file info
                doc_header = (
                    f"\n\n{'=' * 60}\n"
                    f"=== [DOCUMENT {len(all_texts) + 1}: {parsed_doc.file_name} "
                    f"(pages {page_offset + 1}-{page_offset + parsed_doc.page_count})] ===\n"
                    f"{'=' * 60}\n\n"
                )
                all_texts.append(doc_header + parsed_doc.full_text_with_markers)

                # Update individual report metadata
                reports_store[rid]["page_count"] = parsed_doc.page_count
                reports_store[rid]["status"] = "analyzing"

            # Merge all text
            merged_text = "\n".join(all_texts)
            merged_file_name = f"MERGED: {', '.join(file_names)}"

            print(
                f"  📎 Merged {len(report_ids)} documents: "
                f"{total_pages} total pages, "
                f"{len(merged_text)} chars"
            )

            reports_store[primary_report_id]["status"] = "analyzing"

            # Prepare text with context window limits
            estimated_tokens = len(merged_text) // 4
            effective_provider = provider_name or os.getenv("LLM_PROVIDER", "gemini")

            context_limits = {
                "gemini": 900_000,
                "qwen": 240_000,
                "openai": 120_000,
            }

            max_tokens = context_limits.get(effective_provider, 120_000)

            if estimated_tokens > max_tokens:
                char_limit = max_tokens * 4
                half = char_limit // 2
                merged_text = (
                    merged_text[:half]
                    + "\n\n--- [DOCUMENT TRUNCATED FOR CONTEXT LIMIT — MIDDLE SECTIONS OMITTED] ---\n\n"
                    + merged_text[-half:]
                )
                print(f"  📄 Merged text truncated: {estimated_tokens} tokens → ~{max_tokens} tokens")

            # Build the adversarial prompt
            messages = build_analysis_prompt(
                document_text=merged_text,
                file_name=merged_file_name,
                page_count=total_pages
            )

            # ====================================================================
            # HYBRID ROUTING (same logic as run_analysis)
            # ====================================================================
            final_model = ""
            final_input_tokens = 0
            final_output_tokens = 0
            final_cost = 0.0

            if effective_provider == "gemini":
                print(f"  ⚡ [TAHAP 1] Fast Scan merged docs dengan Gemini 3 Flash Preview...")
                from llm_providers import GeminiProvider

                flash_provider = GeminiProvider(api_key=api_key, model_name="gemini-3-flash-preview")
                flash_response = await flash_provider.analyze(messages)
                result_data = parse_llm_json(flash_response.content)

                raw_score = int(result_data.get("overall_risk_score", 0))
                indo_mentioned = result_data.get("indonesia_mentioned")
                indo_status = str(result_data.get("indonesia_status", "")).lower()

                needs_pro = False
                routing_reason = ""

                if raw_score >= 56:
                    needs_pro = True
                    routing_reason = f"Skor risiko tinggi ({raw_score} - High/Critical)"
                elif indo_mentioned is False or indo_status == "silent":
                    needs_pro = True
                    routing_reason = "Terdeteksi Strategic Silence (Indonesia tidak dibahas)"

                if needs_pro:
                    print(f"🚩 [TAHAP 2] Trigger Pro aktif: {routing_reason}. Merutekan ke Gemini 3.1 Pro Preview...")
                    try:
                        pro_provider = GeminiProvider(api_key=api_key, model_name="gemini-3.1-pro-preview")
                        pro_response = await pro_provider.analyze(messages)

                        result_data = parse_llm_json(pro_response.content)
                        final_model = "gemini-3.1-pro-preview (Hybrid Escalate)"
                        final_input_tokens = flash_response.input_tokens + pro_response.input_tokens
                        final_output_tokens = flash_response.output_tokens + pro_response.output_tokens

                    except Exception as pro_error:
                        print(f"⚠️ [TAHAP 2 GAGAL] Gemini Pro bermasalah ({pro_error}). Menggunakan fallback hasil Flash.")

                        # Kembalikan ke hasil Flash (Tahap 1)
                        result_data = parse_llm_json(flash_response.content)
                        catatan = " [CATATAN SISTEM: Eskalasi deep analysis gagal karena server LLM sibuk. Ini adalah hasil Fast Scan.]"
                        result_data["document_confidence_reason"] = str(result_data.get("document_confidence_reason", "")) + catatan

                        final_model = "gemini-3-flash-preview (Pro Fallback)"
                        final_input_tokens = flash_response.input_tokens
                        final_output_tokens = flash_response.output_tokens
                else:
                    print("✅ Laporan tampak aman. Menyelesaikan dengan hasil Flash saja.")
                    final_model = "gemini-3-flash-preview"
                    final_input_tokens = flash_response.input_tokens
                    final_output_tokens = flash_response.output_tokens

                final_cost = 0.0

            else:
                provider = get_provider(provider_name=effective_provider, api_key=api_key)
                llm_response = await provider.analyze(messages)
                result_data = parse_llm_json(llm_response.content)

                final_model = llm_response.model
                final_input_tokens = llm_response.input_tokens
                final_output_tokens = llm_response.output_tokens
                final_cost = llm_response.cost_estimate_usd

            # Validate
            validated = validate_analysis_result(result_data, company_name)

            # Store ONE result
            analysis_result = {
                "id": analysis_id,
                "report_id": primary_report_id,
                "merged_report_ids": report_ids,
                "merged_file_count": len(report_ids),
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
                "llm_provider": effective_provider,
                "llm_model": final_model,
                "input_tokens": final_input_tokens,
                "output_tokens": final_output_tokens,
                "cost_estimate_usd": final_cost,
                "analyzed_at": datetime.utcnow().isoformat()
            }

            analysis_store[analysis_id] = analysis_result

            result_path = RESULTS_DIR / f"{analysis_id}.json"
            with open(result_path, "w") as f:
                json.dump(analysis_result, f, indent=2)

            # Mark ALL reports as completed with same analysis_id
            for rid in report_ids:
                reports_store[rid]["status"] = "completed"
                reports_store[rid]["analysis_id"] = analysis_id
                reports_store[rid]["company_name"] = analysis_result["company_name"]

            print(
                f"  🎉 Multi-analysis selesai! "
                f"{len(report_ids)} files merged → 1 result. "
                f"Model: {final_model}"
            )

        except Exception as e:
            import traceback
            print(f"❌ Multi-analysis failed: {str(e)}")
            print(traceback.format_exc())
            for rid in report_ids:
                if rid in reports_store:
                    reports_store[rid]["status"] = "failed"
                    reports_store[rid]["error"] = str(e)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    """Health check and API info. Supports both GET and HEAD for Cloud Run health checks."""
    body = {
        "name": "Corporate Accountability Engine (CAE)",
        "version": "2.2.0",
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
    body = {"status": "ok", "version": "2.2.0", "timestamp": datetime.utcnow().isoformat()}

    if request.method == "HEAD":
        return Response(status_code=200)

    return body

# --- UPLOAD ---

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload dan simpan satu file PDF dengan perlindungan memori (Max 25MB)."""
    
    # Validasi 1: Format File
    if not file.filename.lower().endswith('.pdf'):
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
            
    # Kembalikan pointer pembacaan ke awal file
    await file.seek(0)
    
    # --- LOGIKA PENYIMPANAN YANG AKTIF ---
    report_id = f"rep_{uuid.uuid4().hex[:8]}"
    file_path = UPLOAD_DIR / f"{report_id}.pdf"
    
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Simpan status ke memory store
    reports_store[report_id] = {
        "id": report_id,
        "file_name": file.filename,
        "file_path": str(file_path),
        "company_name": None,
        "report_year": None,
        "status": "uploaded",
        "page_count": 0,
        "created_at": datetime.utcnow().isoformat()
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
    Max: 10 files per request.
    """
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

    primary_report_id = request.report_ids[0]

    background_tasks.add_task(
        run_multi_analysis,
        primary_report_id=primary_report_id,
        report_ids=request.report_ids,
        company_name=request.company_name,
        report_year=request.report_year,
        provider_name=request.provider,
        api_key=request.api_key
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
                "name": "Google Gemini (Hybrid)",
                "models": {
                    "fast": "gemini-3-flash-preview",
                    "pro": "gemini-3.1-pro-preview"
                },
                "routing": "Auto-escalates to 3.1 Pro Preview for high-risk reports",
                "free": True,
                "context_window": "1M tokens",
                "best_for": "Large documents (200+ pages), single-pass analysis",
                "get_key": "https://aistudio.google.com/apikey"
            },
            "qwen": {
                "name": "Qwen 3.5-397B-A17B (DashScope)",
                "model": "qwen3.5-397b-a17b",
                "free": False,
                "context_window": "256k tokens",
                "best_for": "Backup provider, large reasoning model",
                "get_key": "https://dashscope.console.aliyun.com/"
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
