# Corporate Accountability Engine (CAE) backend v3

Document-based screening of corporate cage-free egg commitments for Sinergia Animal / AFFA. The API now assesses requested countries and areas across Asia, while preserving the Indonesia and Southeast Asia fields consumed by [cae-animals](https://github.com/kendrickassignment/cae-animals).

## Upgrade

- 51 country/area codes across five Asian subregions; select `target_countries` or omit it for the whole catalog.
- Country status, document/local-page citations, deterministic regional scoring and explicit unknown/coverage counts.
- Citation text checked against the source page. Missing evidence is unknown, not automatic misconduct or zero risk.
- Shared single/multi-PDF pipeline, region-aware Gemini escalation, corrected semaphore deadlock and country-safe deduplication.
- No silent context truncation; unreadable PDFs require OCR; missing text pages disable regional silence scoring.
- Old requests and legacy response fields remain valid. Existing frontend does not yet render/save regional fields; see [integration plan](docs/asia-upgrade.md).

This is risk screening, not legal compliance certification. Quote matching verifies provenance, not truth or model interpretation. Human review remains required. Regional scoring has not been calibrated against a labeled real-report corpus.

## Run and test

Python 3.12:

```sh
python -m venv .venv
# Activate .venv using the command for your shell
python -m pip install -r requirements-dev.txt
cp env.example .env
# Set a provider API key in .env
python -m uvicorn main:app --host 127.0.0.1 --port 8000
python -m pytest -q
```

Interactive request documentation: `/docs`. Tests use generated PDFs and mocked providers, with no paid model calls or production data.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /`, `GET /health` | Version and service health (not provider readiness) |
| `GET /countries`, `GET /capabilities` | Catalog, scope and limitations |
| `POST /upload` | One PDF, maximum 25 MB |
| `POST /analyze` | Analyze one uploaded report |
| `POST /analyze-multi` | Analyze 1-3 distinct uploaded reports |
| `GET /reports`, `GET /reports/{id}` | Report list and job status |
| `GET /analysis/{id}` | Full legacy + regional result |
| `GET /analysis/{id}/export` | Findings CSV in the existing JSON envelope |
| `GET /providers`, `POST /providers/test` | Provider configuration / explicit connection test |

```json
{
  "report_id": "ID_FROM_UPLOAD",
  "company_name": "Company under review",
  "report_year": 2025,
  "provider": "gemini",
  "target_countries": ["ID", "JP", "IN", "TH", "VN"]
}
```

Multi-file uses `report_ids`. Poll the returned ID, then fetch `analysis_url` after completion. Active reports return 409; unsupported/empty country selections and duplicate report IDs are rejected.

## Result semantics

`overall_risk_score`, `overall_risk_level`, `indonesia_*`, `sea_countries_*` keep the legacy contract. Overall risk is still Indonesia-specific, model-calculated with server clamping and risk-band mapping, not a server-recomputed regional score.

`country_assessments` includes every requested country, including unknown entries. `regional_risk_score` is the maximum score among assessed countries. Always show `coverage.assessed`, `coverage.requested` and `coverage.unknown` alongside it. If none can be assessed, score is null and level is unknown.

`source_documents` maps bundle IDs such as `document-1` to original filenames/report IDs. Page numbers are local to each PDF. Quotes retain the original language; explanations are requested in English. OCR and translation are not implemented.

## Deployment boundaries

This code upgrade does not deploy Cloud Run or modify Supabase. Model defaults are retained and configurable via `env.example`; verify model availability in your target account before release.

Inherited prototype limitations: PDF/result files are local; report/job status is in memory; no backend user authentication or tenant isolation. CORS is not authentication. Use trusted/private access, one process/instance and persistent storage for evaluation. Cloud Run ephemeral storage, multiple instances and CPU throttling after background responses are unsuitable for durable jobs as currently configured. Existing YAML files are historical examples, not production scaling guarantees.

A public rollout requires authenticated ownership, shared object/result storage and durable workers. See [implementation and rollout plan](docs/asia-upgrade.md) for acceptance gates, frontend/Supabase handoff and rollback.

## Modules

- `main.py`: API, request validation and shared pipeline.
- `geography.py`: versioned country catalog.
- `asia_analysis.py`: citation validation, country scoring and coverage.
- `system_prompt.py`: legacy compatibility plus Asia evidence rules.
- `llm_providers.py`: Gemini and Qwen clients.
- `pdf_parser.py`: page-preserving extraction.
- `tests/`: deterministic unit/API/concurrency regression tests.

Geography follows [UN M49](https://unstats.un.org/unsd/methodology/m49/)-inspired regions plus Taiwan as a separate reporting market; this is not a sovereignty/legal determination.

## Two-provider configuration (31 October 2026)

Only `gemini` and `qwen` are accepted. Qwen now defaults to `qwen3.8-max`.
Gemini retains Flash -> Pro routing; these are two providers, not a two-call limit.
Remove old OpenAI environment configuration and reset saved unsupported selections.
For Qwen, set `DASHSCOPE_API_KEY` and preferably `DASHSCOPE_WORKSPACE_ID` (Singapore),
or a region-matching `QWEN_BASE_URL`. See [provider migration](docs/model-providers.md).
Historical results and the earlier PDF report remain unchanged as dated records.
