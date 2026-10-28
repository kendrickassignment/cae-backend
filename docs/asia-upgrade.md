# CAE Asia: implementation and rollout plan

28 October 2026 (Asia/Jakarta). Backend baseline `597b970603604dcefde18bb3529519b0fa02219f`; frontend reviewed `df848269ae8948e1bbed99bc2d3eb5303c633b08`.

## Understanding the existing product

CAE audits corporate cage-free reporting using nine evasion patterns. React uploads to FastAPI, starts single/bundled analysis, polls status, then copies selected fields into Supabase. Asia positioning was already updated, but `src/services/api.ts`, `src/hooks/useRealAnalyses.ts`, `src/pages/AnalysisDetail.tsx` and Supabase types still use Indonesia/SEA fields. `fetchAndSaveAnalysis` explicitly selects fields and discards unrecognized regional metadata. Backend results were not written to Supabase despite claims in the old README.

## Plan and completion

| Workstream | Implementation | State |
| --- | --- | --- |
| Geography | 51 country/area codes, five subregions, country-selection validation and discovery | Implemented |
| Evidence | Country status, Unicode quotations, source/local-page validation, unknown/not-applicable handling | Implemented |
| Scoring | Versioned deterministic regional policy, maximum assessed-country aggregation and coverage | Implemented |
| Compatibility | Old request/response fields preserved; Indonesia score remains separate | Implemented |
| Pipeline | Shared single/multi-file path, regional escalation, distinct job/provider semaphores, active request reservation | Implemented |
| Input and results | PDF validation, no middle truncation, OCR errors, source IDs, atomic UTF-8 result files, safe errors/CSV | Implemented |
| Verification | Generated PDF fixtures, mocked providers, negative cases and concurrency; CI workflow | Implemented |
| Frontend/Supabase | Optional additive migration and integration guide | Prepared, not applied |
| Production operations | Auth, shared persistence, durable queue, live evaluation and deployment | Rollout work; not claimed complete |

## Contract and geography

Both analyze endpoints accept optional `target_countries: string[]`. Omission selects all 51 entries; values are trimmed, uppercased and deduplicated. Empty/null lists and non-catalog codes are rejected. The registry uses UN M49-inspired Asian subregions plus TW as a separate reporting market, including separate HK/MO entries. Russia is outside this registry. These are explicit product scope choices, not legal classifications or evidence of corporate operations.

New fields include `schema_version`, `analysis_scope`, `target_countries`, `country_catalog_version`, `regional_scoring_version`, `country_assessments`, `regional_risk_score`, `regional_risk_level`, `regional_aggregation`, `coverage`, `validation_warnings`, `overall_risk_scope`, `review_required`, `methodology_note`, `source_documents`, `hybrid_fallback`, `cost_estimate_available`.

Country rows contain code/name/subregion, status, mentioned, operations_in_scope, operation_evidence, evidence, sections_checked, risk_score/level, scoring_breakdown and explanation. Citations use bundle source IDs and 1-based local pages. Unknown/not-applicable rows have null scores. Conflicting duplicate country rows become unknown. Entirely missing or request-mismatched regional output fails the job.

## Scoring: asia-evidence-v1

This new screening heuristic is separate from historical Indonesia scores and requires advocacy-team calibration before policy decisions.

| Factor | Points / condition |
| --- | --- |
| Silent | +35; evidenced operations + global pledge + searched sections + text on all pages |
| Excluded / deferred | +30 with verified citation |
| Partial | +15 with verified citation |
| Compliant | 0 with citation; reported binding commitment/deadline or achievement, not independent audit |
| Silent delisting | +20; two distinct cited passages |
| Franchise firewall / commitment downgrade | +15 each; downgrade requires two passages |
| Geographic tiering/exclusion | +12 unless exclusion already counted by status |
| Availability / ghosting / hedging | +10 each, once per pattern |
| Timeline deferral | +8 unless already counted by deferred status |

Clamp 0-100. Risk bands: 0-30 low, 31-55 medium, 56-79 high, 80-100 critical. Unknown is unscored. Region = maximum scored country, always accompanied by coverage, never an average diluted by unknown countries. Model score values cannot override Python regional scoring. Legacy overall scores remain model-calculated with deterministic clamping/band mapping.

Quote verification normalizes Unicode/whitespace and checks the specific document/page. It verifies provenance, not truth or country applicability. Human review must confirm interpretation and omission claims. Text on all pages is only an extraction heuristic; image text can still be missing. No OCR or independently verified country law database is included.

## Frontend/Supabase handoff

1. Back up data; apply `migrations/20261028_asia_result_metadata.sql` in staging. It adds nullable JSON without changing old rows, RLS or grants. SQL was not executed here.
2. Add an optional regional envelope to frontend types. Save the new fields as `regional_analysis` in `fetchAndSaveAnalysis` and the retry/direct-fetch paths in `AnalysisDetail.tsx`. Preserve old Indonesia fields.
3. Extend the explicit projection of `analysis_results_safe` to include `regional_analysis`, preserving its privacy exclusions and security semantics. Do not replace it with SELECT *.
4. Regenerate Supabase types; add country selection using `/countries`; display score + coverage and country citation rows. Historical rows without the envelope mean not assessed, never regional compliance.
5. Extend exports with country, source/local page and scoring version. Keep legacy Indonesia scores labeled.

No frontend source was changed. API compatibility was tested locally; browser acceptance and database migration testing remain part of that integration release.

## Verification and release gates

Run `python -m pytest -q`, compile the six application modules and run `git diff --check`. Tests use synthetic PDFs and mocked LLMs. They prove application behavior, not provider availability, model accuracy or production load capacity.

Staging: use an account-enabled model and human-labeled reports across five subregions; include scanned/non-Latin PDFs, regional-only metrics, no-operation countries and before/after bundles. Check citation relevance and false-positive rates. Confirm model names/billing in the account. No real credentials or paid model calls were used here.

Production gate: add authenticated ownership to upload/status/result/export/list endpoints, shared object/result storage and a durable worker queue. Existing YAML permits multiple instances but local state is not shared; Cloud Run background CPU/lifecycle also needs redesign. Trusted evaluation must use one process/instance, persistent disk and a host kept alive for jobs.

Rollout: staging revision -> real-provider and frontend/DB acceptance -> internal canary -> human review -> production promotion. Cloud Run deployment, SQL application, real-provider evaluation and load tests were not performed by this code task.

Rollback: route traffic to the retained pre-upgrade revision. Keep nullable regional storage during rollback to avoid data loss. Preserve scoring/catalog versions; do not compare regional v1 scores directly with old Indonesia scores.

## Sources

- https://github.com/kendrickassignment/cae-backend/tree/597b970603604dcefde18bb3529519b0fa02219f
- https://github.com/kendrickassignment/cae-animals/tree/df848269ae8948e1bbed99bc2d3eb5303c633b08
- https://unstats.un.org/unsd/methodology/m49/ (consulted 28 October 2026)
