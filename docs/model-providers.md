# Gemini + Qwen migration (31 October 2026)

CAE now accepts exactly two provider IDs: `gemini` and `qwen`. The Gemini
Flash-to-Pro pipeline remains unchanged. Qwen defaults to `qwen3.8-max`, chosen
for long-document evidence analysis and structured JSON support. This selection
is based on vendor capabilities, not a measured CAE accuracy benchmark.

## Configuration and rollout

1. Deploy this backend revision before the frontend's Qwen selector change.
2. Set `LLM_PROVIDER=gemini` or `qwen`. Remove any old `OPENAI_API_KEY` and
   `OPENAI_MODEL` deployment settings. Explicit unsupported requests return 400;
   an unsupported server default returns 503 before a job is reserved.
3. For Qwen set `DASHSCOPE_API_KEY` and preferably `DASHSCOPE_WORKSPACE_ID` for
   the workspace-specific Singapore URL. Alternatively set `QWEN_BASE_URL` to
   your region's HTTPS Chat Completions base URL (ending in `/compatible-mode/v1`).
   The API key must come from that same region. Never put the key in the URL.
4. With neither workspace nor override set, the existing DashScope International
   endpoint remains the compatibility default. Migrate to the workspace endpoint
   recommended in the current vendor documentation. No workspace ID is invented.
5. The frontend supports Gemini and Qwen only. Unsupported saved selections fall
   back to Gemini without forwarding the previous provider's key. Changing the
   provider in Settings clears the typed key. Save the chosen provider and its key.
6. Use the connection test with your account's enabled model, then run a small
   labeled report in staging. Live API access, quota and accuracy require account
   credentials; automated tests use mocked HTTP responses.

Qwen requests use JSON object mode and the model's default thinking behavior.
The previous nested `extra_body` field was an SDK option mistakenly sent as an
HTTP JSON field; it has been removed. Only the final `content` is parsed, not
`reasoning_content`. Truncated/empty responses fail rather than becoming results.
The output budget is 32,768 tokens. Input uses a conservative 900,000 UTF-8 byte
budget for the default model (220,000 for custom model overrides); this is not a
tokenizer or a guarantee for every alternate model. Model overrides need staging
validation. Costs are marked unavailable rather than estimated as free.

Historical analysis results and the 28 October PDF are retained unchanged.
No database schema migration is required for these provider IDs. Code publication
does not itself verify Cloud Run deployment or successful paid API calls.

## Official references

- https://www.alibabacloud.com/help/en/model-studio/qwen3-8-max
- https://help.aliyun.com/en/model-studio/qwen-structured-output
- https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope
