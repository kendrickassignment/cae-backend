# 🔍 Corporate Accountability Engine (CAE) — Backend

**Adversarial AI auditor for corporate sustainability reports.**
Built for **Sinergia Animal International / Act For Farmed Animals (AFFA).**

Detects greenwashing in cage-free egg commitments with a focus on Indonesia.

> Frontend: [cae-animals.com](https://cae-animals.com)

---

## 💰 Cost

| Component | Service | Cost | Notes |
| :--- | :--- | :--- | :--- |
| **AI (Default)** | Google Gemini API | Free | 15 req/min, 1M token context |
| **AI (Backup)** | Groq API | Free | 30 req/min, 131K context |
| **AI (Backup 2)** | Mistral API | Free | 32K context |
| **AI (Premium)** | OpenAI API (GPT-4o) | ~$25/month | 128K context, highest quality |
| **Hosting** | Render.com | Free | Free tier |
| **PDF Parsing** | PyMuPDF | Free | Open-source |
| **Frontend** | Lovable | $25/month | React hosting + deployment |
| **Database** | Supabase | Free | PostgreSQL + Auth + Storage + Realtime |
| **Email** | Resend | Free | 100 emails/day |
| **Domain** | cae-animals.com | $1/year | Custom domain |
| **Total (free AI)** | | **~$26/month** | Using Gemini/Groq/Mistral |
| **Total (with OpenAI)** | | **~$51/month** | Using GPT-4o |

---

## 🚀 Quick Start (Local)

### 1. Clone and install
```bash
git clone https://github.com/kendrickassignment/cae-backend.git
cd cae-backend
pip install -r requirements.txt
```

### 2. Get your FREE API key
Go to [Google AI Studio](https://aistudio.google.com/apikey) and create a Gemini API key.

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env — see Environment Variables section below
```

### 4. Run the server
```bash
python main.py
```

- Server starts at `http://localhost:8000`
- API docs at `http://localhost:8000/docs` (interactive Swagger UI)

### 5. Test it

```bash
# Upload a PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@sustainability_report.pdf"

# Start analysis (use the report_id from upload response)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"report_id": "YOUR_REPORT_ID"}'

# Check status
curl http://localhost:8000/reports/YOUR_REPORT_ID

# Get results (use analysis_id from status response)
curl http://localhost:8000/analysis/YOUR_ANALYSIS_ID
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | API info and health check |
| `GET` | `/health` | Simple health check |
| `POST` | `/upload` | Upload a PDF report (max 50MB) |
| `POST` | `/analyze` | Trigger AI analysis on uploaded report |
| `GET` | `/reports` | List all reports |
| `GET` | `/reports/{id}` | Get report status |
| `GET` | `/analysis/{id}` | Get full analysis results |
| `GET` | `/analysis/{id}/export` | Export findings as CSV |
| `GET` | `/providers` | List available AI providers |
| `POST` | `/providers/test` | Test an API key |

---

## 🧠 How the Analysis Works

1. **Upload:** PDF is saved and a `report_id` is returned.
2. **Duplicate Check:** SHA-256 file hash compared against existing analyses. Company + year combination also checked. User can proceed or view existing.
3. **Parse:** PyMuPDF extracts all text with page numbers. Footnotes, tables, and appendices are specially tagged.
4. **Prompt:** The adversarial system prompt tells the AI to act as a strict compliance officer hunting for **9 greenwashing evasion patterns**.
5. **Analyze:** The LLM processes the full document and returns structured JSON with findings, risk scores, and exact page citations.
6. **Deduplicate:** Two-layer system removes noisy/duplicate findings:
   - **AI Prompt Layer** — instructs LLM to group related findings (max 3-5 per pattern, target 7-15 total)
   - **Python Safety Net** — `deduplicate_findings()` merges identical findings server-side, collecting all page references into one grouped finding
7. **Score:** Deterministic, tamper-proof scoring — AI suggests a score, Python enforces `score_to_level()` mapping. AI cannot override the risk level.
8. **Store:** Results saved to Supabase PostgreSQL with Row Level Security. Admin notifications triggered via edge functions.

### The 9 Evasion Patterns Detected

| # | Pattern | Description | Severity |
|---|---------|-------------|----------|
| 1 | **Hedging Language** | "We aim to", "where feasible" — soft language avoiding binding commitments | 🟡 Medium |
| 2 | **Strategic Silence** | Indonesia not mentioned at all in cage-free context | 🔴 Critical |
| 3 | **Geographic Tiering** | "Leading markets" get real commitments; everywhere else gets vague deadlines | 🔴 Critical |
| 4 | **Franchise Firewall** | "Company-operated stores only" — franchises excluded | 🟠 High |
| 5 | **Availability Clause** | "Where supply is readily available" — indefinite escape hatch | 🟡 Medium |
| 6 | **Timeline Deferral** | Pushing deadlines indefinitely ("by 2030... by 2035... by 2040...") | 🟠 High |
| 7 | **Silent Delisting** | Quietly removing previously included countries from cage-free programs | 🔴 Critical |
| 8 | **Corporate Ghosting** | No response to inquiries, no progress updates, no accountability | 🟠 High |
| 9 | **Commitment Downgrade** | Weakening language from previous years — absolute → relative, specific → vague | 🟠 High |

### Scoring Algorithm

```python
def score_to_level(score: int) -> str:
    """Deterministic risk mapping — bypasses AI hallucination"""
    if score >= 80: return "critical"
    if score >= 56: return "high"
    if score >= 31: return "medium"
    return "low"
```

| Factor | Points | Condition |
|---|---|---|
| **Strategic Silence** | +35 | Indonesia not mentioned in cage-free context |
| **Geographic Exclusion** | +30 | Indonesia explicitly excluded |
| **Franchise Firewall** | +15 | Commitments limited to company-owned |
| **Corporate Ghosting** | +15 | No external accountability mechanism |
| **Commitment Downgrade** | +15 | Weakened language from previous years |
| **Timeline Deferral** | +10 | Indonesia deadlines pushed beyond 2030 |
| **Hedging Language** | +2 each | Non-binding phrases (max +10) |
| **Availability Clause** | +5 each | Escape conditions (max +10) |
| **Binding Language** | -3 each | Strong commitments (max -15) |
| **Third-Party Audit** | -5 | Independent verification exists |
| **Indonesia Data** | -10 | Indonesia-specific progress reported |

---

## ☁️ Deploy to Render (Free)

1. Push this code to a **GitHub repo**.
2. Go to [Render.com](https://render.com) and connect your GitHub.
3. Create a **New Web Service** → select your repo.
4. Render auto-detects the `render.yaml` config.
5. Add your environment variables in the Render dashboard → Environment tab (see below).
6. Deploy! Your API will be live at `https://cae-backend.onrender.com`.

> **Note:** Set this URL in your Lovable frontend's Settings page as the "Backend API URL." Frontend is at [cae-animals.com](https://cae-animals.com).

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ | Google AI Studio API key (default provider) |
| `GROQ_API_KEY` | Optional | Groq API key (backup provider) |
| `MISTRAL_API_KEY` | Optional | Mistral API key (backup provider) |
| `OPENAI_API_KEY` | Optional | OpenAI API key (premium provider, ~$25/month) |
| `LLM_PROVIDER` | Optional | Default provider: `gemini`, `groq`, `mistral`, or `openai` (default: `gemini`) |
| `SUPABASE_URL` | ✅ | Your Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | ✅ | Supabase service role key (for storing results + notifications) |
| `FRONTEND_URL` | ✅ | Frontend URL for CORS (e.g., `https://cae-animals.com`) |
| `PORT` | Optional | Server port (default: `8000`) |

---

## 🔄 Switching AI Providers

Set the `LLM_PROVIDER` environment variable:
```bash
# Use Gemini (default — best for large docs, 1M token context)
LLM_PROVIDER=gemini

# Use Groq (fastest inference, 131K context)
LLM_PROVIDER=groq

# Use Mistral (backup, 32K context)
LLM_PROVIDER=mistral

# Use OpenAI (highest quality, ~$25/month)
LLM_PROVIDER=openai
```

Or override per-request in the `/analyze` endpoint:

```json
{
  "report_id": "xxx",
  "provider": "groq",
  "api_key": "your-groq-key"
}
```

### AI Provider Comparison

| Provider | Model | Context Window | Cost | Best For |
|---|---|---|---|---|
| **Google Gemini** | `gemini-2.5-flash` | 1M tokens | Free | Large documents (200+ pages) — **Recommended** |
| **Groq** | `llama-3.3-70b-versatile` | 131K tokens | Free | Fast inference, medium documents |
| **Mistral** | `mistral-small-latest` | 32K tokens | Free | Backup, multilingual analysis |
| **OpenAI** | `gpt-4o` | 128K tokens | ~$25/month | Highest quality analysis |

---

## 🔒 Security

- ✅ **CORS** restricted to `FRONTEND_URL` (cae-animals.com) — rejects unauthorized origins
- ✅ **Supabase service role** used only server-side for storing results and triggering notifications
- ✅ **Tamper-proof scoring** — Python enforces `score_to_level()`, AI cannot override risk levels
- ✅ **SHA-256 file hashing** for duplicate PDF detection
- ✅ **Input validation** on all endpoints
- ✅ **No API keys stored server-side** — users provide their own keys per-request or via settings

---

## 📁 File Structure

```text
cae-backend/
├── main.py              # FastAPI app — all endpoints
├── pdf_parser.py        # Forensic PDF parser (PyMuPDF)
├── llm_providers.py     # Multi-provider LLM abstraction
├── system_prompt.py     # Adversarial AI prompt (the brain)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── Dockerfile           # Container for deployment
├── render.yaml          # Render.com free deployment config
├── README.md            # This file
├── uploads/             # Uploaded PDFs (created at runtime)
└── results/             # Analysis JSON results (created at runtime)
```

---

## 🛡️ Built by Kendrick with ❤️

> Frontend: [cae-animals.com](https://cae-animals.com) | Backend: [cae-backend.onrender.com](https://cae-backend.onrender.com)

This tool empowers advocacy campaigns with instant, citation-backed evidence to hold multinational corporations accountable for their cage-free egg commitments in Indonesia.

**2 weeks → 3-5 minutes.** That's the power of adversarial AI.
