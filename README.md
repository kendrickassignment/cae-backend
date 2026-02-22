# 🔍 Corporate Accountability Engine (CAE) — Backend

**Adversarial AI auditor for corporate sustainability reports.**
Built for **Sinergia Animal International / Act For Farmed Animals (AFFA).**

Detects greenwashing in cage-free egg commitments with a focus on Southeast Asia / Indonesia.

---

## 💰 Cost: $0
Everything runs on free tiers:

| Component | Service | Cost |
| :--- | :--- | :--- |
| **AI (LLM)** | Google Gemini API | Free (15 req/min) |
| **AI (Backup)** | Groq API | Free (30 req/min) |
| **AI (Backup 2)** | Mistral API | Free |
| **Hosting** | Render.com | Free tier |
| **PDF Parsing** | PyMuPDF | Open-source |
| **Frontend** | Lovable | Free to build |
| **Database** | Supabase | Free tier |

---

## 🚀 Quick Start (Local)

### 1. Clone and install
```bash
git clone [https://github.com/your-repo/cae-backend.git](https://github.com/your-repo/cae-backend.git)
cd cae-backend
pip install -r requirements.txt

```

### 2. Get your FREE API key
Go to [Google AI Studio](https://aistudio.google.com/apikey) and create a Gemini API key.

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY
```

### 4. Run the server
```bash
python main.py
```

* Server starts at `http://localhost:8000`
* API docs at `http://localhost:8000/docs` (interactive Swagger UI)

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
| `POST` | `/upload` | Upload a PDF report |
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
2. **Parse:** PyMuPDF extracts all text with page numbers. Footnotes, tables, and appendices are specially tagged.
3. **Prompt:** The adversarial system prompt tells the AI to act as a strict compliance officer hunting for 5 greenwashing evasion patterns.
4. **Analyze:** The LLM processes the full document and returns structured JSON with findings, risk scores, and exact page citations.
5. **Return:** Results are stored and available via API.

### The 5 Evasion Patterns Detected:
* **Hedging Language** — "we aim to", "where feasible"
* **Strategic Silence** — Indonesia not mentioned at all
* **Geographic Tiering** — "leading markets" vs "elsewhere"
* **Franchise Firewall** — "company-operated stores only"
* **Availability Clause** — "where supply is readily available"

---

## ☁️ Deploy to Render (Free)
1. Push this code to a **GitHub repo**.
2. Go to [Render.com](https://render.com) and connect your GitHub.
3. Create a **New Web Service** → select your repo.
4. Render auto-detects the `render.yaml` config.
5. Add your `GEMINI_API_KEY` in the Render dashboard → Environment tab.
6. Deploy! Your API will be live at `https://cae-backend.onrender.com`.

> **Note:** Set this URL in your Lovable frontend's Settings page as the "Backend API URL."
---

## 🔄 Switching AI Providers

Set the `LLM_PROVIDER` environment variable:
```bash
# Use Gemini (default — best for large docs)
LLM_PROVIDER=gemini

# Use Groq (fastest inference)
LLM_PROVIDER=groq

# Use Mistral (backup)
LLM_PROVIDER=mistral

# Use OpenAI (if grant funded)
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

## 🛡️ Built by AFFA / Sinergia Animal International

This tool empowers advocacy campaigns with instant, citation-backed evidence to hold multinational corporations accountable for their cage-free egg commitments in Southeast Asia.

**2 weeks → 60 seconds.** That's the power of adversarial AI.

```
