# FinanceNarrate AI — Executive Financial Insight Engine

> Transform raw financial spreadsheets into board-ready executive narratives using Claude AI + Pandas.

---

## Architecture

```
financenarrateai/
├── backend/
│   ├── main.py          # FastAPI app — REST endpoints
│   ├── analyzer.py      # Pandas financial analysis engine
│   ├── narrator.py      # Claude AI narrative generator
│   └── requirements.txt
├── frontend/
│   └── index.html       # Executive-grade UI (single-file, zero dependencies)
├── sample_data/
│   └── sample_financials.csv
└── start.sh             # One-command startup
```

## Tech Stack

| Layer | Tech |
|-------|------|
| API Server | FastAPI + Uvicorn |
| Data Analysis | Pandas + NumPy |
| AI Narrative | Anthropic Claude (claude-sonnet-4) |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| File Parsing | Pandas (CSV + Excel) |

---

## Quick Start

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

> Without a key, the app runs with rule-based fallback narratives (still fully functional).

### 3. Start the server

```bash
chmod +x start.sh
./start.sh
# or manually:
cd backend && uvicorn main:app --reload --port 8000
```

### 4. Open the app

Visit `http://localhost:8000` in your browser.

---

## API Reference

### `POST /api/analyze`
Upload a CSV or Excel file for analysis.

**Request:** `multipart/form-data` with `file` field

**Response:**
```json
{
  "success": true,
  "filename": "Q4_results.csv",
  "rows": 12,
  "columns": ["Month", "Revenue", "Expenses", "Profit"],
  "analysis": {
    "kpis": { "total_revenue": 21700000, "profit_margin": 32.5, ... },
    "revenue": { "total": 21700000, "overall_growth": 124.0, ... },
    "expenses": { "spikes": [...], ... },
    "anomalies": [{ "column": "Expenses", "severity": "high", ... }],
    "period_breakdown": [{ "period": "Jan 2024", "revenue": 1250000, ... }]
  },
  "narrative": {
    "full_text": "## Executive Summary\n...",
    "sections": { "executive_summary": "..." }
  }
}
```

### `POST /api/analyze-json`
Send data as JSON directly.

```json
{ "data": [{"Month":"Jan","Revenue":100000,"Expenses":75000}] }
```

### `GET /api/sample`
Run analysis on the built-in sample dataset (no upload needed).

---

## Input File Format

The analyzer auto-detects column names using aliases. Any of these work:

| Category | Accepted Column Names |
|----------|----------------------|
| Revenue  | revenue, sales, income, gross_revenue, total_revenue |
| Expenses | expense, expenses, cost, costs, opex, cogs |
| Profit   | profit, net_profit, net_income, ebitda |
| Period   | date, month, quarter, period, year, fiscal_year |

**Sample CSV:**
```csv
Month,Revenue,Expenses,Profit
Jan 2024,1250000,890000,360000
Feb 2024,1380000,920000,460000
...
```

---

## Features

- **AI Executive Summary** — Claude drafts board-ready prose with CFO-level authority
- **Revenue Trend Analysis** — Growth rates, period-over-period sparklines, best/worst periods
- **Expense Anomaly Detection** — Z-score statistical outliers flagged by severity (2.5σ threshold)
- **KPI Dashboard** — Total revenue, expenses, profit margin, expense ratio, growth
- **Period Breakdown Table** — Full detail view with profit coloring
- **Fallback Narratives** — Rule-based summaries when API key is unavailable
- **Excel + CSV Support** — Auto-parsed via Pandas

---

## Extending the Project

### Add authentication
```python
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.post("/api/analyze")
async def analyze(file: UploadFile, token: str = Depends(security)):
    ...
```

### Add database persistence
```bash
pip install sqlalchemy alembic
```

### Add category-level expense breakdown
Add columns like `Marketing`, `R&D`, `Operations` and the analyzer will
automatically detect numeric columns for cross-category analysis.

### Deploy to cloud
```bash
# Docker
docker build -t financenarrateai .
docker run -e ANTHROPIC_API_KEY=$KEY -p 8000:8000 financenarrateai
```
