from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import numpy as np
import json
import io
import os
from pathlib import Path
from analyzer import FinancialAnalyzer
from narrator import FinanceNarrator

# Base directory = parent of backend/
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR   = FRONTEND_DIR / "static"
SAMPLE_DIR   = BASE_DIR / "sample_data"

# Ensure static dir exists
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FinanceNarrate AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

analyzer = FinancialAnalyzer()
narrator  = FinanceNarrator()


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.post("/api/analyze")
async def analyze_financial_data(file: UploadFile = File(...)):
    """Upload CSV/Excel financial data and get AI-powered analysis."""
    try:
        content = await file.read()
        
        # Parse file
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Only CSV or Excel files supported.")
        
        # Analyze
        analysis = analyzer.analyze(df)
        
        # Generate narrative via LLM
        narrative = await narrator.generate_narrative(analysis)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "analysis": analysis,
            "narrative": narrative
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-json")
async def analyze_json_data(payload: dict):
    """Accept JSON data directly for analysis."""
    try:
        df = pd.DataFrame(payload.get("data", []))
        if df.empty:
            raise HTTPException(status_code=400, detail="No data provided.")
        
        analysis = analyzer.analyze(df)
        narrative = await narrator.generate_narrative(analysis)
        
        return JSONResponse(content={
            "success": True,
            "rows": len(df),
            "columns": list(df.columns),
            "analysis": analysis,
            "narrative": narrative
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sample")
async def get_sample_analysis():
    """Return a sample analysis using built-in demo data."""
    try:
        df = pd.read_csv(str(SAMPLE_DIR / "sample_financials.csv"))
        analysis = analyzer.analyze(df)
        narrative = await narrator.generate_narrative(analysis)
        return JSONResponse(content={
            "success": True,
            "filename": "sample_financials.csv",
            "rows": len(df),
            "columns": list(df.columns),
            "analysis": analysis,
            "narrative": narrative
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "FinanceNarrate AI"}
