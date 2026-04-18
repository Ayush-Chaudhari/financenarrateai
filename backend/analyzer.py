import pandas as pd
import numpy as np
from typing import Any
import warnings
warnings.filterwarnings("ignore")


class FinancialAnalyzer:
    """
    Analyzes structured financial DataFrames and produces a rich
    analysis dict consumed by the LLM narrator and the frontend.
    """

    # Column name aliases so the analyzer works across varied CSVs
    REVENUE_ALIASES = ["revenue", "sales", "income", "gross_revenue", "total_revenue", "net_revenue"]
    EXPENSE_ALIASES = ["expense", "expenses", "cost", "costs", "opex", "cogs", "total_expenses"]
    PROFIT_ALIASES  = ["profit", "net_profit", "net_income", "ebitda", "operating_income", "margin"]
    PERIOD_ALIASES  = ["date", "month", "quarter", "period", "year", "fiscal_year", "time"]

    # ------------------------------------------------------------------ helpers

    def _find_col(self, df: pd.DataFrame, aliases: list[str]) -> str | None:
        lower = {c.lower().strip(): c for c in df.columns}
        for a in aliases:
            if a in lower:
                return lower[a]
        return None

    def _safe_pct(self, new: float, old: float) -> float | None:
        if old == 0 or pd.isna(old) or pd.isna(new):
            return None
        return round((new - old) / abs(old) * 100, 2)

    def _to_native(self, obj: Any) -> Any:
        """Recursively convert numpy types → Python native (JSON-serialisable)."""
        if isinstance(obj, dict):
            return {k: self._to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_native(i) for i in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, (np.ndarray,)):
            return self._to_native(obj.tolist())
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    # ------------------------------------------------------------------ core

    def analyze(self, df: pd.DataFrame) -> dict:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        rev_col  = self._find_col(df, self.REVENUE_ALIASES)
        exp_col  = self._find_col(df, self.EXPENSE_ALIASES)
        prof_col = self._find_col(df, self.PROFIT_ALIASES)
        per_col  = self._find_col(df, self.PERIOD_ALIASES)

        result: dict = {
            "schema":           self._schema(df),
            "revenue":          self._revenue_analysis(df, rev_col, per_col),
            "expenses":         self._expense_analysis(df, exp_col, per_col),
            "profitability":    self._profitability(df, rev_col, exp_col, prof_col),
            "anomalies":        self._detect_anomalies(df, rev_col, exp_col),
            "trends":           self._trend_summary(df, rev_col, exp_col, per_col),
            "kpis":             self._kpis(df, rev_col, exp_col, prof_col),
            "period_breakdown": self._period_breakdown(df, rev_col, exp_col, prof_col, per_col),
        }
        return self._to_native(result)

    # ------------------------------------------------------------------ sections

    def _schema(self, df: pd.DataFrame) -> dict:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        return {
            "total_rows":    len(df),
            "total_columns": len(df.columns),
            "columns":       list(df.columns),
            "numeric_cols":  numeric_cols,
            "missing_cells": int(df.isna().sum().sum()),
        }

    def _revenue_analysis(self, df, col, per_col) -> dict:
        if col is None:
            return {"available": False, "reason": "No revenue column detected"}
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        total   = float(s.sum())
        mean    = float(s.mean())
        median  = float(s.median())
        mx_idx  = int(s.idxmax())
        mn_idx  = int(s.idxmin())
        growth  = self._safe_pct(s.iloc[-1], s.iloc[0]) if len(s) > 1 else None
        mom     = [self._safe_pct(s.iloc[i], s.iloc[i-1]) for i in range(1, len(s))]

        return {
            "available":        True,
            "column":           col,
            "total":            round(total, 2),
            "mean":             round(mean, 2),
            "median":           round(median, 2),
            "max":              round(float(s.max()), 2),
            "min":              round(float(s.min()), 2),
            "max_period":       str(df.iloc[mx_idx][per_col]) if per_col else str(mx_idx),
            "min_period":       str(df.iloc[mn_idx][per_col]) if per_col else str(mn_idx),
            "overall_growth":   growth,
            "mom_growth":       [round(v, 2) if v is not None else None for v in mom],
            "volatility":       round(float(s.std()), 2),
            "values":           [round(float(v), 2) for v in s.tolist()],
        }

    def _expense_analysis(self, df, col, per_col) -> dict:
        if col is None:
            return {"available": False, "reason": "No expense column detected"}
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        growth = self._safe_pct(s.iloc[-1], s.iloc[0]) if len(s) > 1 else None
        z_scores = ((s - s.mean()) / s.std()).tolist() if s.std() > 0 else [0] * len(s)

        spikes = []
        for i, z in enumerate(z_scores):
            if abs(z) > 2:
                spikes.append({
                    "period":    str(df.iloc[i][per_col]) if per_col else str(i),
                    "value":     round(float(s.iloc[i]), 2),
                    "z_score":   round(z, 2),
                    "direction": "spike" if z > 0 else "dip",
                })

        return {
            "available":      True,
            "column":         col,
            "total":          round(float(s.sum()), 2),
            "mean":           round(float(s.mean()), 2),
            "max":            round(float(s.max()), 2),
            "min":            round(float(s.min()), 2),
            "overall_growth": growth,
            "volatility":     round(float(s.std()), 2),
            "spikes":         spikes,
            "values":         [round(float(v), 2) for v in s.tolist()],
        }

    def _profitability(self, df, rev_col, exp_col, prof_col) -> dict:
        rev  = pd.to_numeric(df[rev_col],  errors="coerce") if rev_col  else None
        exp  = pd.to_numeric(df[exp_col],  errors="coerce") if exp_col  else None
        prof = pd.to_numeric(df[prof_col], errors="coerce") if prof_col else None

        if prof is None and rev is not None and exp is not None:
            prof = rev - exp

        if prof is None:
            return {"available": False}

        margin = (prof / rev * 100).round(2) if rev is not None else None
        avg_margin = float(margin.mean()) if margin is not None else None

        return {
            "available":          True,
            "total_profit":       round(float(prof.sum()), 2),
            "mean_profit":        round(float(prof.mean()), 2),
            "best_profit":        round(float(prof.max()), 2),
            "worst_profit":       round(float(prof.min()), 2),
            "avg_margin_pct":     round(avg_margin, 2) if avg_margin is not None else None,
            "profit_values":      [round(float(v), 2) for v in prof.tolist()],
            "margin_values":      [round(float(v), 2) if not np.isnan(v) else None
                                   for v in (margin.tolist() if margin is not None else [])],
        }

    def _detect_anomalies(self, df, rev_col, exp_col) -> list:
        anomalies = []
        for col in [rev_col, exp_col]:
            if col is None:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < 3:
                continue
            mean, std = s.mean(), s.std()
            if std == 0:
                continue
            for i, v in enumerate(s):
                z = (v - mean) / std
                if abs(z) > 2.5:
                    anomalies.append({
                        "column":    col,
                        "index":     i,
                        "value":     round(float(v), 2),
                        "z_score":   round(float(z), 2),
                        "severity":  "high" if abs(z) > 3 else "medium",
                        "type":      "spike" if z > 0 else "drop",
                    })
        return anomalies

    def _trend_summary(self, df, rev_col, exp_col, per_col) -> dict:
        trends = {}
        for label, col in [("revenue", rev_col), ("expenses", exp_col)]:
            if col is None:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < 2:
                continue
            x = np.arange(len(s))
            slope, _ = np.polyfit(x, s.values, 1)
            direction = "upward" if slope > 0 else "downward"
            strength  = "strong" if abs(slope / s.mean()) > 0.05 else "moderate" if abs(slope / s.mean()) > 0.01 else "flat"
            trends[label] = {
                "direction": direction,
                "strength":  strength,
                "slope":     round(float(slope), 2),
            }
        return trends

    def _kpis(self, df, rev_col, exp_col, prof_col) -> dict:
        kpis = {}
        rev  = pd.to_numeric(df[rev_col],  errors="coerce").dropna() if rev_col  else None
        exp  = pd.to_numeric(df[exp_col],  errors="coerce").dropna() if exp_col  else None
        prof = pd.to_numeric(df[prof_col], errors="coerce").dropna() if prof_col else None

        if rev is not None:
            kpis["total_revenue"]   = round(float(rev.sum()), 2)
            kpis["avg_revenue"]     = round(float(rev.mean()), 2)
            kpis["revenue_growth"]  = self._safe_pct(rev.iloc[-1], rev.iloc[0])
        if exp is not None:
            kpis["total_expenses"]  = round(float(exp.sum()), 2)
            kpis["avg_expenses"]    = round(float(exp.mean()), 2)
        if rev is not None and exp is not None:
            net = rev.sum() - exp.sum()
            kpis["net_profit"]      = round(float(net), 2)
            kpis["expense_ratio"]   = round(float(exp.sum() / rev.sum() * 100), 2) if rev.sum() != 0 else None
            kpis["profit_margin"]   = round(float(net / rev.sum() * 100), 2) if rev.sum() != 0 else None
        return kpis

    def _period_breakdown(self, df, rev_col, exp_col, prof_col, per_col) -> list:
        if per_col is None:
            return []
        rows = []
        for _, row in df.iterrows():
            entry = {"period": str(row[per_col])}
            for label, col in [("revenue", rev_col), ("expenses", exp_col), ("profit", prof_col)]:
                if col and col in df.columns:
                    v = pd.to_numeric(row[col], errors="coerce")
                    entry[label] = round(float(v), 2) if not pd.isna(v) else None
            rows.append(entry)
        return rows
