import os
import json
import httpx


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are FinanceNarrate AI — a world-class financial analyst and executive communications expert.
Your role is to transform structured financial data analysis into crisp, authoritative, board-ready narratives.

Guidelines:
- Write in executive prose: confident, precise, action-oriented
- Lead with the most critical insight
- Use specific numbers (with $ and % signs) from the data
- Highlight both risks and opportunities
- Keep language accessible to non-financial board members
- Structure as: Executive Summary → Revenue Insights → Expense Analysis → Profitability → Strategic Recommendations
- Each section: 2-4 sentences max
- Flag anomalies with urgency when severity is high
- End with 2-3 concrete next steps

Tone: Professional CFO presenting to the board of directors."""


class FinanceNarrator:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")

    async def generate_narrative(self, analysis: dict) -> dict:
        """Call Claude API to generate board-ready narrative from analysis dict."""
        
        # Build a rich prompt from the analysis
        prompt = self._build_prompt(analysis)
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        
        payload = {
            "model": MODEL,
            "max_tokens": 1500,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                narrative_text = data["content"][0]["text"]
                return self._parse_narrative(narrative_text)
        except Exception as e:
            # Fallback: generate a rule-based narrative if API fails
            return self._fallback_narrative(analysis, str(e))

    def _build_prompt(self, analysis: dict) -> str:
        kpis   = analysis.get("kpis", {})
        rev    = analysis.get("revenue", {})
        exp    = analysis.get("expenses", {})
        prof   = analysis.get("profitability", {})
        trends = analysis.get("trends", {})
        anom   = analysis.get("anomalies", [])

        lines = ["Analyze the following financial data and generate an executive board summary:\n"]

        # KPIs
        if kpis:
            lines.append("KEY PERFORMANCE INDICATORS:")
            for k, v in kpis.items():
                if v is not None:
                    unit = "%" if "margin" in k or "ratio" in k or "growth" in k else "$"
                    lines.append(f"  {k.replace('_',' ').title()}: {unit}{v:,.2f}" if unit == "$" else f"  {k.replace('_',' ').title()}: {v:.2f}{unit}")

        # Revenue
        if rev.get("available"):
            lines.append(f"\nREVENUE ANALYSIS:")
            lines.append(f"  Total Revenue: ${rev['total']:,.2f}")
            lines.append(f"  Average Revenue: ${rev['mean']:,.2f}")
            lines.append(f"  Overall Growth: {rev.get('overall_growth', 'N/A')}%")
            lines.append(f"  Revenue Volatility (std): ${rev['volatility']:,.2f}")
            lines.append(f"  Best Period: {rev.get('max_period')} (${rev['max']:,.2f})")
            lines.append(f"  Worst Period: {rev.get('min_period')} (${rev['min']:,.2f})")

        # Expenses
        if exp.get("available"):
            lines.append(f"\nEXPENSE ANALYSIS:")
            lines.append(f"  Total Expenses: ${exp['total']:,.2f}")
            lines.append(f"  Average Expenses: ${exp['mean']:,.2f}")
            lines.append(f"  Expense Growth: {exp.get('overall_growth', 'N/A')}%")
            if exp.get("spikes"):
                lines.append(f"  Anomalous Periods: {len(exp['spikes'])} spike(s) detected")
                for sp in exp["spikes"][:3]:
                    lines.append(f"    - {sp['period']}: ${sp['value']:,.2f} (z={sp['z_score']})")

        # Profitability
        if prof.get("available"):
            lines.append(f"\nPROFITABILITY:")
            lines.append(f"  Total Profit: ${prof['total_profit']:,.2f}")
            lines.append(f"  Average Profit Margin: {prof.get('avg_margin_pct', 'N/A')}%")
            lines.append(f"  Best Period Profit: ${prof['best_profit']:,.2f}")
            lines.append(f"  Worst Period Profit: ${prof['worst_profit']:,.2f}")

        # Trends
        if trends:
            lines.append(f"\nTRENDS:")
            for metric, t in trends.items():
                lines.append(f"  {metric.title()}: {t['strength']} {t['direction']} trend (slope=${t['slope']:,.2f}/period)")

        # Anomalies
        if anom:
            lines.append(f"\nANOMALIES DETECTED ({len(anom)} total):")
            for a in anom[:5]:
                lines.append(f"  [{a['severity'].upper()}] {a['column']} at index {a['index']}: ${a['value']:,.2f} (z={a['z_score']})")

        lines.append("\nGenerate the executive board summary now, structured with clear section headings.")
        return "\n".join(lines)

    def _parse_narrative(self, text: str) -> dict:
        """Return the narrative as both raw text and structured sections."""
        sections = {}
        current_section = "overview"
        buffer = []

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # Detect section headers (lines ending with : or all caps or starting with **)
            if (stripped.endswith(":") and len(stripped) < 60) or stripped.startswith("##") or stripped.startswith("**"):
                if buffer:
                    sections[current_section] = " ".join(buffer).strip()
                current_section = stripped.strip("#* :").lower().replace(" ", "_")[:40]
                buffer = []
            else:
                buffer.append(stripped)

        if buffer:
            sections[current_section] = " ".join(buffer).strip()

        return {
            "full_text": text,
            "sections":  sections,
        }

    def _fallback_narrative(self, analysis: dict, error: str) -> dict:
        """Rule-based narrative when LLM is unavailable."""
        kpis = analysis.get("kpis", {})
        rev  = analysis.get("revenue", {})
        exp  = analysis.get("expenses", {})
        anom = analysis.get("anomalies", [])
        trends = analysis.get("trends", {})

        parts = []

        # Executive Summary
        parts.append("## Executive Summary")
        if kpis.get("total_revenue") and kpis.get("total_expenses"):
            pm = kpis.get("profit_margin", 0) or 0
            health = "strong" if pm > 20 else "moderate" if pm > 10 else "under pressure"
            parts.append(f"The business generated ${kpis['total_revenue']:,.0f} in total revenue against ${kpis['total_expenses']:,.0f} in expenses, yielding a {pm:.1f}% profit margin — indicating {health} financial performance.")
        
        # Revenue
        if rev.get("available"):
            parts.append("\n## Revenue Insights")
            g = rev.get("overall_growth")
            g_str = f"grew {g:.1f}%" if g and g > 0 else f"declined {abs(g):.1f}%" if g else "remained stable"
            parts.append(f"Revenue {g_str} over the analysis period, averaging ${rev['mean']:,.0f} per period. Peak performance was recorded at {rev.get('max_period','N/A')} (${rev['max']:,.0f}).")

        # Expenses
        if exp.get("available"):
            parts.append("\n## Expense Analysis")
            spikes = exp.get("spikes", [])
            if spikes:
                parts.append(f"Expense management requires attention: {len(spikes)} anomalous period(s) detected with statistically significant deviations. Immediate review of cost drivers is recommended.")
            else:
                parts.append(f"Expenses remained within normal bounds, averaging ${exp['mean']:,.0f} per period with controlled volatility.")

        # Anomalies
        if anom:
            parts.append("\n## Risk Flags")
            high = [a for a in anom if a["severity"] == "high"]
            parts.append(f"{len(anom)} data anomalies identified ({len(high)} high-severity). These warrant investigation before the next board cycle.")

        # Recommendations
        parts.append("\n## Strategic Recommendations")
        recs = []
        if kpis.get("profit_margin", 100) < 15:
            recs.append("1. Launch a cost-optimization initiative targeting the highest-variance expense categories.")
        rev_trend = trends.get("revenue", {})
        if rev_trend.get("direction") == "downward":
            recs.append("2. Convene a revenue recovery task force; current trajectory requires intervention.")
        recs.append(f"{'2' if not recs else str(len(recs)+1)}. Implement monthly variance reporting to catch anomalies earlier in the fiscal cycle.")
        parts.extend(recs)

        if error:
            parts.append(f"\n_(Note: AI narrative engine unavailable — using analytical fallback. Error: {error[:100]})_")

        full = "\n".join(parts)
        return {"full_text": full, "sections": {"summary": full}, "fallback": True}
