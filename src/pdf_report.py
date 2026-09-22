"""
pdf_report.py
Generates a clean PDF prediction report for AttritionIQ.

Uses reportlab if available; falls back to a plain HTML/text download.
The report is a decision-support document and must NOT be used
as an automated employment decision.
"""

import os
import io
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METADATA_PATH = os.path.join(BASE_DIR, "model", "model_metadata.json")


def _get_model_name() -> str:
    try:
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        return meta.get("best_model", "Logistic Regression")
    except Exception:
        return "Logistic Regression"


def generate_pdf_report(
    prediction_result: dict,
    employee_data: dict = None,
    scenario_result: dict = None,
) -> bytes:
    """
    Generate a PDF prediction report.
    Returns bytes of the PDF (or HTML fallback).
    """
    try:
        return _generate_reportlab_pdf(prediction_result, employee_data, scenario_result)
    except ImportError:
        return _generate_html_pdf_fallback(prediction_result, employee_data, scenario_result)


def _generate_reportlab_pdf(prediction_result, employee_data, scenario_result) -> bytes:
    """Generate PDF using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm
    )

    styles = getSampleStyleSheet()
    accent = colors.HexColor("#3b82d4")
    danger = colors.HexColor("#dc2626")
    success = colors.HexColor("#16a34a")
    warning = colors.HexColor("#d97706")
    muted = colors.HexColor("#57606a")
    light_bg = colors.HexColor("#f7f8fa")
    border = colors.HexColor("#e5e7eb")

    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=20, textColor=accent, spaceAfter=4
    )
    heading_style = ParagraphStyle(
        "Heading", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#1f2328"),
        spaceBefore=14, spaceAfter=6,
        borderPad=4,
    )
    normal_style = ParagraphStyle(
        "Normal", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#1f2328"), leading=14
    )
    muted_style = ParagraphStyle(
        "Muted", parent=styles["Normal"],
        fontSize=9, textColor=muted, leading=13
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#92400e"),
        backColor=colors.HexColor("#fffbeb"),
        borderPad=6, leading=14
    )

    story = []

    # ── Header ──
    story.append(Paragraph("AttritionIQ", title_style))
    story.append(Paragraph("Employee Attrition Prediction Report", heading_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Model: {_get_model_name()}",
        muted_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=border, spaceAfter=12))

    # ── Disclaimer ──
    story.append(Paragraph(
        "⚠ IMPORTANT DISCLAIMER: This report is a decision-support tool only. "
        "It must not be used as the sole basis for employment decisions. "
        "All predictions are statistical estimates and do not establish causation. "
        "HR professionals must review and apply professional judgment before taking any action.",
        disclaimer_style
    ))
    story.append(Spacer(1, 12))

    # ── Prediction Summary ──
    story.append(Paragraph("Prediction Summary", heading_style))

    prob = prediction_result.get("probability", 0)
    prob_pct = round(prob * 100, 1)
    risk = prediction_result.get("risk_level", "Unknown")
    pred_label = prediction_result.get("prediction", "Unknown")

    risk_color = danger if risk == "High" else (warning if risk == "Medium" else success)

    summary_data = [
        ["Metric", "Value"],
        ["Attrition Prediction", pred_label],
        ["Attrition Probability", f"{prob_pct}%"],
        ["Risk Level", risk],
        ["Classification Threshold", "0.50 (default)"],
    ]
    t = Table(summary_data, colWidths=[6*cm, 10*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
        ("GRID", (0, 0), (-1, -1), 0.5, border),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # ── Explainability ──
    explanation = prediction_result.get("explanation")
    if explanation and explanation.get("factors"):
        story.append(Paragraph("Why This Prediction?", heading_style))
        story.append(Paragraph(
            "These factors are associated with the model prediction and "
            "do not establish causation.",
            muted_style
        ))
        story.append(Spacer(1, 6))

        factors = explanation["factors"]
        fdata = [["Factor", "Direction", "Contribution"]]
        for fac in factors:
            direction = fac.get("direction", "unknown")
            fdata.append([
                fac.get("feature", ""),
                "Risk-Increasing" if direction == "risk-increasing" else "Protective",
                "↑ Higher Risk" if direction == "risk-increasing" else "↓ Lower Risk",
            ])
        ft = Table(fdata, colWidths=[8*cm, 4*cm, 4*cm])
        ft.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), accent),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
            ("GRID", (0, 0), (-1, -1), 0.5, border),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(ft)
        story.append(Spacer(1, 12))

    # ── Scenario Analysis ──
    if scenario_result:
        story.append(Paragraph("What-If Scenario Analysis", heading_style))
        story.append(Paragraph(
            "This is a model simulation of hypothetical changes. "
            "It does not predict whether an employee will actually leave.",
            muted_style
        ))
        story.append(Spacer(1, 6))

        curr_prob = round(scenario_result.get("current_probability", prob) * 100, 1)
        scen_prob = round(scenario_result.get("scenario_probability", 0) * 100, 1)
        diff = round(scen_prob - curr_prob, 1)
        diff_str = f"{diff:+.1f} pp"

        sdata = [
            ["", "Value"],
            ["Current Probability", f"{curr_prob}%"],
            ["Scenario Probability", f"{scen_prob}%"],
            ["Change", diff_str],
            ["Current Risk", scenario_result.get("current_risk", risk)],
            ["Scenario Risk", scenario_result.get("scenario_risk", "Unknown")],
        ]
        st_table = Table(sdata, colWidths=[6*cm, 10*cm])
        st_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), accent),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
            ("GRID", (0, 0), (-1, -1), 0.5, border),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(st_table)
        story.append(Spacer(1, 12))

    # ── Recommendations ──
    recs = prediction_result.get("recommendations", [])
    if recs:
        story.append(Paragraph("HR Retention Recommendations", heading_style))
        story.append(Paragraph(
            "These are decision-support suggestions only. HR professionals should "
            "review before taking any action.",
            muted_style
        ))
        story.append(Spacer(1, 6))
        for i, rec in enumerate(recs, 1):
            story.append(Paragraph(f"{i}. {rec}", normal_style))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 8))

    # ── Footer disclaimer ──
    story.append(HRFlowable(width="100%", thickness=1, color=border, spaceBefore=12))
    story.append(Paragraph(
        "This report is generated by AttritionIQ, an IBM SkillsBuild educational project. "
        "It is a decision-support tool and NOT an automated employment decision system. "
        "The IBM HR dataset used is fictional. No real employees are represented.",
        disclaimer_style
    ))

    doc.build(story)
    return buf.getvalue()


def _generate_html_pdf_fallback(prediction_result, employee_data, scenario_result) -> bytes:
    """Generate an HTML report when reportlab is unavailable."""
    prob = prediction_result.get("probability", 0)
    prob_pct = round(prob * 100, 1)
    risk = prediction_result.get("risk_level", "Unknown")
    pred_label = prediction_result.get("prediction", "Unknown")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    model_name = _get_model_name()

    recs_html = "".join(
        f"<li>{r}</li>" for r in prediction_result.get("recommendations", [])
    )

    factors_html = ""
    explanation = prediction_result.get("explanation")
    if explanation and explanation.get("factors"):
        rows = "".join(
            f"<tr><td>{f['feature']}</td>"
            f"<td>{'Risk-Increasing' if f['direction']=='risk-increasing' else 'Protective'}</td></tr>"
            for f in explanation["factors"]
        )
        factors_html = f"""
        <h2>Why This Prediction?</h2>
        <p style="color:#57606a;font-size:12px;">These factors are associated with the
        model prediction and do not establish causation.</p>
        <table border="1" cellpadding="6" style="border-collapse:collapse;width:100%;font-size:12px;">
        <tr style="background:#3b82d4;color:white;"><th>Factor</th><th>Direction</th></tr>
        {rows}
        </table>"""

    scenario_html = ""
    if scenario_result:
        curr_p = round(scenario_result.get("current_probability", prob) * 100, 1)
        scen_p = round(scenario_result.get("scenario_probability", 0) * 100, 1)
        diff = round(scen_p - curr_p, 1)
        scenario_html = f"""
        <h2>What-If Scenario Analysis</h2>
        <p style="color:#57606a;font-size:12px;">Model simulation only — not a causal prediction.</p>
        <table border="1" cellpadding="6" style="border-collapse:collapse;width:100%;font-size:12px;">
        <tr><td><b>Current Probability</b></td><td>{curr_p}%</td></tr>
        <tr><td><b>Scenario Probability</b></td><td>{scen_p}%</td></tr>
        <tr><td><b>Change</b></td><td>{diff:+.1f} pp</td></tr>
        <tr><td><b>Current Risk</b></td><td>{scenario_result.get("current_risk", risk)}</td></tr>
        <tr><td><b>Scenario Risk</b></td><td>{scenario_result.get("scenario_risk","")}</td></tr>
        </table>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/>
<title>AttritionIQ Prediction Report</title>
<style>
body{{font-family:-apple-system,"Segoe UI",sans-serif;font-size:13px;color:#1f2328;max-width:800px;margin:40px auto;padding:20px;}}
h1{{color:#3b82d4;}} h2{{color:#1f2328;border-bottom:1px solid #e5e7eb;padding-bottom:4px;}}
.disclaimer{{background:#fffbeb;border:1px solid #fde68a;padding:12px;border-radius:6px;color:#92400e;font-size:12px;margin:16px 0;}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0;}}
th{{background:#3b82d4;color:white;padding:8px;text-align:left;}}
td{{padding:6px 8px;border:1px solid #e5e7eb;}}
tr:nth-child(even){{background:#f7f8fa;}}
ul{{color:#1f2328;}}
</style></head><body>
<h1>AttritionIQ</h1>
<h2>Employee Attrition Prediction Report</h2>
<p style="color:#57606a;">Generated: {now} | Model: {model_name}</p>
<div class="disclaimer">
⚠ IMPORTANT DISCLAIMER: This report is a decision-support tool only.
It must not be used as the sole basis for employment decisions.
All predictions are statistical estimates and do not establish causation.
HR professionals must review and apply professional judgment before taking any action.
</div>
<h2>Prediction Summary</h2>
<table>
<tr><td><b>Attrition Prediction</b></td><td>{pred_label}</td></tr>
<tr><td><b>Attrition Probability</b></td><td>{prob_pct}%</td></tr>
<tr><td><b>Risk Level</b></td><td>{risk}</td></tr>
<tr><td><b>Classification Threshold</b></td><td>0.50 (default)</td></tr>
</table>
{factors_html}
{scenario_html}
<h2>HR Retention Recommendations</h2>
<p style="color:#57606a;font-size:12px;">These are decision-support suggestions only.</p>
<ul>{recs_html}</ul>
<hr style="margin-top:24px;"/>
<div class="disclaimer">
This report is generated by AttritionIQ, an IBM SkillsBuild educational project.
It is a decision-support tool and NOT an automated employment decision system.
The IBM HR dataset used is fictional. No real employees are represented.
</div>
</body></html>"""

    return html.encode("utf-8")


def get_report_mimetype(prediction_result: dict, employee_data: dict = None,
                        scenario_result: dict = None):
    """Return (bytes, mimetype, filename) for the report."""
    try:
        import reportlab  # noqa: F401
        data = generate_pdf_report(prediction_result, employee_data, scenario_result)
        return data, "application/pdf", "attritioniq_report.pdf"
    except ImportError:
        data = generate_pdf_report(prediction_result, employee_data, scenario_result)
        return data, "text/html; charset=utf-8", "attritioniq_report.html"
