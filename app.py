# app.py
import os
import json
import subprocess
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ----- Paths / Globals -----
ROOT = Path(__file__).parent.resolve()

# Default local writes go to ./reports; in Railway we'll set DATA_DIR=/data (mounted volume)
DEFAULT_REPORTS_DIR = ROOT / "reports"
DATA_DIR = Path(os.getenv("DATA_DIR", str(DEFAULT_REPORTS_DIR))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = ROOT / "reports_manifest.json"
templates = Jinja2Templates(directory=str(ROOT / "templates"))

app = FastAPI()

# Serve generated artifacts (charts, CSVs) from the writable dir
# Your chart_generator.py should write pnl_dashboard.html into DATA_DIR
app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")

# (Optional) expose repo files for debugging or seed assets, if you need it:
# app.mount("/repo", StaticFiles(directory=str(ROOT)), name="repo")


# ---------- Utilities ----------
def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"{path.name} not found")

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"CSV not found: {path.name}")
    return pd.read_csv(path)


# ---------- Optional: integrate your chart generator ----------
def ensure_chart_generated(timeframe: str = "1M"):
    """
    Calls chart_generator.py to (re)create DATA_DIR/pnl_dashboard.html
    with an initial timeframe selection.
    """
    gen = ROOT / "chart_generator.py"  # change to "pnl_plotly.py" if that's your filename
    if not gen.exists():
        # Silently skip if generator not present; the /files HTML might already exist.
        return

    env = os.environ.copy()
    env["DATA_DIR"] = str(DATA_DIR)
    try:
        subprocess.run(
            ["python", str(gen), "--timeframe", timeframe],
            cwd=str(ROOT),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        # Non-fatal — we can still serve the last generated file if present
        print("chart generation error:", e)


def read_closed_positions(limit: int = 200):
    """Read closed_positions.csv for the 'Trade History' table on the home page."""
    path = ROOT / "closed_positions.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    preferred = ["Symbol", "Total_Trades", "Realized_PnL", "Current_Position", "Position_Status"]
    cols = [c for c in preferred if c in df.columns] or list(df.columns)
    return df[cols].head(limit).fillna("").to_dict(orient="records")


# ---------- Reports helpers ----------
def load_manifest():
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def find_report(slug: str):
    for item in load_manifest():
        if item.get("slug") == slug:
            return item
    return None


# ------------------ Routes ------------------
@app.get("/healthz")
def healthz():
    return {"ok": True, "data_dir": str(DATA_DIR)}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, tf: str = "1M"):
    # If you want the chart to always be fresh when the homepage loads,
    # uncomment the next line. Otherwise rely on the button/HTMX flow.
    # ensure_chart_generated(tf)
    closed_rows = read_closed_positions()
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "timeframe": tf, "closed_positions": closed_rows, "pnl_url": "/files/pnl_dashboard.html"},
    )

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/manifest.json")
def manifest():
    """Expose a simple manifest if you track report outputs."""
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            return JSONResponse(json.load(f))
    return JSONResponse({"reports": []})


# ---------------- Swing Reports (grid of cards + detail) ----------------
@app.get("/reports", response_class=HTMLResponse)
async def swing_reports(request: Request):
    items = load_manifest()
    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)  # newest first when available
    return templates.TemplateResponse("swing_reports.html", {"request": request, "items": items})

@app.get("/reports/{slug}", response_class=HTMLResponse)
async def report_detail(request: Request, slug: str):
    item = find_report(slug)
    if not item:
        raise HTTPException(status_code=404, detail="Report not found")

    file_path = ROOT / item["file"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report file missing")

    ext = file_path.suffix.lower()

    # Markdown → render to HTML
    if ext in {".md", ".markdown"}:
        try:
            import markdown  # ensure 'markdown' is listed in requirements.txt
        except ImportError:
            raise HTTPException(status_code=500, detail="Install 'markdown' to render .md files")
        html = markdown.markdown(file_path.read_text(encoding="utf-8"), extensions=["extra", "tables", "toc"])
        return templates.TemplateResponse(
            "report_detail.html",
            {"request": request, "item": item, "content_html": html, "embed_src": None,
             "is_csv": False, "csv_rows": [], "csv_cols": []},
        )

    # HTML/PDF → show in iframe (served via /files mount)
    if ext in {".html", ".pdf"}:
        embed_src = f"/files/{item['file']}"
        return templates.TemplateResponse(
            "report_detail.html",
            {"request": request, "item": item, "content_html": None, "embed_src": embed_src,
             "is_csv": False, "csv_rows": [], "csv_cols": []},
        )

    # CSV → preview first N rows
    if ext == ".csv":
        df = pd.read_csv(file_path)
        preview = df.head(200).fillna("").to_dict(orient="records")
        return templates.TemplateResponse(
            "report_detail.html",
            {"request": request, "item": item, "content_html": None, "embed_src": None,
             "is_csv": True, "csv_rows": preview, "csv_cols": list(df.columns)},
        )

    # Fallback: embed or allow download via iframe
    embed_src = f"/files/{item['file']}"
    return templates.TemplateResponse(
        "report_detail.html",
        {"request": request, "item": item, "content_html": None, "embed_src": embed_src,
         "is_csv": False, "csv_rows": [], "csv_cols": []},
    )


# ---------------- Legacy redirects (keep old links working) ----------------
@app.get("/positions")
async def legacy_positions_redirect():
    # Old nav label was "Positions" — send to the new Swing Reports grid
    return RedirectResponse(url="/reports", status_code=307)

@app.get("/swing_reports")
async def legacy_swing_reports_redirect():
    return RedirectResponse(url="/reports", status_code=307)

@app.get("/chart", response_class=RedirectResponse)
def chart_redirect():
    """Shortcut to the latest chart HTML."""
    pnl_path = DATA_DIR / "pnl_dashboard.html"
    if not pnl_path.exists():
        raise HTTPException(status_code=404, detail="Chart has not been generated yet.")
    return RedirectResponse(url="/files/pnl_dashboard.html")


# ---------------- HTMX partials ----------------
@app.get("/partial/chart", response_class=HTMLResponse)
async def partial_chart(request: Request, tf: str = "1M"):
    """
    Regenerates (if generator present) and returns an iframe that loads the saved HTML.
    """
    ensure_chart_generated(tf)
    html = """
    <div id="chart" class="rounded-xl border border-neutral-800 bg-neutral-950/60 p-2">
      <iframe src="/files/pnl_dashboard.html"
              style="width:100%;height:520px;border:0;border-radius:12px;"></iframe>
    </div>
    """
    return HTMLResponse(html)


# --------------- Run ---------------
# Dev:  uvicorn app:app --reload
# Prod: gunicorn -k uvicorn.workers.UvicornWorker app:app -b 0.0.0.0:$PORT
