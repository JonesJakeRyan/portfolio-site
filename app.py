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

# Writable dir for generated artifacts (Railway Volume at /data)
DEFAULT_REPORTS_DIR = ROOT / "reports"
DATA_DIR = Path(os.getenv("DATA_DIR", str(DEFAULT_REPORTS_DIR))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = ROOT / "reports_manifest.json"
templates = Jinja2Templates(directory=str(ROOT / "templates"))

app = FastAPI()

# Serve generated artifacts (charts/CSVs) from the writable dir
app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")

# Serve repo assets (PDFs/HTML that live inside the git repo)
app.mount("/assets", StaticFiles(directory=str(ROOT)), name="assets")


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


# ---------- Chart generator integration ----------
def ensure_chart_generated(timeframe: str = "YTD"):
    """
    Calls chart_generator.py (or pnl_plotly.py if you renamed) to write
    DATA_DIR/pnl_dashboard.html. Prints helpful logs on Railway.
    """
    gen = ROOT / "chart_generator.py"  # change to "pnl_plotly.py" if that’s your filename
    if not gen.exists():
        print(f"[chart] generator missing at {gen}")
        return

    env = os.environ.copy()
    env["DATA_DIR"] = str(DATA_DIR)
    cmd = ["python", str(gen), "--timeframe", timeframe]
    print(f"[chart] running: {' '.join(cmd)} DATA_DIR={DATA_DIR}")
    try:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), env=env,
            capture_output=True, text=True, check=False
        )
        if proc.returncode != 0:
            print("[chart] non-zero exit:", proc.returncode)
        if proc.stdout:
            print("[chart][stdout]\n", proc.stdout)
        if proc.stderr:
            print("[chart][stderr]\n", proc.stderr)
    except Exception as e:
        print("[chart] exception:", e)


def read_closed_positions(limit: int = 200):
    """Read closed_positions.csv for the Trade History table."""
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
async def home(request: Request, tf: str = "YTD"):
    # Optional: force refresh on load
    # ensure_chart_generated(tf)
    closed_rows = read_closed_positions()
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "timeframe": tf,
            "closed_positions": closed_rows,
            "pnl_url": "/files/pnl_dashboard.html",
        },
    )

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/manifest.json")
def manifest():
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            return JSONResponse(json.load(f))
    return JSONResponse({"reports": []})


# ---------------- Swing Reports (grid of cards + detail) ----------------
@app.get("/reports", response_class=HTMLResponse)
async def swing_reports(request: Request):
    items = load_manifest()
    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return templates.TemplateResponse("swing_reports.html", {"request": request, "items": items})

@app.get("/reports/{slug}", response_class=HTMLResponse)
async def report_detail(request: Request, slug: str):
    item = find_report(slug)
    if not item:
        raise HTTPException(status_code=404, detail="Report not found")

    file_path = (ROOT / item["file"]).resolve()
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report file missing")

    ext = file_path.suffix.lower()

    # Decide which mount to use: /files (DATA_DIR) or /assets (repo)
    try:
        in_data_dir = DATA_DIR in file_path.parents or file_path == DATA_DIR
    except Exception:
        in_data_dir = False
    base_prefix = "/files" if in_data_dir else "/assets"

    # Markdown → render to HTML
    if ext in {".md", ".markdown"}:
        try:
            import markdown  # ensure 'markdown' is in requirements.txt
        except ImportError:
            raise HTTPException(status_code=500, detail="Install 'markdown' to render .md files")
        html = markdown.markdown(file_path.read_text(encoding="utf-8"), extensions=["extra", "tables", "toc"])
        return templates.TemplateResponse(
            "report_detail.html",
            {"request": request, "item": item, "content_html": html, "embed_src": None,
             "is_csv": False, "csv_rows": [], "csv_cols": []},
        )

    # HTML/PDF → show in iframe (served via correct static mount)
    if ext in {".html", ".pdf"}:
        embed_src = f"{base_prefix}/{item['file']}"
        return templates.TemplateResponse(
            "report_detail.html",
            {"request": request, "item": item, "content_html": None, "embed_src": embed_src,
             "is_csv": False, "csv_rows": [], "csv_cols": []},
        )

    # CSV → preview first rows
    if ext == ".csv":
        df = pd.read_csv(file_path)
        preview = df.head(200).fillna("").to_dict(orient="records")
        return templates.TemplateResponse(
            "report_detail.html",
            {"request": request, "item": item, "content_html": None, "embed_src": None,
             "is_csv": True, "csv_rows": preview, "csv_cols": list(df.columns)},
        )

    # Fallback: try to embed/download
    embed_src = f"{base_prefix}/{item['file']}"
    return templates.TemplateResponse(
        "report_detail.html",
        {"request": request, "item": item, "content_html": None, "embed_src": embed_src,
         "is_csv": False, "csv_rows": [], "csv_cols": []},
    )


# ---------------- Legacy redirects ----------------
@app.get("/positions")
async def legacy_positions_redirect():
    return RedirectResponse(url="/reports", status_code=307)

@app.get("/swing_reports")
async def legacy_swing_reports_redirect():
    return RedirectResponse(url="/reports", status_code=307)

@app.get("/chart", response_class=RedirectResponse)
def chart_redirect():
    pnl_path = DATA_DIR / "pnl_dashboard.html"
    if not pnl_path.exists():
        raise HTTPException(status_code=404, detail="Chart has not been generated yet.")
    return RedirectResponse(url="/files/pnl_dashboard.html")


# ---------------- HTMX partials ----------------
@app.get("/partial/chart", response_class=HTMLResponse)
async def partial_chart(request: Request, tf: str = "YTD"):
    ensure_chart_generated(tf)  # make/refresh the saved HTML
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
