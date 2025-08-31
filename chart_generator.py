# chart_generator.py  (rename from pnl_plotly.py if you want to match app.py)
import os
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- Paths & env ----------
BASE_DIR = Path(__file__).parent.resolve()

# Read CSVs from the repo by default. You can override with env if needed.
PNL_CSV = os.getenv("PNL_CSV", str(BASE_DIR / "account_summary.csv"))
POSITIONS_CSV = os.getenv("POSITIONS_CSV", str(BASE_DIR / "current_positions.csv"))

# Writable output directory:
# Locally defaults to ./reports; on Railway set: DATA_DIR=/data (with a Volume mounted there)
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "reports"))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "Date"
PNL_COL = "Daily_Realized_PnL"

# ---------- Data loading ----------
def load_data():
    df = pd.read_csv(PNL_CSV)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[PNL_COL] = pd.to_numeric(df[PNL_COL], errors="coerce").fillna(0.0)
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)

    pos_df = pd.read_csv(POSITIONS_CSV)
    pos_df["Average_Price"] = pd.to_numeric(pos_df["Average_Price"], errors="coerce").round(2)
    return df, pos_df

# ---------- Timeframe helpers ----------
def compute_window(dates: pd.Series, timeframe: str):
    if dates.empty:
        raise ValueError("No dates found in P&L CSV.")
    end_dt = pd.to_datetime(dates.max()).normalize()
    if timeframe == "YTD":
        start_dt = pd.Timestamp(year=end_dt.year, month=1, day=1)
        label = f"YTD ({end_dt.year})"
    else:
        months = {"1M": 1, "3M": 3, "6M": 6}[timeframe]
        start_dt = (end_dt - pd.DateOffset(months=months)) + pd.Timedelta(days=1)
        label = f"Last {months}M (ending {end_dt.date()})"
    return start_dt, end_dt, label

def build_daily_window(df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    mask = (df[DATE_COL] >= start_dt) & (df[DATE_COL] <= end_dt)
    df_win = df.loc[mask].copy()
    if df_win.empty:
        daily = pd.DataFrame(index=pd.date_range(start_dt, end_dt, freq="D"))
        daily["daily_pnl"] = 0.0
        daily["cum_pnl"] = 0.0
        return daily

    daily = (
        df_win.set_index(DATE_COL)[PNL_COL]
              .resample("D")
              .sum()
              .to_frame("daily_pnl")
    )
    daily["cum_pnl"] = daily["daily_pnl"].cumsum()

    if start_dt not in daily.index:
        daily.loc[start_dt, ["daily_pnl", "cum_pnl"]] = [0.0, 0.0]
    daily = daily[~daily.index.duplicated(keep="first")].sort_index()
    daily["cum_pnl"] = daily["cum_pnl"] - float(daily.loc[start_dt, "cum_pnl"])
    return daily

def insert_zero_crossings(x: np.ndarray, y: np.ndarray):
    x = x.astype("datetime64[ns]")
    x_out = [x[0]]
    y_out = [y[0]]
    for i in range(len(y) - 1):
        y0, y1 = y[i], y[i+1]
        x0, x1 = x[i], x[i+1]
        if (y0 > 0 and y1 < 0) or (y0 < 0 and y1 > 0):
            t = -y0 / (y1 - y0) if (y1 - y0) != 0 else 0.0
            xz = x0 + (x1 - x0) * t
            x_out.append(xz); y_out.append(0.0)
        x_out.append(x1); y_out.append(y1)
    return np.array(x_out, dtype="datetime64[ns]"), np.array(y_out, dtype=float)

def split_pos_neg(x: np.ndarray, y: np.ndarray):
    y_pos = y.copy()
    y_neg = y.copy()
    y_pos[y_pos < 0] = np.nan
    y_neg[y_neg > 0] = np.nan
    return y_pos, y_neg

def make_timeframe_payloads(df):
    payloads = {}
    for tf in ["1M", "3M", "6M", "YTD"]:
        start_dt, end_dt, label = compute_window(df[DATE_COL], tf)
        daily = build_daily_window(df, start_dt, end_dt)
        x = daily.index.values
        y = daily["cum_pnl"].values.astype(float)

        xz, yz = insert_zero_crossings(x, y)
        y_pos, y_neg = split_pos_neg(xz, yz)

        payloads[tf] = {
            "label": label,
            "x": xz,
            "y": yz,
            "y_pos": y_pos,
            "y_neg": y_neg,
            "start": start_dt,
            "end": xz[-1] if len(xz) else end_dt,
            "last_val": float(yz[-1]) if len(yz) else 0.0,
        }
    return payloads

def make_positions_table(pos_df: pd.DataFrame):
    view = pos_df[["Symbol", "Current_Position", "Position_Type", "Average_Price"]].copy()
    header = ["Symbol", "Position Size", "Position Type", "Average Price"]
    cells = [view[c].astype(str).tolist() for c in ["Symbol", "Current_Position", "Position_Type", "Average_Price"]]
    return header, cells

def make_monthly_table_cells_for_window(df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    mask = (df[DATE_COL] >= start_dt) & (df[DATE_COL] <= end_dt)
    win = df.loc[mask, [DATE_COL, PNL_COL]].copy()
    if win.empty:
        header = ["Month", "P&L"]
        cells = [["—"], ["0.00"]]
        fills_col0 = ['#1a1a1a']
        fills_col1 = ['#1a1a1a']
        return header, cells, [fills_col0, fills_col1]

    monthly = (
        win.set_index(DATE_COL)[PNL_COL]
           .resample("ME")
           .sum()
           .round(2)
    )
    months = [d.strftime("%b %Y") for d in monthly.index]
    pnls = monthly.tolist()
    pnl_str = [f"{v:+,.2f}" for v in pnls]

    n_rows = len(months)
    zebra = ['#1a1a1a' if i % 2 == 0 else '#151515' for i in range(n_rows)]
    fills_col0 = zebra[:]
    fills_col1 = [("rgba(0,255,136,0.10)" if v >= 0 else "rgba(255,77,109,0.12)") for v in pnls]

    header = ["Month", "P&L"]
    cells = [months, pnl_str]
    return header, cells, [fills_col0, fills_col1]

def _compute_dynamic_layout_sizes(pos_cells, monthly_cells_by_tf, top_chart_min_px=460, cell_px=26, header_px=34, extra_padding_px=24, margins_px=140):
    pos_rows = len(pos_cells[0]) if pos_cells and len(pos_cells) > 0 else 1
    monthly_rows_max = 1
    for cells in monthly_cells_by_tf.values():
        try:
            monthly_rows_max = max(monthly_rows_max, len(cells[0]))
        except Exception:
            pass

    bottom_needed_px = max(pos_rows, monthly_rows_max) * cell_px + header_px + extra_padding_px
    total_height_px = int(top_chart_min_px + bottom_needed_px + margins_px)

    top = float(top_chart_min_px)
    bottom = float(bottom_needed_px)
    total = top + bottom
    row_heights = [round(top / total, 4), round(bottom / total, 4)]
    return total_height_px, row_heights

# ---------- Figure ----------
def build_figure(df, pos_df, default_tf: str = "YTD"):
    payloads = make_timeframe_payloads(df)
    pos_header, pos_cells = make_positions_table(pos_df)
    order = ["1M", "3M", "6M", "YTD"]
    PRETTY_TF = {"1M": "1 Month", "3M": "3 Months", "6M": "6 Months", "YTD": "YTD"}

    def _title_html(tf):
        p = payloads[tf]
        val = p["last_val"]
        color = "#00ff88" if val >= 0 else "#ff4d6d"
        pretty = PRETTY_TF.get(tf, tf)
        return f"Account P&L — {pretty} — <span style='color:{color}'>{val:+,.2f}</span>"

    monthly_headers, monthly_cells, monthly_fills = {}, {}, {}
    for tf in order:
        p = payloads[tf]
        mh, mc, mf = make_monthly_table_cells_for_window(df, p["start"], p["end"])
        monthly_headers[tf] = mh; monthly_cells[tf] = mc; monthly_fills[tf] = mf

    total_height_px, dynamic_row_heights = _compute_dynamic_layout_sizes(
        pos_cells=pos_cells,
        monthly_cells_by_tf=monthly_cells,
        top_chart_min_px=460,
        cell_px=26,
        header_px=34,
        extra_padding_px=24,
        margins_px=140
    )

    def _visibility_args(active_tf):
        visible = []
        for tf in order:
            is_active = (tf == active_tf)
            visible.extend([is_active, is_active, is_active, is_active])
        visible.append(True)  # positions table
        for tf in order:
            visible.append(tf == active_tf)

        p = payloads[active_tf]
        xrange = [p["start"], p["end"]]
        return [
            {"visible": visible},
            {"xaxis": {"range": xrange}, "title": {"text": _title_html(active_tf)}}
        ]

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.6, 0.4],
        row_heights=dynamic_row_heights,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        specs=[[{"type": "xy", "colspan": 2}, None], [{"type": "table"}, {"type": "table"}]],
        shared_xaxes=False,
    )

    # default visible = default_tf (instead of hardcoded YTD)
    visible_mask = {tf: [False, False, False, False] for tf in order}
    visible_mask[default_tf] = [True, True, True, True]

    for tf in order:
        p = payloads[tf]
        fig.add_trace(
            go.Scatter(
                x=p["x"], y=p["y_pos"], mode="lines",
                line=dict(color="#00ff88", width=3),
                hoverinfo="skip", showlegend=False, visible=visible_mask[tf][0],
                fill="tozeroy", fillcolor="rgba(0,255,136,0.10)",
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=p["x"], y=p["y_neg"], mode="lines",
                line=dict(color="#ff4d6d", width=3),
                hoverinfo="skip", showlegend=False, visible=visible_mask[tf][1],
                fill="tozeroy", fillcolor="rgba(255,77,109,0.12)",
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[p["x"][-1]] if len(p["x"]) else [],
                y=[p["y"][-1]] if len(p["y"]) else [],
                mode="markers",
                marker=dict(color=("#00ff88" if p["last_val"] >= 0 else "#ff4d6d"), size=8),
                hoverinfo="skip", showlegend=False, visible=visible_mask[tf][2],
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=p["x"], y=p["y"], mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=12),
                hovertemplate="%{x|%b %d, %Y}<br>Δ P&L: %{y:,.2f}<extra></extra>",
                showlegend=False, visible=visible_mask[tf][3],
            ),
            row=1, col=1
        )

    fig.add_hline(y=0, line=dict(color="#888", width=1, dash="dash"), row=1, col=1)

    n_rows = len(pos_cells[0]) if pos_cells else 0
    n_cols = len(pos_header)
    row_colors = ['#1a1a1a' if i % 2 == 0 else '#151515' for i in range(n_rows)]
    fill_colors = [row_colors[:] for _ in range(n_cols)]
    fig.add_trace(
        go.Table(
            header=dict(values=pos_header, fill_color="#2a2a2a",
                        font=dict(color="white", size=12), align="center"),
            cells=dict(values=pos_cells, fill_color=fill_colors,
                       font=dict(color="white", size=11), align="center", height=26),
        ),
        row=2, col=1
    )

    for tf in order:
        fig.add_trace(
            go.Table(
                header=dict(values=monthly_headers[tf], fill_color="#2a2a2a",
                            font=dict(color="white", size=12), align="center"),
                cells=dict(values=monthly_cells[tf],
                           fill_color=monthly_fills[tf],
                           font=dict(color="white", size=11), align="center", height=26),
                columnwidth=[0.6, 0.4],
                visible=(tf == default_tf),
            ),
            row=2, col=2
        )

    p0 = make_timeframe_payloads(df)[default_tf]
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"Account P&L — {default_tf if default_tf!='YTD' else 'YTD'} — "
                 f"<span style='color:{'#00ff88' if p0['last_val'] >= 0 else '#ff4d6d'}'>{p0['last_val']:+,.2f}</span>",
            x=0.02, y=0.98, xanchor="left",
            font=dict(size=20, family="Arial", color="white")
        ),
        margin=dict(l=50, r=20, t=60, b=40),
        paper_bgcolor="#121212",
        plot_bgcolor="#151515",
        height=_compute_dynamic_layout_sizes(
            pos_cells=pos_cells,
            monthly_cells_by_tf={tf: make_monthly_table_cells_for_window(df, payloads[tf]["start"], payloads[tf]["end"])[1] for tf in order},
            top_chart_min_px=460, cell_px=26, header_px=34, extra_padding_px=24, margins_px=140
        )[0],
        hovermode="x unified",
        xaxis=dict(
            showgrid=True, gridcolor="#2a2a2a",
            showspikes=True, spikemode="across", spikethickness=0.5, spikecolor="#888",
            range=[payloads[default_tf]["start"], payloads[default_tf]["end"]],
        ),
        yaxis=dict(
            title="Cumulative P&L (window)",
            zeroline=False,
            showgrid=True, gridcolor="#2a2a2a",
            tickformat=",",
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.02, y=0.36,
            xanchor="left", yanchor="bottom",
            bgcolor="#1f1f1f",
            bordercolor="#00ff88",
            borderwidth=1,
            font=dict(color="#eaeaea", size=12),
            pad=dict(r=6, t=4, b=4, l=6),
            showactive=False,
            buttons=[
                dict(label="1M",  method="update", args=_visibility_args("1M")),
                dict(label="3M",  method="update", args=_visibility_args("3M")),
                dict(label="6M",  method="update", args=_visibility_args("6M")),
                dict(label="YTD", method="update", args=_visibility_args("YTD")),
            ]
        )],
    )
    return fig

# ---------- Save helper ----------
def save_outputs(fig):
    out_path = DATA_DIR / "pnl_dashboard.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    # No webbrowser.open in server environments
    print(f"Wrote chart to {out_path}")

# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate P&L dashboard HTML")
    parser.add_argument("--timeframe", choices=["1M", "3M", "6M", "YTD"], default="YTD",
                        help="Initial timeframe selection for the rendered figure")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    df, pos_df = load_data()
    fig = build_figure(df, pos_df, default_tf=args.timeframe)
    save_outputs(fig)
