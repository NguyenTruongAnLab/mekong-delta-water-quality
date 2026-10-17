"""Analyze MRC discharge time-series JSON for a station.

Input is the raw JSON from the MRC time-series API endpoint:
  https://timeseries.api.mrcmekong.org/api/v1/ts/data/timeSeriesCorrectedData/{ts}

This script:
- flattens Points[] to a tidy dataframe
- computes summary statistics and simple QA/gap metrics
- computes monthly climatology + annual statistics
- produces flow-duration curve
- writes CSV + PNG outputs to analysis_outputs/

Example:
  py -3 analyze_mrc_discharge_station.py \
    --json analysis_outputs/mrc_discharge_kratie_ts22a8130e_1923-12-31_2026-01-05.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Matplotlib is used for writing figures (no GUI needed).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Meta:
    ts_id: str
    location_identifier: str
    parameter: str
    interval: str
    unit: str


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def payload_meta(payload: dict[str, Any], fallback_ts_id: str) -> Meta:
    return Meta(
        ts_id=str(payload.get("UniqueId") or fallback_ts_id),
        location_identifier=str(payload.get("LocationIdentifier") or "unknown_location"),
        parameter=str(payload.get("Parameter") or "unknown_parameter"),
        interval=str(payload.get("Interval") or "unknown_interval"),
        unit=str(payload.get("Unit") or ""),
    )


def points_to_df(payload: dict[str, Any]) -> pd.DataFrame:
    points = payload.get("Points") or []
    if not points:
        return pd.DataFrame(columns=["timestamp", "value"]).set_index("timestamp")

    ts = pd.to_datetime([p.get("Timestamp") for p in points], utc=True, errors="coerce")
    val = pd.to_numeric([(p.get("Value") or {}).get("Numeric") for p in points], errors="coerce")

    df = pd.DataFrame({"timestamp": ts, "value": val}).dropna(subset=["timestamp"]).set_index("timestamp")
    df = df.sort_index()
    return df


def gap_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n_points": 0, "n_gaps_gt_1d": 0, "max_gap_days": np.nan}

    # Expect daily-ish data, but compute actual deltas.
    deltas = df.index.to_series().diff().dropna()
    gap_days = deltas.dt.total_seconds() / 86400.0
    gaps_gt_1d = gap_days[gap_days > 1.5]  # tolerate DST/timezone oddities

    return {
        "n_points": int(df.shape[0]),
        "n_gaps_gt_1d": int(gaps_gt_1d.shape[0]),
        "max_gap_days": float(gap_days.max()) if len(gap_days) else np.nan,
        "start": df.index.min().isoformat(),
        "end": df.index.max().isoformat(),
    }


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            {
                "metric": ["count"],
                "value": [0],
            }
        )

    s = df["value"].dropna()
    pct = s.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

    rows = [
        ("count", int(s.shape[0])),
        ("mean", float(s.mean())),
        ("std", float(s.std(ddof=1))),
        ("min", float(s.min())),
        ("max", float(s.max())),
    ]
    rows += [(f"p{int(q*100):02d}", float(v)) for q, v in pct.items()]

    return pd.DataFrame(rows, columns=["metric", "value"])


def monthly_climatology(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "mean", "p10", "p50", "p90", "count"]).set_index("month")

    g = df["value"].groupby(df.index.month)
    out = pd.DataFrame(
        {
            "mean": g.mean(),
            "p10": g.quantile(0.10),
            "p50": g.quantile(0.50),
            "p90": g.quantile(0.90),
            "count": g.count(),
        }
    )
    out.index.name = "month"
    return out


def annual_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["year", "mean", "max", "min", "count"]).set_index("year")

    g = df["value"].groupby(df.index.year)
    out = pd.DataFrame({"mean": g.mean(), "max": g.max(), "min": g.min(), "count": g.count()})
    out.index.name = "year"
    return out


def linear_trend_annual_mean(annual: pd.DataFrame) -> dict[str, Any]:
    """Simple OLS on annual mean vs year (no autocorrelation correction)."""
    if annual.empty:
        return {"slope_per_year": np.nan, "intercept": np.nan, "r2": np.nan}

    y = annual["mean"].to_numpy(dtype=float)
    x = annual.index.to_numpy(dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 3:
        return {"slope_per_year": np.nan, "intercept": np.nan, "r2": np.nan}

    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"slope_per_year": float(slope), "intercept": float(intercept), "r2": float(r2)}


def flow_duration_curve(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["exceedance_probability", "value"])

    v = df["value"].dropna().to_numpy(dtype=float)
    v = np.sort(v)[::-1]  # descending
    n = v.size
    # Weibull plotting position: P(exceed) = rank/(n+1)
    ranks = np.arange(1, n + 1, dtype=float)
    p = ranks / (n + 1.0)
    return pd.DataFrame({"exceedance_probability": p, "value": v})


def plot_monthly_climatology(meta: Meta, clim: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)

    months = np.arange(1, 13)
    ax.plot(months, clim.reindex(months)["mean"], label="Mean", color="#1f77b4")
    ax.fill_between(
        months,
        clim.reindex(months)["p10"].to_numpy(),
        clim.reindex(months)["p90"].to_numpy(),
        color="#1f77b4",
        alpha=0.2,
        label="P10–P90",
    )
    ax.plot(months, clim.reindex(months)["p50"], label="Median", color="#ff7f0e", linewidth=1.5)

    ax.set_title(f"Monthly discharge climatology — {meta.location_identifier}")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Discharge ({meta.unit})" if meta.unit else "Discharge")
    ax.set_xticks(months)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_annual_mean(meta: Meta, annual: pd.DataFrame, trend: dict[str, Any], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)

    ax.plot(annual.index, annual["mean"], color="#2ca02c", linewidth=1.2, label="Annual mean")

    if np.isfinite(trend.get("slope_per_year", np.nan)) and np.isfinite(trend.get("intercept", np.nan)):
        x = annual.index.to_numpy(dtype=float)
        yhat = trend["slope_per_year"] * x + trend["intercept"]
        ax.plot(annual.index, yhat, color="black", linestyle="--", linewidth=1.2, label=f"OLS trend ({trend['slope_per_year']:+.1f} /yr)")

    ax.set_title(f"Annual mean discharge — {meta.location_identifier}")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Discharge ({meta.unit})" if meta.unit else "Discharge")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_flow_duration_curve(meta: Meta, fdc: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)

    ax.plot(fdc["exceedance_probability"], fdc["value"], color="#9467bd", linewidth=1.2)
    ax.set_xscale("log")
    ax.invert_xaxis()  # common FDC convention: high flows on left (low exceedance)

    ax.set_title(f"Flow duration curve — {meta.location_identifier}")
    ax.set_xlabel("Exceedance probability (Weibull)")
    ax.set_ylabel(f"Discharge ({meta.unit})" if meta.unit else "Discharge")
    ax.grid(True, which="both", alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to raw JSON payload")
    ap.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent / "analysis_outputs"),
        help="Output directory (default: analysis_outputs next to this script)",
    )
    args = ap.parse_args()

    json_path = Path(args.json)
    outdir = Path(args.outdir)

    payload = load_payload(json_path)
    meta = payload_meta(payload, fallback_ts_id="unknown_ts")

    df = points_to_df(payload)

    # If the API didn't include a unit, infer common discharge unit.
    if not meta.unit and meta.parameter.lower() == "discharge":
        meta = Meta(
            ts_id=meta.ts_id,
            location_identifier=meta.location_identifier,
            parameter=meta.parameter,
            interval=meta.interval,
            unit="m³/s",
        )

    gaps = gap_metrics(df)
    summ = summary_stats(df)
    clim = monthly_climatology(df)
    annual = annual_stats(df)
    trend = linear_trend_annual_mean(annual)
    fdc = flow_duration_curve(df)

    stem = "mrc_discharge_" + "".join(ch if ch.isalnum() else "_" for ch in meta.location_identifier)

    # Write tidy time series
    ts_csv = outdir / f"{stem}_timeseries.csv"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(ts_csv)

    # Write summary outputs
    summary_csv = outdir / f"{stem}_summary_stats.csv"
    summ.to_csv(summary_csv, index=False)

    gaps_json = outdir / f"{stem}_gap_metrics.json"
    with gaps_json.open("w", encoding="utf-8") as f:
        json.dump({"meta": meta.__dict__, "gap_metrics": gaps, "trend_annual_mean": trend}, f, indent=2)

    clim_csv = outdir / f"{stem}_monthly_climatology.csv"
    clim.to_csv(clim_csv)

    annual_csv = outdir / f"{stem}_annual_stats.csv"
    annual.to_csv(annual_csv)

    fdc_csv = outdir / f"{stem}_flow_duration_curve.csv"
    fdc.to_csv(fdc_csv, index=False)

    # Plots
    plot_monthly_climatology(meta, clim, outdir / f"{stem}_monthly_climatology.png")
    plot_annual_mean(meta, annual, trend, outdir / f"{stem}_annual_mean.png")
    plot_flow_duration_curve(meta, fdc, outdir / f"{stem}_fdc.png")

    print(f"Loaded points: {gaps.get('n_points')}")
    print(f"Date range (UTC): {gaps.get('start')}  →  {gaps.get('end')}")
    print(f"Saved: {ts_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {gaps_json}")
    print(f"Saved: {clim_csv}")
    print(f"Saved: {annual_csv}")
    print(f"Saved: {fdc_csv}")
    print(f"Saved plots under: {outdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
