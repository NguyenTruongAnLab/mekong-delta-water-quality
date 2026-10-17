"""
Prepare EGRET-format input files for WRTDS analysis.

Creates Daily (discharge) and Sample (WQ) CSV files for each MRC station,
formatted for EGRET's readUserDaily() and readUserSample() functions.

Stations: Tan Chau, My Thuan, Can Tho, My Tho
Parameters: COD (KMnO4 method), NO3-N
"""
import pandas as pd
from pathlib import Path

# Paths
ROOT = Path(r"h:\My Drive\Project\CARE projects 2024-20xx\4. Mekong long-term\Python\JAXA\scripts")
WQ_CSV = ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv"
Q_CSV = ROOT / "data" / "hydrology" / "outputs" / "final_datasets" / "Mekong_Delta_Gold_Standard_Master_1985_2025.csv"
OUT_DIR = ROOT / "analysis" / "wrtds_egret"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Station mapping: WQ name -> discharge column name
STATIONS = {
    "Tan Chau": "TanChau",
    "My Thuan": "MyThuan",
    "Can Tho":  "CanTho",
    "My Tho":   "MyTho",
}

# Parameters to analyze
PARAMS = {
    "COD": "COD (mg/L)",
    "NO3": "NO3-N (mg/L)",
}


def prepare_daily(q_df: pd.DataFrame, q_col: str, stn_label: str) -> pd.DataFrame:
    """Format daily discharge for EGRET readUserDaily().

    Expected columns: agency, site, dateTime, Q (m³/s), code
    """
    daily = pd.DataFrame({
        "agency":   "MRC",
        "site":     stn_label,
        "dateTime": q_df["date"].dt.strftime("%Y-%m-%d"),
        "Q":        q_df[q_col],
        "code":     "",
    })
    return daily.dropna(subset=["Q"])


def prepare_sample(wq_df: pd.DataFrame, conc_col: str, stn_label: str) -> pd.DataFrame:
    """Format WQ samples for EGRET readUserSample().

    Expected columns: agency, site, dateTime, ConcLow, ConcHigh, Uncen
    """
    sub = wq_df[(wq_df[conc_col].notna()) & (wq_df[conc_col] > 0)].copy()
    sample = pd.DataFrame({
        "agency":   "DONRE",
        "site":     stn_label,
        "dateTime": sub["datetime"].dt.strftime("%Y-%m-%d"),
        "ConcLow":  sub[conc_col],
        "ConcHigh": sub[conc_col],
        "Uncen":    1,  # 1 = uncensored (detected)
    })
    return sample


def main():
    # Load water quality
    wq = pd.read_csv(WQ_CSV)
    wq["datetime"] = pd.to_datetime(wq["datetime"])

    # Load daily discharge
    q_raw = pd.read_csv(Q_CSV)
    q_raw["date"] = pd.to_datetime(q_raw["Timestamp"], format="mixed", dayfirst=False)

    print("=== Preparing EGRET input files ===\n")

    for wq_name, q_col in STATIONS.items():
        stn_wq = wq[wq["stationName"] == wq_name]
        stn_label = q_col

        # Daily discharge file (shared across parameters)
        daily = prepare_daily(q_raw, q_col, stn_label)
        daily_path = OUT_DIR / f"{stn_label}_Daily.csv"
        daily.to_csv(daily_path, index=False)
        print(f"  {stn_label}_Daily.csv: {len(daily)} days "
              f"({daily['dateTime'].iloc[0]} to {daily['dateTime'].iloc[-1]})")

        # Sample files per parameter
        for param_label, conc_col in PARAMS.items():
            sample = prepare_sample(stn_wq, conc_col, stn_label)
            sample_path = OUT_DIR / f"{stn_label}_{param_label}_Sample.csv"
            sample.to_csv(sample_path, index=False)
            print(f"  {stn_label}_{param_label}_Sample.csv: {len(sample)} samples")

        print()

    print(f"All files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
