#!/usr/bin/env python3
"""
Generate Buffer Scale Analysis Figure - KNEEDLE ELBOW DETECTION
----------------------------------------------------------------
Key improvements:
    1. Uses FDR-corrected bootstrap results
    2. Applies Kneedle algorithm for elbow detection (Satopaa et al., 2011)
    3. Shows Kneedle elbow point instead of arbitrary 90% threshold

Outputs:
    - figures/Figure_Buffer_Scale_ByCategory.png
    - figures/Figure_Buffer_Scale_Urban_Focus.png  
    - analysis_outputs/Table_Optimal_Buffer_Kneedle.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# CONFIG - Using FDR-corrected bootstrap results
# ==============================================================================
file_path = './analysis_outputs/Table_Temporal_Bootstrap_FDR_Corrected.csv'
OUTPUT_DIR = '../manuscript/figures'
OUTPUT_CSV = './analysis_outputs/Table_Optimal_Buffer_Kneedle.csv'

df = pd.read_csv(file_path)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# AESTHETICS
# ==============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8

# Colors for LULC categories
category_colors = {
    'Residential_Urban': "#D55E00",  # Vermilion
    'Water_Bodies': "#0072B2",       # Blue
    'Aquaculture': "#CC79A7",        # Reddish Purple
    'Rice_Cultivation': "#009E73",
    'Rice_Paddy': "#009E73",
    'Forest': "#228B22",
    'Wetland': "#56B4E9",
    'Other_Crops': "#E69F00",
    'Others': "#999999"
}

metric_styles = {
    'composition_pct': '-',
    'pct': '-',
    '%': '-',
    'LPI': '--',
    'PD': '-.',
    'ED': ':',
    'LSI': (0, (5, 5))
}

def get_style_and_category(metric_full_name):
    c = "#333333"
    ls = "-"
    found_cat = "Unknown"
    for cat, color in category_colors.items():
        if cat in metric_full_name:
            c = color
            found_cat = cat
            break
    for m_type, style in metric_styles.items():
        if m_type in metric_full_name:
            ls = style
            break
    return c, ls, found_cat


def kneedle_elbow(buffers, correlations):
    """Find elbow point using Kneedle algorithm (Satopaa et al., 2011).
    Returns: buffer distance at elbow (maximum curvature point)."""
    x = np.array(buffers).astype(float)
    y = np.abs(np.array(correlations))
    
    if len(x) < 3:
        return x[y.argmax()]
    
    # Normalize to [0,1]
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-10)
    
    # Find point with maximum distance from diagonal (elbow)
    p1 = np.array([x_n[0], y_n[0]])
    p2 = np.array([x_n[-1], y_n[-1]])
    
    distances = []
    for i in range(len(x_n)):
        p = np.array([x_n[i], y_n[i]])
        d = np.abs(np.cross(p2 - p1, p1 - p)) / (np.linalg.norm(p2 - p1) + 1e-10)
        distances.append(d)
    
    elbow_idx = np.argmax(distances)
    return x[elbow_idx]


def clean_label(metric):
    return (metric.replace('Residential_Urban', 'Urban')
                  .replace('Rice_Cultivation', 'Rice')
                  .replace('Rice_Paddy', 'Rice')
                  .replace('Aquaculture', 'Aqua')
                  .replace('Water_Bodies', 'Water')
                  .replace('Other_Crops', 'Other')
                  .replace('_composition_pct', ' %')
                  .replace('_config_', ' ')
                  .replace('composition_pct', '%')
                  .replace('_', ' '))


wq_params = ['DO', 'BOD5', 'COD', 'NH4', 'NO3', 'TN', 'TSS', 'PO4', 'TP']
TOP_N_CATEGORIES = 5

# ==============================================================================
# FIGURE 1: 3x3 Panel by Category Champions (Original Style)
# ==============================================================================
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.flatten()

optimal_results = []

for i, wq in enumerate(wq_params):
    ax = axes[i]
    wq_data = df[df['WQ_Parameter'] == wq].copy()
    
    if wq_data.empty:
        ax.set_title(f"{wq} (No Data)", fontweight='bold')
        ax.axis('off')
        continue
    
    target_positive = (wq != 'DO')
    
    # Find champion for each category
    category_champions = {}
    for metric in wq_data['LULC_Metric'].unique():
        _, _, category = get_style_and_category(metric)
        if category == "Unknown": continue
        
        subset = wq_data[wq_data['LULC_Metric'] == metric]
        if subset.empty: continue
        
        idx_max = subset['Spearman_r'].abs().idxmax()
        r_val = subset.loc[idx_max, 'Spearman_r']
        buffer_at_peak = subset.loc[idx_max, 'Buffer_km']
        
        # Find Kneedle elbow buffer
        sub_sorted = subset.sort_values('Buffer_km')
        elbow_buf = kneedle_elbow(sub_sorted['Buffer_km'].values, 
                                  sub_sorted['Spearman_r'].values)
        
        is_expected_sign = (r_val > 0) if target_positive else (r_val < 0)
        score = abs(r_val) + (1.0 if is_expected_sign else 0.0)
        
        if category not in category_champions:
            category_champions[category] = (metric, score, r_val, buffer_at_peak, elbow_buf)
        elif score > category_champions[category][1]:
            category_champions[category] = (metric, score, r_val, buffer_at_peak, elbow_buf)
    
    ranked = sorted(category_champions.values(), key=lambda x: x[1], reverse=True)[:TOP_N_CATEGORIES]
    
    if not ranked: continue
    
    # Global best
    best_metric, _, best_r, peak_buf, elbow_buf = ranked[0]
    best_data = wq_data[wq_data['LULC_Metric'] == best_metric]
    
    n_col = 'n_raw_observations' if 'n_raw_observations' in best_data.columns else 'n_station_years'
    peak_row = best_data[best_data['Buffer_km'] == peak_buf]
    peak_n = int(peak_row[n_col].values[0]) if len(peak_row) > 0 else 0
    
    # Store result
    optimal_results.append({
        'WQ_Parameter': wq,
        'Peak_Buffer_km': int(peak_buf),
        'Kneedle_Elbow_km': int(elbow_buf),
        'Best_LULC_Metric': best_metric,
        'Peak_Spearman_r': best_r,
        'n_station_years': peak_n
    })
    
    # Plot lines
    for j, (metric, score, r_val, buf_peak, buf_elbow) in enumerate(ranked):
        subset = wq_data[wq_data['LULC_Metric'] == metric].sort_values('Buffer_km')
        
        label_clean = clean_label(metric)
        max_r = subset.loc[subset['Spearman_r'].abs().idxmax(), 'Spearman_r']
        line_peak = subset.loc[subset['Spearman_r'].abs().idxmax(), 'Buffer_km']
        legend_label = f"{label_clean} ($r_s$={max_r:.2f}@{int(line_peak)}km)"
        
        color, ls, _ = get_style_and_category(metric)
        lw = 2.5 if j == 0 else 1.5
        z_order = 10 - j
        
        ax.plot(subset['Buffer_km'], subset['Spearman_r'],
                label=legend_label, color=color, linestyle=ls, linewidth=lw, zorder=z_order)
        
        if j < 3 and 'r_CI_lower' in subset.columns:
            ax.fill_between(subset['Buffer_km'], subset['r_CI_lower'], subset['r_CI_upper'],
                           color=color, alpha=0.15, linewidth=0, zorder=z_order-5)
    
    # Reference lines - show BOTH peak and plateau
    ax.axvline(peak_buf, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(elbow_buf, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Highlight 90% plateau zone
    ax.axvspan(elbow_buf, peak_buf, alpha=0.1, color='green', zorder=0)
    
    title_str = f"{wq} (Peak: {int(peak_buf)}km, Elbow: {int(elbow_buf)}km)"
    ax.set_title(title_str, fontweight='bold')
    ax.set_xlim(0, 31)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='best', fontsize=7, frameon=True, framealpha=0.95, edgecolor='lightgray')
    
    if i >= 6: ax.set_xlabel('Buffer Radius (km)', fontweight='bold')
    if i % 3 == 0: ax.set_ylabel("Spearman's $r_s$", fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'Figure_Buffer_Scale_ByCategory.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'Figure_Buffer_Scale_ByCategory.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/Figure_Buffer_Scale_ByCategory.png")

# ==============================================================================
# FIGURE 2: URBAN-FOCUSED (All Urban metrics for each WQ)
# ==============================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(12, 7))
axes2 = axes2.flatten()

urban_wq = ['DO', 'BOD5', 'COD', 'TN', 'TSS', 'NH4']
urban_results = []

for i, wq in enumerate(urban_wq):
    ax = axes2[i]
    wq_data = df[df['WQ_Parameter'] == wq].copy()
    
    # Get ALL Urban metrics
    urban_metrics = wq_data[wq_data['LULC_Metric'].str.contains('Residential_Urban')]
    
    if urban_metrics.empty:
        ax.set_title(f"{wq} vs Urban (No Data)", fontweight='bold')
        continue
    
    # Find overall best for this WQ-Urban combo
    global_peak_idx = urban_metrics['Spearman_r'].abs().idxmax()
    global_peak_r = urban_metrics.loc[global_peak_idx, 'Spearman_r']
    global_peak_buf = urban_metrics.loc[global_peak_idx, 'Buffer_km']
    global_peak_metric = urban_metrics.loc[global_peak_idx, 'LULC_Metric']
    
    # Find Kneedle elbow buffer
    best_sub = urban_metrics[urban_metrics['LULC_Metric'] == global_peak_metric].sort_values('Buffer_km')
    elbow_buf = kneedle_elbow(best_sub['Buffer_km'].values, 
                              best_sub['Spearman_r'].values)
    
    urban_results.append({
        'WQ_Parameter': wq,
        'Peak_Buffer_km': int(global_peak_buf),
        'Kneedle_Elbow_km': int(elbow_buf),
        'Best_Urban_Metric': global_peak_metric,
        'Peak_r': global_peak_r
    })
    
    # Plot ALL Urban metrics with different line styles
    for metric in urban_metrics['LULC_Metric'].unique():
        sub = urban_metrics[urban_metrics['LULC_Metric'] == metric].sort_values('Buffer_km')
        
        label_clean = metric.replace('Residential_Urban_', '').replace('composition_', '').replace('config_', '')
        max_r = sub.loc[sub['Spearman_r'].abs().idxmax(), 'Spearman_r']
        peak_km = sub.loc[sub['Spearman_r'].abs().idxmax(), 'Buffer_km']
        
        _, ls, _ = get_style_and_category(metric)
        lw = 2.5 if metric == global_peak_metric else 1.2
        alpha = 1.0 if metric == global_peak_metric else 0.6
        
        ax.plot(sub['Buffer_km'], sub['Spearman_r'], linestyle=ls, linewidth=lw, alpha=alpha,
                color='#D55E00', label=f"{label_clean} ({max_r:+.2f}@{int(peak_km)}km)")
    
    # Reference lines
    ax.axvline(global_peak_buf, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(elbow_buf, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label=f"Elbow @ {int(elbow_buf)}km")
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_title(f"{wq} vs Urban (Peak: {int(global_peak_buf)}km, r={global_peak_r:+.2f})", fontweight='bold')
    ax.set_xlim(0, 31)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='best', fontsize=7, frameon=True, framealpha=0.95)
    
    if i >= 3: ax.set_xlabel('Buffer Radius (km)', fontweight='bold')
    if i % 3 == 0: ax.set_ylabel("Spearman's $r_s$", fontweight='bold')

plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, 'Figure_Buffer_Scale_Urban_Focus.png'), dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(OUTPUT_DIR, 'Figure_Buffer_Scale_Urban_Focus.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/Figure_Buffer_Scale_Urban_Focus.png")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
results_df = pd.DataFrame(optimal_results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")

print("\n=== OPTIMAL BUFFER SUMMARY (All Categories) ===")
print(results_df.to_string(index=False))
print(f"\nMedian Peak: {results_df['Peak_Buffer_km'].median():.0f} km")
print(f"Median Kneedle Elbow: {results_df['Kneedle_Elbow_km'].median():.0f} km")

print("\n=== URBAN-FOCUSED SUMMARY ===")
urban_df = pd.DataFrame(urban_results)
print(urban_df.to_string(index=False))
print(f"\nUrban Median Peak: {urban_df['Peak_Buffer_km'].median():.0f} km")
print(f"Urban Median Elbow: {urban_df['Kneedle_Elbow_km'].median():.0f} km")

plt.show()
