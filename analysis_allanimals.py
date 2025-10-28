# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 09:04:54 2025

@author: JoanaCatarino
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
from matplotlib.lines import Line2D

# ----------------- CONFIG -----------------
# Map each condition to the list of animal IDs in that condition
animals_by_condition = {
    "1day": [916931, 943882, 943883, 952505],
    "2days": [943886, 943890, 943891], 
    "3days": [916934, 943884],
    "5days": [943885, 943889, 943888]
}

# Base path for your per-animal files
BASE = r"L:/dmclab/Joana/Tamoxifen_pilot/Analysis/data_prep"

# Where to save plots
OUT = r"L:/dmclab/Joana/Tamoxifen_pilot/Analysis/plots"

# PFC acronyms
PFC_acronyms = ["FRP", "PL", "ILA", "ACAd", "ACAv", 
                "ORBm", "ORBl", "ORBvl", "Mos", "AId", "AIv"]

# ------------------------------------------

def load_one_animal(animal_id, condition):
    path = f"{BASE}/{animal_id}_{condition}_Tamoxifen_data.csv"
    df = pd.read_csv(path)
    return df

def is_pfc_series(df):
    return df["acronym"].str.startswith(tuple(PFC_acronyms), na=False)

def clean_layer_name(layer_value):
    # Make layer labels consistent across animals
    s = str(layer_value)
    return s.replace("layer ", "").strip()

# Collect per-animal totals and layers
totals_records = []   # rows: animal_id, condition, metric, value
layer_records  = []   # rows: animal_id, condition, layer, count

# Track all layer names we see so we can align columns later
all_layers = set()

for condition, ids in animals_by_condition.items():
    for animal_id in ids:
        df = load_one_animal(animal_id, condition)

        # Totals
        total_cells = len(df)
        pfc_mask = is_pfc_series(df)
        pfc_cells = int(pfc_mask.sum())

        totals_records.append({
            "animal_id": animal_id,
            "condition": condition,
            "metric": "Total cells",
            "value": total_cells
        })
        totals_records.append({
            "animal_id": animal_id,
            "condition": condition,
            "metric": "PFC cells",
            "value": pfc_cells
        })

        # PFC-by-layer counts (per animal)
        pfc_df = df[pfc_mask].copy()
        if not pfc_df.empty:
            pfc_df["layer_clean"] = pfc_df["layer"].map(clean_layer_name)
            counts = pfc_df["layer_clean"].value_counts()
        else:
            counts = pd.Series(dtype=int)

        # Record counts for each layer we observed in this animal
        for layer_name, count in counts.items():
            all_layers.add(layer_name)
            layer_records.append({
                "animal_id": animal_id,
                "condition": condition,
                "layer": layer_name,
                "count": int(count)
            })

# Convert to DataFrames
totals_df = pd.DataFrame(totals_records)
layer_df  = pd.DataFrame(layer_records)

# Make sure layer_df has rows for missing layers as zeros (so SEM works properly)
if len(layer_df):
    # Create a complete index of (animal_id, condition, layer)
    all_layers = sorted(all_layers, key=lambda x: (x.isdigit(), x))  # nice ordering if numeric
    animals_long = []
    for condition, ids in animals_by_condition.items():
        for animal_id in ids:
            for L in all_layers:
                animals_long.append({"animal_id": animal_id, "condition": condition, "layer": L})
    complete = pd.DataFrame(animals_long)
    layer_df = complete.merge(layer_df, on=["animal_id", "condition", "layer"], how="left")
    layer_df["count"] = layer_df["count"].fillna(0).astype(int)

# -------- Aggregate: mean ± SEM --------
def sem(x):
    x = np.asarray(x, dtype=float)
    n = np.isfinite(x).sum()
    if n <= 1:
        return np.nan
    return np.nanstd(x, ddof=1) / np.sqrt(n)

# Totals summary per condition and metric
totals_summary = (
    totals_df
    .groupby(["condition", "metric"], as_index=False)
    .agg(mean=("value", "mean"), sem=("value", sem), n=("value", "size"))
)

# Layers summary per condition and layer
layers_summary = (
    layer_df
    .groupby(["condition", "layer"], as_index=False)
    .agg(mean=("count", "mean"), sem=("count", sem), n=("count", "size"))
)


# Exclude Nan from layers
layers_summary = layers_summary[layers_summary["layer"].notna()]
layers_summary = layers_summary[~layers_summary["layer"].astype(str).str.lower().eq("nan")]

print("Totals summary:\n", totals_summary)
print("\nLayers summary (first rows):\n", layers_summary.head())

# ------------------ Plot A: Total vs PFC cells (mean ± SEM by condition) ------------------
sns.set_context("talk")
sns.set_style("white")

conditions = list(animals_by_condition.keys())
metrics = ["Total cells", "PFC cells"]
width = 0.38
x = np.arange(len(conditions))

means = {m: [totals_summary.query("condition == @c and metric == @m")["mean"].values[0]
             for c in conditions] for m in metrics}
sems  = {m: [totals_summary.query("condition == @c and metric == @m")["sem"].values[0]
             for c in conditions] for m in metrics}

plt.figure(figsize=(7,6), dpi=500)
ax = plt.gca()

bars1 = ax.bar(x - width/2, means["Total cells"], width,capsize=4, label="Total cells", color="#85BDA6", alpha=0.9)
bars2 = ax.bar(x + width/2, means["PFC cells"], width, capsize=4, label="PFC cells",  color="#08605F", alpha=0.9)

#use when we have more animals
'''
#bars1 = ax.bar(x - width/2, means["Total cells"], width, yerr=sems["Total cells"],
               capsize=4, label="Total cells", color="#85BDA6", alpha=0.9)
#bars2 = ax.bar(x + width/2, means["PFC cells"],   width, yerr=sems["PFC cells"],
               capsize=4, label="PFC cells",  color="#08605F", alpha=0.9)
'''
# ---- overlay animal dots ----
for j, cond in enumerate(conditions):
    for m, x_offset in zip(metrics, [-width/2, width/2]):
        yvals = (totals_df
                 .query("condition == @cond and metric == @m")
                 .sort_values("animal_id")["value"].to_numpy())
        if yvals.size == 0:
            continue
        x_center = j + x_offset
        ax.scatter(np.full(yvals.size, x_center), yvals,
                   s=28, marker='o', facecolors="#1f1f1f", edgecolors="white",
                   linewidths=0.5, alpha=0.9, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel("Total Cells")
ax.set_title("Total vs PFC cells")

# add a legend entry for the dots
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0],[0], marker='o', linestyle='', color="#1f1f1f",
                      markerfacecolor="#1f1f1f", markeredgecolor="white",
                      label="single animal"))
labels.append("Animals (dots)")
ax.legend(handles, labels)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()


plt.savefig(f"{OUT}/Tamoxifen_Total_vs_PFC_by_condition.png", dpi=500)
plt.savefig(f"{OUT}/Tamoxifen_Total_vs_PFC_by_condition.svg", dpi=500)

# ------------------ Plot B: PFC split by layer (mean ± SEM by condition) ------------------
# ---- choose a fixed color for each condition ----
colors_by_condition = {
    "1day": "#B5D9CC",
    "2days":  "#5B9E9A",
    "3days": "#9490B4",
    "5days":  "#815D9F"
}

default_color = "#5c5c5c"

if len(layers_summary):
    
    # enforce order: 1, 2/3, 5, 6a, 6b (preserving original casing found in your data)
    desired = ["1", "2/3", "5", "6a", "6b"]

    # map normalized label -> original label present in your data
    present = layers_summary["layer"].astype(str).str.strip()
    norm_to_orig = dict(zip(present.str.lower(), present))

    # keep only those that actually exist, in desired order
    layers_order = [norm_to_orig[d] for d in desired if d in norm_to_orig]
   
    conds = conditions
    width = 0.8 / max(1, len(conds))
    x = np.arange(len(layers_order))

    plt.figure(figsize=(10,6), dpi=500)
    ax = plt.gca()

    # ---- bars with condition colors ----
    for i, cond in enumerate(conds):
        sub = layers_summary[layers_summary["condition"] == cond].set_index("layer").reindex(layers_order)
        means = sub["mean"].values
        errs  = sub["sem"].values
        c     = colors_by_condition.get(cond, default_color)

        ax.bar(
            x + i*width - (len(conds)-1)*width/2,
            means,
            width,
            #yerr=errs,
            capsize=3,
            label=cond,
            color=c,
            edgecolor="none",
            alpha=0.95,
            zorder=2
        )

    # ---- centered per-animal dots using the same condition color ----
    for i, cond in enumerate(conds):
        per_cond = layer_df[layer_df["condition"] == cond]
        c = colors_by_condition.get(cond, default_color)
        for k, L in enumerate(layers_order):
            yvals = per_cond.loc[per_cond["layer"] == L, "count"].to_numpy()
            if yvals.size == 0:
                continue
            x_center = k + i*width - (len(conds)-1)*width/2
            ax.scatter(
                np.full(yvals.size, x_center),
                yvals,
                s=22,
                marker='o',
                facecolors="black",
                edgecolors="gray",   # keeps dots visible on top of bar
                linewidths=0.6,
                alpha=0.95,
                zorder=3
            )

    ax.set_xticks(x)
    ax.set_xticklabels(layers_order, rotation=0)
    ax.set_ylabel("Total cells")
    ax.set_title("PFC cells split by layer")
    ax.legend(title="Condition")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    plt.savefig(f"{OUT}/Tamoxifen_PFC_layers_by_condition.png", dpi=500)
    plt.savefig(f"{OUT}/Tamoxifen_PFC_layers_by_condition.svg", dpi=500)