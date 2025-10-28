# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 17:16:34 2025

@author: JoanaCatarino
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Select the animal to pre-process and injection days
animal_id = 952502
injection_days = '3days'

# Import prep data for a single animal
data_p = pd.read_csv('L:/dmclab/Joana/Tamoxifen_pilot/Analysis/data_prep/'f'{animal_id}_{injection_days}_Tamoxifen_data.csv')

# Define PFC acronyms
PFC_acronyms = ["FRP", "PL", "ILA", "ACAd", "ACAv", 
                "ORBm", "ORBl", "ORBvl", "Mos", "AId", "AIv"]

# -------- Plot 1 ----------

# Plot total number of cells and cells found in PFC (FRP, PL, ILA, ACAd, ACAv, ORBm, ORBl, ORBvl, Mos, AId, AIv)
total_cells = len(data_p)
cells_PFC =  len(data_p[data_p["acronym"].str.startswith(tuple(PFC_acronyms), na=False)])

# Data for plotting
labels = ["Total cells", "PFC cells"]
values = [total_cells, cells_PFC]

# Plot total vs PFC cells
plt.figure(figsize=(3, 5), dpi= 500)
bars = plt.bar(labels, values, color=["#85BDA6", "#08605F"], width=0.6)
plt.ylabel("Number of cells")
plt.title(f"{injection_days} Tamoxifen", pad=20)
sns.despine()
plt.tight_layout()
plt.gca().spines["bottom"].set_visible(False)

# Add numbers on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + (0.01 * max(values)),
             f"{height:,}", ha='center', va='bottom', fontsize=10, color='#494949')

print("Total cells:", total_cells)
print("PFC cells:", cells_PFC)

plt.savefig('L:/dmclab/Joana/Tamoxifen_pilot/Analysis/plots/'f'{animal_id}_{injection_days}_Tamoxifen_Total_cells.png', dpi=500)
plt.savefig('L:/dmclab/Joana/Tamoxifen_pilot/Analysis/plots/'f'{animal_id}_{injection_days}_Tamoxifen_Total_cells.svg', dpi=500)

# --------- Plot 2 ---------

pfc_df = data_p[data_p["acronym"].str.startswith(tuple(PFC_acronyms), na=False)]

# Count by layer
PFC_by_layer = pfc_df["layer"].value_counts().sort_index()

# Clean labels: remove 'layer ' prefix if present
clean_labels = [str(l).replace("layer ", "") for l in PFC_by_layer.index]

# Plot bar chart by layer
plt.figure(figsize=(4,8), dpi= 500)
ax=PFC_by_layer.plot(kind="bar", color="#623B5A", width=0.6)

plt.ylabel("Number of cells")
plt.title(f"PFC cells split by layer ({injection_days} Tamoxifen)", pad=20)
plt.xticks(ticks=range(len(clean_labels)), labels=clean_labels, rotation=0)
sns.despine()

# Add numbers on top of bars
for i, v in enumerate(PFC_by_layer):
    ax.text(i, v + (0.01 * max(PFC_by_layer)), str(v), 
            ha='center', va='bottom', fontsize=10, color='#494949')

print(PFC_by_layer)

plt.savefig('L:/dmclab/Joana/Tamoxifen_pilot/Analysis/plots/'f'{animal_id}_{injection_days}_Tamoxifen_PFC_layers.png', dpi=500)
plt.savefig('L:/dmclab/Joana/Tamoxifen_pilot/Analysis/plots/'f'{animal_id}_{injection_days}_Tamoxifen_PFC_layers.svg', dpi=500)



