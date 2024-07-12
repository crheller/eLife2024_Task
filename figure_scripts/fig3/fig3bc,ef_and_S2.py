"""
Summary of active vs. passive delta dprime for each target pair
category
"""
import os
rdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
import sys
sys.path.append(rdir)

from settings import RESULTS_DIR
from helpers.path_helpers import local_results_file
import pandas as pd
import numpy as np
import scipy.stats as ss
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sqrt = True
db = pd.read_csv(os.path.join(RESULTS_DIR, "db.csv"), index_col=0)
sites = db.site

pupil_regress = "_PR" # if pupil_regress="", then use raw data. if = "_PR" use the pupil-corrected results

# load decoding results
amodel = 'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'+pupil_regress
pmodel = 'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'+pupil_regress
active = []
passive = []
for site in sites:
    try:
        ares = pd.read_pickle(local_results_file(RESULTS_DIR, site, amodel, "output.pickle"))
        pres = pd.read_pickle(local_results_file(RESULTS_DIR, site, pmodel, "output.pickle"))
        ares["site"] = site
        pres["site"] = site
        area = db.loc[site, "area"]
        ares["area"] = area
        pres["area"] = area
        if sqrt:
            ares["dp"] = np.sqrt(ares["dp"])
            pres["dp"] = np.sqrt(pres["dp"])
        active.append(ares)
        passive.append(pres)
    except:
        print(f"results not found for site: {site}")
    
active = pd.concat(active)
passive = pd.concat(passive)
df = passive.merge(active, on=["site", "class", "e1", "e2", "area"])
df_drop = df[["dp_x", "dp_y", "e1", "e2", "site"]].drop_duplicates()
df = df.loc[df_drop.index]
df["delta"] = (df["dp_y"] - df["dp_x"]) / (df["dp_y"] + df["dp_x"])
df["delta_raw"] = df["dp_y"] - df["dp_x"]
df = df.loc[:, ["class", "area", "site", "dp_x", "dp_y", "delta", "delta_raw"]] # keep only needed, numeric types


## Scatter plot, group by site
categories = ["ref_ref", "tar_tar", "tar_cat"]
colors = ["tab:blue", "tab:grey", "tab:red"]
s = 15
f, ax = plt.subplots(1, 2, figsize=(4, 2))

for cat, c in zip(categories, colors):
    ax[0].scatter(
        df[(df["class"]==cat) & (df.area=="A1")].drop(columns=["class"]).groupby(by=["site", "area"]).mean()["dp_x"],
        df[(df["class"]==cat) & (df.area=="A1")].drop(columns=["class"]).groupby(by=["site", "area"]).mean()["dp_y"],
        facecolor=c, edgecolor="none", s=s
    )
mm = 10 #np.min(ax[0].get_xlim() + ax[0].get_ylim())
m = 0 #np.max(ax[0].get_xlim() + ax[0].get_ylim())
ax[0].plot([mm, m], [mm, m], "grey", linestyle="--")
# ax[0].set_title("A1")
ax[0].set_xlim((m, mm))
ax[0].set_ylim((m, mm))

for cat, c in zip(categories, colors):
    ax[1].scatter(
        df[(df["class"]==cat) & (df.area=="PEG")].drop(columns=["class"]).groupby(by=["site", "area"]).mean()["dp_x"],
        df[(df["class"]==cat) & (df.area=="PEG")].drop(columns=["class"]).groupby(by=["site", "area"]).mean()["dp_y"],
        facecolor=c, edgecolor="none", s=s
    )
mm = 8 #np.min(ax[1].get_xlim() + ax[1].get_ylim())
m = 0 #np.max(ax[1].get_xlim() + ax[1].get_ylim())
ax[1].plot([mm, m], [mm, m], "grey", linestyle="--")
# ax[1].set_title("PEG")
ax[1].set_xlim((m, mm))
ax[1].set_ylim((m, mm))

# for a in ax:
#     a.set_ylabel(r"Active $d'$")
#     a.set_xlabel(r"Passive $d'$")

f.tight_layout()

## Delta dprime strip plot
s = 10
delta_metric = "delta"

f, ax = plt.subplots(1, 2, figsize=(2, 2), sharey=True)

xx = np.random.normal(0, 0.03, len(df[(df.area=="A1") & (df["class"]=="tar_cat")].site.unique()))
for i, (cat, c) in enumerate(zip(categories, colors)):
    # A1
    try:
        ax[0].scatter(
            xx,
            df[(df["class"]==cat) & (df.area=="A1")].drop(columns=["class"]).groupby(by=["site", "area"]).mean()[delta_metric],
            s=s, c=c, edgecolor="none"
        )
        u = df[(df["class"]==cat) & (df.area=="A1")].drop(columns=["class"]).groupby(by=["site", "area"]).mean()[delta_metric].mean()
        yerr = df[(df["class"]==cat) & (df.area=="A1")].drop(columns=["class"]).groupby(by=["site", "area"]).mean()[delta_metric].std() / np.sqrt(len(xx))
        ax[0].errorbar(i, u, yerr=yerr, marker="o", 
                    capsize=2, lw=1, markerfacecolor=c, markeredgecolor="k", color="k")     
        xx += 1
    except ValueError:
        xx += 1
        print(f"error with category: {cat}")

# for site in df[df.area=="A1"].site.unique():
#     x2 = df[(df.site==site) & (df["class"]=="tar_tar")][delta_metric].mean()
#     x3 = df[(df.site==site) & (df["class"]=="tar_cat")][delta_metric].mean()
#     x1 = df[(df.site==site) & (df["class"]=="ref_ref")][delta_metric].mean()
#     ax[0].plot([0.2, 0.8], [x1, x2], color="k", lw=0.5)
#     ax[0].plot([1.2, 1.8], [x2, x3], color="k", lw=0.5)

ax[0].set_title("A1")
# PEG
xx = np.random.normal(0, 0.03, len(df[(df.area=="PEG") & (df["class"]=="tar_cat")].site.unique()))
for i, (cat, c) in enumerate(zip(categories, colors)):
    try:
        ax[1].scatter(
            xx,
            df[(df["class"]==cat) & (df.area=="PEG")].drop(columns=["class"]).groupby(by=["site", "area"]).mean()[delta_metric],
            s=s, c=c, edgecolor="none"
        )
        u = df[(df["class"]==cat) & (df.area=="PEG")].drop(columns=["class"]).groupby(by=["site", "area"]).mean()[delta_metric].mean()
        yerr = df[(df["class"]==cat) & (df.area=="PEG")].drop(columns=["class"]).groupby(by=["site", "area"]).mean()[delta_metric].std() / np.sqrt(len(xx))
        ax[1].errorbar(i, u, yerr=yerr, marker="o", 
                    capsize=2, lw=1, markerfacecolor=c, markeredgecolor="k", color="k")     
        xx += 1
    except:
        xx += 1
        print(f"error with category: {cat}")

# for site in df[df.area=="PEG"].site.unique():
#     x2 = df[(df.site==site) & (df["class"]=="tar_tar")][delta_metric].mean()
#     x3 = df[(df.site==site) & (df["class"]=="tar_cat")][delta_metric].mean()
#     x1 = df[(df.site==site) & (df["class"]=="ref_ref")][delta_metric].mean()
#     ax[1].plot([0.2, 0.8], [x1, x2], color="k", lw=0.5)
#     ax[1].plot([1.2, 1.8], [x2, x3], color="k", lw=0.5)

ax[1].set_title("dPEG")

for a in ax:
    a.axhline(0, linestyle="--", color="k")
    a.set_xticks([])

f.tight_layout()


# Do pairwise stats
for cc in list(combinations(categories, 2)):
    pval = ss.wilcoxon(
        df[(df["class"]==cc[0]) & (df.area=="PEG")].drop(columns=["class"]).groupby(by=["site", "area"]).mean()[delta_metric],
        df[(df["class"]==cc[1]) & (df.area=="PEG")].drop(columns=["class"]).groupby(by=["site", "area"]).mean()[delta_metric]
    ).pvalue
    print(f"{cc}, PEG, pval: {pval}")

    pval = ss.wilcoxon(
        df[(df["class"]==cc[0]) & (df.area=="A1")].drop(columns=["class"]).groupby(by=["site", "area"]).mean()[delta_metric],
        df[(df["class"]==cc[1]) & (df.area=="A1")].drop(columns=["class"]).groupby(by=["site", "area"]).mean()[delta_metric]
    ).pvalue
    print(f"{cc}, A1, pval: {pval}")


plt.show() # show plots for interactive Qt backend