"""
Compare behavioral vs. neural dprimes
"""
import os
rdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
import sys
sys.path.append(rdir)

from helpers.path_helpers import local_results_file
from settings import RESULTS_DIR
import scipy.stats as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 10

sqrt = True
db = pd.read_csv(os.path.join(RESULTS_DIR, "db.csv"), index_col=0)
sites = db.site

# load neural dprime
amodel = 'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'
pmodel = 'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'
active = []
passive = []
for site in sites:
    try:
        area = db.loc[site, "area"]
        ares = pd.read_pickle(local_results_file(RESULTS_DIR, site, amodel, "output.pickle"))
        pres = pd.read_pickle(local_results_file(RESULTS_DIR, site, pmodel, "output.pickle"))
        ares["site"] = site
        pres["site"] = site
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
df["delta"] = (df["dp_y"] - df["dp_x"]) / (df["dp_y"] + df["dp_x"])
df["delta_raw"] = df["dp_y"] - df["dp_x"]
df = df.loc[:, ["class", "area", "site", "dp_x", "dp_y", "delta", "delta_raw", "e1", "e2"]] # keep only needed, numeric types

# load behavioral dprimes
beh_df = pd.read_pickle(os.path.join(RESULTS_DIR, "behavior_recording", "all_trials.pickle"))

# Plot relationship between behavior and neural dprime
bg = beh_df.drop(columns=["e2"]).groupby(by=["site", "e1"]).mean()
ng_peg = df[(df["class"]=="tar_cat") & (df.area=="PEG")].drop(columns=["class", "area", "e1"]).groupby(by=["site", "e2"]).mean()
ng_peg.index.set_names("e1", level=1, inplace=True)
ng_peg.index = ng_peg.index.set_levels(ng_peg.index.levels[1].str.strip("TAR_"), level=1)
ng_a1 = df[(df["class"]=="tar_cat") & (df.area=="A1")].drop(columns=["class", "area", "e1"]).groupby(by=["site", "e2"]).mean()
ng_a1.index.set_names("e1", level=1, inplace=True)
ng_a1.index = ng_a1.index.set_levels(ng_a1.index.levels[1].str.strip("TAR_"), level=1)

peg_merge = ng_peg.merge(bg, right_index=True, left_index=True)
a1_merge = ng_a1.merge(bg, right_index=True, left_index=True)

delta_metric = "delta_raw"
nboots = 500
s = 10
delta_ylim = (-1, 3.5)
abs_ylim = (-0.1, 5)
colors = ["grey", "k"]
f, ax = plt.subplots(1, 2, figsize=(4, 2))

for i, (df, c) in enumerate(zip([a1_merge, peg_merge], colors)):

    x = df["dprime"]
    xp = np.linspace(np.min(x), np.max(x), 100)
    # delta dprime
    r, p = ss.pearsonr(x, df[delta_metric])
    leg = f"r={round(r, 3)}, p={round(p, 3)}"
    ax[i].scatter(x, df[delta_metric], 
                    s=s, c=c, edgecolor="none", lw=0)
    # ax[i].set_title(f"{leg}")

    # get line of best fit
    z = np.polyfit(x.values.astype(np.float32), df[delta_metric].values, 1)
    # plot line of best fit
    p_y = z[1] + z[0] * xp
    ax[i].plot(xp, p_y, lw=2, color=c)
    
    # bootstrap condifence interval
    boot_preds = []
    for bb in range(nboots):
        ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
        zb = np.polyfit(x.iloc[ii].values.astype(np.float32), df[delta_metric].iloc[ii], 1)
        p_yb = zb[1] + zb[0] * xp
        boot_preds.append(p_yb)
    bse = np.stack(boot_preds).std(axis=0)
    lower = p_y - bse
    upper = p_y + bse
    ax[i].fill_between(xp, lower, upper, color=c, alpha=0.5, lw=0)

    ax[i].set_ylim(delta_ylim)

f.tight_layout()

# quantify significance of correlation using bootstrapping
np.random.seed(123)
nboots = 1000
x = a1_merge["dprime"]
rb_a1 = []
for bb in range(nboots):
    ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    rb_a1.append(np.corrcoef(x.iloc[ii].values.astype(np.float32), a1_merge[delta_metric].iloc[ii])[0, 1])
x = peg_merge["dprime"]
rb_peg = []
for bb in range(nboots):
    ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    rb_peg.append(np.corrcoef(x.iloc[ii].values.astype(np.float32), peg_merge[delta_metric].iloc[ii])[0, 1])

# compute bootstrapped p-values
np.random.seed(123)
nboots = 1000
x = a1_merge["dprime"]
rb_a1_null = []
for bb in range(nboots):
    ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    jj = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    rb_a1_null.append(np.corrcoef(x.iloc[ii].values.astype(np.float32), a1_merge[delta_metric].iloc[jj])[0, 1])
x = peg_merge["dprime"]
rb_peg_null = []
for bb in range(nboots):
    ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    jj = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    rb_peg_null.append(np.corrcoef(x.iloc[ii].values.astype(np.float32), peg_merge[delta_metric].iloc[jj])[0, 1])
a1_pval = np.mean(np.array(rb_a1_null) > np.corrcoef(a1_merge["dprime"].values.astype(np.float32), a1_merge[delta_metric].values.astype(np.float32))[0, 1])
peg_pval = np.mean(np.array(rb_peg_null) > np.corrcoef(peg_merge["dprime"].values.astype(np.float32), peg_merge[delta_metric].values.astype(np.float32))[0, 1])
print(f"A1 pval: {a1_pval}")
print(f"PEG pval: {peg_pval}")

f, ax = plt.subplots(1, 1, figsize=(1, 2))

lower = np.quantile(rb_a1, 0.025)
upper = np.quantile(rb_a1, 0.975)
ax.plot([0, 0], [lower, upper], color="grey", zorder=-1)
ax.scatter([0], [np.mean(rb_a1)], s=50, edgecolor="k", c="grey")

lower = np.quantile(rb_peg, 0.025)
upper = np.quantile(rb_peg, 0.975)
ax.plot([1, 1], [lower, upper], color="k")
ax.scatter([1], [np.mean(rb_peg)], s=50, edgecolor="k", c="k")

ax.axhline(0, linestyle="--", color="grey")

ax.set_xlim((-0.1, 1.1))
ax.set_xticks([])