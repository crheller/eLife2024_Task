"""
2x2
top row, mean tar vs. cat response for active/passive, each neuron
bottom row:
    active vs. passive d' scatter plot
    delta dprime histogram
"""
import os
rdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
import sys
sys.path.append(rdir)

from settings import RESULTS_DIR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_resp = pd.read_csv(os.path.join(RESULTS_DIR, "tar_vs_cat.csv"), index_col=0).drop(columns=["site", "epoch"])
df_dprime = pd.read_csv(os.path.join(RESULTS_DIR, "singleNeuronDprime.csv"), index_col=0)
gg_resp = df_resp.groupby(by=["snr", "cellid", "area"]).mean()
s = 3
alpha = 1


# PEG
ggm = gg_resp[gg_resp.index.get_level_values(2)=="PEG"]
yvals = ggm[ggm.index.get_level_values(0)=="InfdB"]
xvals = ggm[ggm.index.get_level_values(0)=="-InfdB"]
vals = yvals.merge(xvals, on="cellid")

# get dprimes to merge on responses
mask = (df_dprime.category=="tar_cat") & (df_dprime.e2.str.contains("InfdB")) & (df_dprime.area=="PEG")
vals_dprime = df_dprime[mask]
asig = vals_dprime["active"] > vals_dprime["null_active_high_ci"]
psig = vals_dprime["passive"] > vals_dprime["null_passive_high_ci"]
sig = np.vstack([asig, psig])
pval_df = pd.DataFrame(index=vals_dprime["cellid"], columns=["asig", "psig"], data=sig.T)

vals = vals.merge(pval_df, left_index=True, right_index=True)

# mean responses passive
f, ax = plt.subplots(1, 1, figsize=(1, 1))

ax.scatter(
    vals["passive_y"],
    vals["passive_x"],
    s=s, c="grey", alpha=alpha,
    edgecolor="none", rasterized=True
)
ax.scatter(
    vals["passive_y"][vals.psig],
    vals["passive_x"][vals.psig],
    s=s*2, c="k", alpha=alpha,
    edgecolor="none", rasterized=True
)
ax.set_xlim((vals.values.min(), vals.values.max()+0.5))
ax.set_ylim((vals.values.min(), vals.values.max()+0.5))
ax.plot([vals.values.min(), vals.values.max()],
            [vals.values.min(), vals.values.max()], 
            "grey", linestyle="--", zorder=-1)
print(f"{sum(vals.psig)}/{vals.shape[0]} significant, passive, dPEG \n")

# mean responses active
f, ax = plt.subplots(1, 1, figsize=(1, 1))
ax.scatter(
    vals["active_y"],
    vals["active_x"],
    s=s, c="grey", alpha=alpha,
    edgecolor="none", rasterized=False
)
ax.scatter(
    vals["active_y"][vals.asig],
    vals["active_x"][vals.asig],
    s=s*2, c="k", alpha=alpha,
    edgecolor="none", rasterized=True
)
ax.set_xlim((vals.values.min(), vals.values.max()+0.5))
ax.set_ylim((vals.values.min(), vals.values.max()+0.5))
ax.plot([vals.values.min(), vals.values.max()],
            [vals.values.min(), vals.values.max()], 
            "grey", linestyle="--", zorder=-1)
print(f"{sum(vals.asig)}/{vals.shape[0]} significant, active, dPEG \n")

# delta dprime histogram
dd = df_dprime[(df_dprime.area=="PEG") & (df_dprime.category=="tar_cat")]
dd = dd[(dd.e1.str.contains("\+InfdB")) | (dd.e2.str.contains("\+InfdB"))]
f, ax = plt.subplots(1, 1, figsize=(1, 1))

ax.hist(
    dd["active"]-dd["passive"],
    bins=np.arange(-3, 3.2, 0.2),
    facecolor="lightgrey",
    edgecolor="k",
    histtype="stepfilled"
)
sig_bool = ((dd["active"] - dd["passive"]) > dd["null_delta_high_ci"]) | \
     ((dd["active"] - dd["passive"]) < dd["null_delta_low_ci"])
ax.hist(
    (dd["active"]-dd["passive"])[sig_bool],
    bins=np.arange(-3, 3.2, 0.2),
    facecolor="k",
    edgecolor="k",
    histtype="stepfilled"
)
print(f"{sum(sig_bool)}/{len(sig_bool)} significant change in d-prime dPEG \n")


# A1
ggm = gg_resp[gg_resp.index.get_level_values(2)=="A1"]
yvals = ggm[ggm.index.get_level_values(0)=="InfdB"]
xvals = ggm[ggm.index.get_level_values(0)=="-InfdB"]
vals = yvals.merge(xvals, on="cellid")

# get dprimes to merge on responses
mask = (df_dprime.category=="tar_cat") & (df_dprime.e2.str.contains("InfdB")) & (df_dprime.area=="A1")
vals_dprime = df_dprime[mask]
asig = vals_dprime["active"] > vals_dprime["null_active_high_ci"]
psig = vals_dprime["passive"] > vals_dprime["null_passive_high_ci"]
sig = np.vstack([asig, psig])
pval_df = pd.DataFrame(index=vals_dprime["cellid"], columns=["asig", "psig"], data=sig.T)

vals = vals.merge(pval_df, left_index=True, right_index=True)

# mean responses passive
inrange = (vals["passive_y"]>-5) & (vals["passive_y"]<10) & (vals["passive_x"]>-5) & (vals["passive_y"]<10)
f, ax = plt.subplots(1, 1, figsize=(1, 1))

ax.scatter(
    vals["passive_y"][inrange],
    vals["passive_x"][inrange],
    s=s, c="grey", alpha=alpha,
    edgecolor="none", rasterized=False
)
ax.scatter(
    vals["passive_y"][vals.psig & inrange],
    vals["passive_x"][vals.psig & inrange],
    s=s*2, c="k", alpha=alpha,
    edgecolor="none", rasterized=False
)
# ax.set_xlim((vals.values.min(), vals.values.max()))
# ax.set_ylim((vals.values.min(), vals.values.max()))
ax.set_xlim((-5, 10))
ax.set_ylim((-5, 10))
ax.plot([vals.values.min(), vals.values.max()],
            [vals.values.min(), vals.values.max()], 
            "grey", linestyle="--", zorder=-1)
print(f"{sum(vals.psig)}/{vals.shape[0]} significant, passive, A1 \n")

# mean responses active
inrange = (vals["active_y"]>-5) & (vals["active_y"]<10) & (vals["active_x"]>-5) & (vals["active_y"]<10)
f, ax = plt.subplots(1, 1, figsize=(1, 1))
ax.scatter(
    vals["active_y"],
    vals["active_x"],
    s=s, c="grey", alpha=alpha,
    edgecolor="none", rasterized=False
)
ax.scatter(
    vals["active_y"][vals.asig],
    vals["active_x"][vals.asig],
    s=s*2, c="k", alpha=alpha,
    edgecolor="none", rasterized=False
)
# ax.set_xlim((vals.values.min(), vals.values.max()))
# ax.set_ylim((vals.values.min(), vals.values.max()))
ax.set_xlim((-5, 10))
ax.set_ylim((-5, 10))
ax.plot([vals.values.min(), vals.values.max()],
            [vals.values.min(), vals.values.max()], 
            "grey", linestyle="--", zorder=-1)
print(f"{sum(vals.asig)}/{vals.shape[0]} significant, active, A1 \n")

# delta dprime histogram
dd = df_dprime[(df_dprime.area=="A1") & (df_dprime.category=="tar_cat")]
dd = dd[(dd.e1.str.contains("\+InfdB")) | (dd.e2.str.contains("\+InfdB"))]
f, ax = plt.subplots(1, 1, figsize=(1, 1))

ax.hist(
    dd["active"]-dd["passive"],
    bins=np.arange(-3, 3.2, 0.2),
    facecolor="lightgrey",
    edgecolor="k",
    histtype="stepfilled"
)
sig_bool = ((dd["active"] - dd["passive"]) > dd["null_delta_high_ci"]) | \
     ((dd["active"] - dd["passive"]) < dd["null_delta_low_ci"])
ax.hist(
    (dd["active"]-dd["passive"])[sig_bool],
    bins=np.arange(-3, 3.2, 0.2),
    facecolor="k",
    edgecolor="k",
    histtype="stepfilled"
)
print(f"{sum(sig_bool)}/{len(sig_bool)} significant change in d-prime A1 \n")

plt.show() # show plots for interactive Qt backend