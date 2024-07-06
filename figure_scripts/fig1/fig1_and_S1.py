"""
Summary of behavior performance across all training (on real task)
for each animal

RT plots for each animal (for a supplmental)
d' performance across all animals as summary
"""
# set up python path to access helper functions
import os
rdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
import sys
sys.path.append(rdir)

from settings import RESULTS_DIR
import json
import helpers.tin_helpers as thelp
from helpers.plotting import plot_RT_histogram
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8
mpl.rcParams['xtick.labelsize'] = 8 
mpl.rcParams['ytick.labelsize'] = 8 

results_path = os.path.join(RESULTS_DIR, "behavior_training")

animals = [
    "Armillaria",
    "Cordyceps",
    "Jellybaby",
    "Clathrus"
]

# plot RT histogram for each animal independently
# save d-prime / DI per session for summary psychometric
dp = {}
di = {}
for an in animals:
    results = json.load(open( os.path.join(results_path, f"{an}_training.json"), 'r' ) )

    # make a "dummy" freq for builing targets list
    rts = results["RTs"]
    dprime = results["dprime"]
    DI = results["DI"]
    nsessions = results["n"]
    keep = [k for k, v in nsessions.items() if (v > 10) | (k == '-inf')]
    rts = {k: v for k, v in rts.items() if k in keep}
    dprime = {k: v for k, v in dprime.items() if k in keep}
    DI = {k: v for k, v in DI.items() if k in keep}
    nsessions = {k: v for k, v in nsessions.items() if k in keep}

    targets = [k for k in rts.keys()]
    targets = [("300+"+t+"+Noise").replace("inf", "Inf") for t in targets]
    cat = targets[0]
    BwG, gR = thelp.make_tbp_colormaps([cat], targets, use_tar_freq_idx=0)
    legend = [s+ f" dB, n={nsessions[s]}, d'={round(np.mean(dprime[s]), 3)}" if '-inf' not in s else 'Catch' for s in rts.keys()]

    f, ax = plt.subplots(1, 1, figsize=(2, 2))
    bins = np.arange(0, 1.4, 0.001)
    plot_RT_histogram(rts, bins=bins, ax=ax, cmap=gR, lw=2, legend=legend)
    ax.set_title(an)
    f.tight_layout()

    # save di / dprime
    di[an] = DI
    dp[an] = dprime


# plot mean / se for each animal
kk = ["-10", "-5", "0", "inf"]
xx = [0, 1, 2, 3]
all_an = np.zeros((len(animals), len(xx)))
f, ax = plt.subplots(1, 1, figsize=(2, 2))

for j, an in enumerate(animals[::-1]):
    dprime = dp[an]
    yy = np.zeros(len(xx))
    yye = np.zeros(len(xx))
    for (i, k) in enumerate(kk):
        if k in dprime.keys():
            yy[i] = np.mean(dprime[k])
            yye[i] = np.std(dprime[k]) / np.sqrt(len(dprime[k]))
        else:
            yy[i] = np.nan
            yye[i] = np.nan

    all_an[j, :] = yy
    ff = np.isnan(yy)==False
    ax.plot(np.array(xx)[ff], yy[ff], "-", lw=1)

all_mean = np.nanmean(all_an, axis=0)
all_sem = np.nanstd(all_an, axis=0) / np.sqrt((np.isnan(all_an)==False).sum(axis=0))
ax.errorbar(np.array(xx), all_mean, yerr=all_sem, 
             capsize=4, marker="o", markersize=5, color="k", lw=2)
ax.axhline(0, linestyle="--", color="grey")
ax.set_xticks(xx)
ax.set_xticklabels(kk)
ax.set_xlabel("Target SNR (dB)")
ax.set_ylabel(r"Performance ($d'$)")
# ax.legend(frameon=False, bbox_to_anchor=(0, 1), loc="upper left")

f.tight_layout()
