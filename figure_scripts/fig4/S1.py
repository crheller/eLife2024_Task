"""
Summary figure of choice decoding

Show early / late trial choice decoding for hit vs. miss (don't use correct reject for this)
Break up my SNR and by brain region.

Point is that choice decoding does not change signficantly over the course of the trial.
This suggests that "choice" decoder readout is probably more like impulsivity (see pupil results -
we have a per trial baseline difference in pupil between hit and miss trials
"""
import os
rdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
import sys
sys.path.append(rdir)

import matplotlib.pyplot as plt
import pandas as pd
import helpers.tin_helpers as thelp
from helpers.path_helpers import local_results_file
from settings import RESULTS_DIR
import numpy as np
import scipy.stats as ss


target_model = "tbpChoiceDecoding_fs10_decision.h.m_DRops.dim2.ddr"

twindows_early = [
    "ws0.0_we0.1_trial_fromfirst", 
    "ws0.1_we0.2_trial_fromfirst", 
    "ws0.2_we0.3_trial_fromfirst", 
    "ws0.3_we0.4_trial_fromfirst", 
    "ws0.4_we0.5_trial_fromfirst"
]

twindows_late = [
    "ws0.0_we0.1_trial", 
    "ws0.1_we0.2_trial", 
    "ws0.2_we0.3_trial", 
    "ws0.3_we0.4_trial", 
    "ws0.4_we0.5_trial"
]

sqrt = True
db = pd.read_csv(os.path.join(RESULTS_DIR, "db.csv"), index_col=0)
sites = db.site

# TARGET ANALYSIS
df = pd.DataFrame(columns=["dp", "cp", "tstart", "tend", "early", "snr", "area", "site", "shuffle"])
i = 0
for site in sites:
    try:
        for twin in twindows_early+twindows_late:
            # actual results
            model = target_model.replace("Decoding_fs10_", f"Decoding_fs10_{twin}_")
            res = pd.read_pickle(local_results_file(RESULTS_DIR, site, model, "output.pickle"))    
            area = db.loc[site, "area"]
            if sqrt:
                res["dp"] = np.sqrt(res["dp"])

            # loop over results and append to dataframe
            for j in range(res.shape[0]):
                df.loc[i, :] = [
                    res.iloc[j]["dp"],
                    res.iloc[j]["percent_correct"],
                    float(twin.split("_")[0].strip("ws")),
                    float(twin.split("_")[1].strip("we")),
                    "fromfirst" in twin,
                    thelp.get_snrs([res["stimulus"].iloc[j]])[0],
                    area,
                    site,
                    False
                ]   

                i += 1

    except:
        print(f"{twin} model didn't exsit for site {site}. Prob too few reps")

snrs = [-5, 0, np.inf] # don't use -10db, not enough trials
cmap = plt.get_cmap("Reds", 5)

f = plt.figure(figsize=(6, 4))
a1_taxis = plt.subplot2grid((2, 4), (0, 0), colspan=3)
a1_evl = plt.subplot2grid((2, 4), (0, 3), colspan=1)
peg_taxis = plt.subplot2grid((2, 4), (1, 0), colspan=3)
peg_evl = plt.subplot2grid((2, 4), (1, 3), colspan=1)

# plot time decoding, early windows, A1
for row, (axis, area, comp_ax) in enumerate(zip([a1_taxis, peg_taxis], ["A1", "PEG"], [a1_evl, peg_evl])):
       
    # time plots
    early_mask = (df.area==area) & (df.early==True) & (df.shuffle==False)
    for (i, snr) in enumerate(snrs):
        m = early_mask & (df.snr==snr)
        d = df[m][["cp", "tstart"]].groupby("tstart").mean()["cp"]
        axis.plot(d, color=cmap(i+2))

    # plot time decoding, late windows, A1
    late_mask = (df.area==area) & (df.early==False) & (df.shuffle==False)
    for (i, snr) in enumerate(snrs):
        m = late_mask & (df.snr==snr)
        d = df[m][["cp", "tstart"]].groupby("tstart").mean()["cp"]
        axis.plot(d.index+0.7, d, color=cmap(i+2), label=snr)

    axis.axhline(0.5, linestyle="--", color="k")
    axis.axvspan(0.75, 1.05, color="lightgrey", alpha=0.5, lw=0)
    axis.axvline(0.9, linestyle="-", color="k")
    axis.set_ylim((None, 1))
    axis.set_ylabel("Choice prob.")
    axis.set_xlabel("Time (s)")
    axis.legend(frameon=False, bbox_to_anchor=(0, 1), loc="upper left")

    # combine across SNRs and simply test if mean early choice % is different than late
    early_data = df[early_mask & (df.tstart<0.2)][["cp", "site"]].groupby(by=["site"]).mean()
    late_data = df[late_mask & (df.tstart<0.2)][["cp", "site"]].groupby(by=["site"]).mean()

    x = np.stack([
        np.zeros((early_data.shape[0])),
        np.ones((early_data.shape[0]))
    ])
    y = np.stack([
        early_data["cp"].to_numpy(),
        late_data["cp"].to_numpy()
    ])
    comp_ax.plot(x, y, ".-", color="lightgrey", lw=0.5)
    comp_ax.plot(x.mean(axis=1, keepdims=True), y.mean(axis=1, keepdims=True), "o-", color="k")
    comp_ax.axhline(0.5, linestyle="--", color="k")
    comp_ax.set_ylim((None, 1))
    comp_ax.set_xlim((-1, 2))
    comp_ax.set_xticks([])

    # wilcoxon test
    pval, stat = ss.wilcoxon(y[0, :], y[1, :])
    print(f"{area}, early vs. late, pval: {pval}, stat: {stat}")

f.tight_layout()