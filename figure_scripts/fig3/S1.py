"""
Loop over datasets.
For each, load pupil.
Split pupil into trial types (HIT, CORRECT_REJECT, MISS, FALSE ALARM vs. PASSIVE)

Create supplemental figure that shows
    1. baseline pupil reflects behavior (so impulsivity, basically)
    1.2. baseline pupil shows inverted U (incorrect trials have either bigger or smaller pupil)
    2. delta pupil reflects reward (bigger delta for rewarded - correct, trials)

"""
import os
rdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
import sys
sys.path.append(rdir)

from nems.tools import recording

from settings import RESULTS_DIR
from itertools import combinations
import pandas as pd
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

db = pd.read_csv(os.path.join(RESULTS_DIR, "db.csv"), index_col=0)
sites = db.site

# chop off trials at 4 seconds
active_trials = []
passive_trials = []
fa_trials = []
hit_trials = []
miss_trials = []
cr_trials = []
for (i, site) in enumerate(sites):
    uri = os.path.join(RESULTS_DIR, "recordings", db.loc[site, "10hz_uri"])
    rec = recording.load_recording(uri)
    rec['resp'] = rec['resp'].rasterize()

    max_pupil = rec["pupil"]._data.max()

    # get active pupil per trial
    ra = rec.copy()
    ra = ra.create_mask(True)
    ra = ra.and_mask(["CORRECT_REJECT_TRIAL", "FALSE_ALARM_TRIAL" "HIT_TRIAL", "MISS_TRIAL"])
    pa_trial = ra["pupil"].extract_epochs("TRIAL", mask=ra["mask"])
    active_trials.append(pa_trial["TRIAL"][:, 0, :40] / max_pupil)

    # get passive pupil per trial
    rp = rec.copy()
    rp = rp.create_mask(True)
    rp = rp.and_mask(["PASSIVE_EXPERIMENT"])
    pp_trial = rec["pupil"].extract_epochs("TRIAL", mask=rp["mask"])
    passive_trials.append(pp_trial["TRIAL"][:, 0, :40] / max_pupil)

    # Do same for specific behavioral outcomes
    r = rec.copy()
    r = r.create_mask(True)
    r = r.and_mask(["HIT_TRIAL"])
    hit_trial = rec["pupil"].extract_epochs("TRIAL", mask=r["mask"])
    hit_trials.append(hit_trial["TRIAL"][:, 0, :40] / max_pupil)

    r = rec.copy()
    r = r.create_mask(True)
    r = r.and_mask(["MISS_TRIAL"])
    miss_trial = rec["pupil"].extract_epochs("TRIAL", mask=r["mask"])
    miss_trials.append(miss_trial["TRIAL"][:, 0, :40] / max_pupil)

    r = rec.copy()
    r = r.create_mask(True)
    r = r.and_mask(["CORRECT_REJECT_TRIAL"])
    cr_trial = rec["pupil"].extract_epochs("TRIAL", mask=r["mask"])
    cr_trials.append(cr_trial["TRIAL"][:, 0, :40] / max_pupil)

    r = rec.copy()
    r = r.create_mask(True)
    r = r.and_mask(["FALSE_ALARM_TRIAL"])
    fa_trial = rec["pupil"].extract_epochs("TRIAL", mask=r["mask"])
    fa_trials.append(fa_trial["TRIAL"][:, 0, :40] / max_pupil)

# concatenate means and trials across recording sites
active_mean = np.concatenate([a.mean(axis=0, keepdims=True) for a in active_trials], axis=0)
active_reps = [s.shape[0] for s in active_trials] 
active_weight =  1 / np.array(active_reps) # weight each experiment equally
active_weight = np.concatenate([np.repeat(p, active_reps[i]) for i, p in enumerate(active_weight)])
active_trials = np.concatenate(active_trials, axis=0)

passive_mean = np.concatenate([a.mean(axis=0, keepdims=True) for a in passive_trials], axis=0)
passive_reps = [s.shape[0] for s in passive_trials] 
passive_weight =  1 / np.array(passive_reps) # weight each experiment equally
passive_weight = np.concatenate([np.repeat(p, passive_reps[i]) for i, p in enumerate(passive_weight)])
passive_trials = np.concatenate(passive_trials, axis=0)


hit_mean = np.concatenate([a.mean(axis=0, keepdims=True) for a in hit_trials], axis=0)
hit_reps = [s.shape[0] for s in hit_trials] 
hit_weight = 1 / np.array(hit_reps) # weight each experiment equally
hit_weight = np.concatenate([np.repeat(p, hit_reps[i]) for i, p in enumerate(hit_weight)])
hit_trials = np.concatenate(hit_trials, axis=0)

miss_mean = np.concatenate([a.mean(axis=0, keepdims=True) for a in miss_trials], axis=0)
miss_reps = [s.shape[0] for s in miss_trials] 
miss_weight = 1 / np.array(miss_reps) # weight each experiment equally
miss_weight = np.concatenate([np.repeat(p, miss_reps[i]) for i, p in enumerate(miss_weight)])
miss_trials = np.concatenate(miss_trials, axis=0)

fa_mean = np.concatenate([a.mean(axis=0, keepdims=True) for a in fa_trials], axis=0)
fa_reps = [s.shape[0] for s in fa_trials] 
fa_weight = 1 / np.array(fa_reps) # weight each experiment equally
fa_weight = np.concatenate([np.repeat(p, fa_reps[i]) for i, p in enumerate(fa_weight)])
fa_trials = np.concatenate(fa_trials, axis=0)

cr_mean = np.concatenate([a.mean(axis=0, keepdims=True) for a in cr_trials], axis=0)
cr_reps = [s.shape[0] for s in cr_trials] 
cr_weight = 1 / np.array(cr_reps) # weight each experiment equally
cr_weight = np.concatenate([np.repeat(p, cr_reps[i]) for i, p in enumerate(cr_weight)])
cr_trials = np.concatenate(cr_trials, axis=0)


# summary histogram of active vs. passive pupil (to make general arousal point)

# Split active by behavioral output
bw = (0, 2)
f, ax = plt.subplots(2, 3, figsize=(7, 4.5))

# active vs. passive pupil
counts, bins = np.histogram(passive_trials.flatten(), bins=np.arange(0, 1.05, step=0.05))
counts = counts / counts.sum()
ax[0, 0].hist(bins[:-1], bins, weights=counts, histtype="step", lw=2, label="passive")
counts, bins = np.histogram(active_trials.flatten(), bins=np.arange(0, 1, step=0.05))
counts = counts / counts.sum()
ax[0, 0].hist(bins[:-1], bins, weights=counts, histtype="step", lw=2, label="active")
ax[0, 0].set_ylabel("Fraction")
ax[0, 0].set_xlabel(r"Pupil size (max$^{-1}$)")
ax[0, 0].legend(frameon=False, bbox_to_anchor=(1, 1), loc="lower right")

yy0 = passive_mean[:, bw[0]:bw[1]].mean(axis=1)
yy1 = active_mean[:, bw[0]:bw[1]].mean(axis=1)
ax[1, 0].errorbar(0, yy0.mean(), yerr=yy0.std()/np.sqrt(len(yy0)),
                    marker="o", capsize=4, lw=2)
ax[1, 0].errorbar(1, yy1.mean(), yerr=yy1.std()/np.sqrt(len(yy0)),
                    marker="o", capsize=4, lw=2)
ax[1, 0].set_xlim((-1, 3))
ax[1, 0].set_ylabel(r"Pupil size (max$^{-1}$)")
ax[1, 0].set_xticks([])

stat, pval = ss.wilcoxon(yy0, yy1)
print(f"Active vs. passive: p = {pval}, stat: {stat}")

# behavior pupil
t = np.linspace(0, 4, 40)
keys = ["hit", "correct reject", "miss", "false alarm"]
colors = ["darkblue", "cornflowerblue", "lightcoral", "firebrick"] # reds for incorrect, blues for correct
data_all = [hit_trials, cr_trials, miss_trials, fa_trials]
data = [hit_mean, cr_mean, miss_mean, fa_mean]
weights = [hit_weight, cr_weight, miss_weight, fa_weight]
for i, (kk, mm, mmt, mmw, col) in enumerate(zip(keys, data, data_all, weights, colors)):

    # plot in time
    u = np.nanmean(mm, axis=0)
    ma = np.ma.MaskedArray(mmt, mask=np.isnan(mmt))
    # u = np.average(ma, axis=0, weights=mmw)
    # sem = np.nanstd(mmt, axis=0) / np.sqrt(mmt.shape[0])
    var = np.average((ma - u)**2, axis=0, weights=mmw)
    sem = np.sqrt(var) / np.sqrt(mmt.shape[0])
    ax[0, 1].plot(t, u, label=kk, color=col)
    ax[0, 1].fill_between(t, u-sem, u+sem, color=col, alpha=0.3, lw=0)

    # plot baseline summary
    yy = mm[:, bw[0]:bw[1]].mean(axis=1)
    ax[1, 1].errorbar(i, yy.mean(), yerr=yy.std()/np.sqrt(len(yy)),
                        marker="o", capsize=4, lw=2, c=col)

    mm = (mm.T - mm[:, 0]).T
    u = np.nanmean(mm, axis=0)
    mmt = (mmt.T - mmt[:, 0]).T
    ma = np.ma.MaskedArray(mmt, mask=np.isnan(mmt))
    var = np.average((ma - u)**2, axis=0, weights=mmw)
    sem = np.sqrt(var) / np.sqrt(mmt.shape[0])
    # sem = np.nanstd(mmt, axis=0) / np.sqrt(mmt.shape[0])
    ax[0, 2].plot(t, u, label=kk, color=col)
    ax[0, 2].fill_between(t, u-sem, u+sem, color=col, alpha=0.3, lw=0)

    mm = (mm.T - mm[:, 0]).T
    # plot delta summary
    yy = np.nanmax(mm, axis=1)
    ax[1, 2].errorbar(i, yy.mean(), yerr=yy.std()/np.sqrt(len(yy)),
                        marker="o", capsize=4, lw=2, c=col)

ax[0, 1].set_ylabel(r"Pupil size (max$^{-1}$)")
ax[0, 1].set_xlabel("Time (s)")
ax[1, 1].set_ylabel(r"Baseline pupil size (max$^{-1}$)")
ax[1, 1].set_xticks([])

ax[0, 2].set_ylabel(r"Pupil change (max$^{-1}$)")
ax[0, 2].set_xlabel("Time (s)")
ax[1, 2].set_ylabel(r"Max pupil change (max$^{-1}$)")
ax[1, 2].set_xticks([])

ax[0, 2].legend(frameon=False, bbox_to_anchor=(1, 1), loc="lower right")

f.tight_layout()

# pairwise stats
key_combos = list(combinations(keys, 2))
d_combos = list(combinations(data, 2))
for (kk, mm) in zip(key_combos, d_combos):
    # baseline
    pval, stat = ss.wilcoxon(mm[0][:, bw[0]:bw[1]].mean(axis=1), mm[1][:, bw[0]:bw[1]].mean(axis=1))
    print(f"basline {kk[0]} vs. {kk[1]}, pval: {pval}, stat: {stat}")
    # delta
    yy0 = (mm[0].T - mm[0][:, 0]).T
    yy0 = np.nanmax(yy0, axis=1)
    yy1 = (mm[1].T - mm[1][:, 0]).T
    yy1 = np.nanmax(yy1, axis=1)
    pval, stat = ss.wilcoxon(yy0, yy1)
    print(f"delta {kk[0]} vs. {kk[1]}, pval: {pval}, stat: {stat}\n")


plt.show() # show plots for interactive Qt backend