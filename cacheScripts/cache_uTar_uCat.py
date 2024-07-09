"""
Compute (pre-stim normalized) response for each neuron / target.
Mean of response in evoked window.

Cache results to load for figure 2.
"""
import os
rdir = os.path.dirname(os.path.dirname(__file__))
import sys
sys.path.append(rdir)

from nems.tools import recording

import helpers.tin_helpers as thelp
from settings import RESULTS_DIR
import scipy.stats as ss
import numpy as np
import pandas as pd

db = pd.read_csv(os.path.join(RESULTS_DIR, "db.csv"), index_col=0)
sites = db.site
fs = 50
amask = ["HIT_TRIAL", "CORRECT_REJECT_TRIAL"]
pmask = ["PASSIVE_EXPERIMENT"]

dfs = []
for site in sites:
    area = db.loc[site, "area"]
    uri = os.path.join(RESULTS_DIR, "recordings", db.loc[site, "50hz_uri"])
    rec = recording.load_recording(uri)
    rec['resp'] = rec['resp'].rasterize()
    rec = rec.create_mask(True)
    arec = rec.and_mask(amask)
    prec = rec.and_mask(pmask)

    ref, tars, _ = thelp.get_sound_labels(arec)

    for tar in tars:
        try:
            atresp = rec["resp"].extract_epoch(str(tar), mask=arec["mask"])
            ptresp = rec["resp"].extract_epoch(str(tar), mask=prec["mask"])

            # statistical test across trials
            pval = np.zeros(atresp.shape[1])
            for n in range(atresp.shape[1]):
                pval[n] = ss.ranksums(atresp[:, n, int(0.1*fs):int(0.4*fs)].mean(axis=-1), ptresp[:, n, int(0.1*fs):int(0.4*fs)].mean(axis=-1)).pvalue

            at_psth = atresp.mean(axis=0)[:, :int(0.5 * fs)]
            pt_psth = ptresp.mean(axis=0)[:, :int(0.5 * fs)]

            # normalize to prestim baseline
            at_psth = ((at_psth.T - at_psth[:, :int(0.1*fs)].mean(axis=1))).T
            pt_psth = ((pt_psth.T - pt_psth[:, :int(0.1*fs)].mean(axis=1))).T

            # get mean evoked response
            dur_s = 0.3
            aresp = np.sum(at_psth[:, int(0.1*fs):int(0.4*fs)], axis=1) * dur_s
            presp = np.sum(pt_psth[:, int(0.1*fs):int(0.4*fs)], axis=1) * dur_s

            df = pd.DataFrame(index=rec["resp"].chans,
                                data=np.vstack([aresp, presp, pval, [tar] * at_psth.shape[0], 
                                                [site] * at_psth.shape[0],
                                                [area] * at_psth.shape[0]]).T,
                                columns=["active", "passive", "pval", "epoch", "site", "area"])
            dfs.append(df)
        except:
            print(f"didn't find epoch: {tar} for at least one state (active or passive)")

df = pd.concat(dfs)
df["snr"] = [v[1] for v in df["epoch"].str.split("+").values]
df = df.astype({
    "active": float,
    "passive": float,
    "pval": float,
    "snr": object,
    "area": object,
    "epoch": object,
    "site": object
})
df["cellid"] = df.index

df.to_csv(os.path.join(RESULTS_DIR, "tar_vs_cat.csv"))
