"""
Quick analysis to calculate d-prime between each target and catch
for each neuron. Save in one big dataframe.
"""
import os
rdir = os.path.dirname(os.path.dirname(__file__))
import sys
sys.path.append(rdir)

from nems0 import recording

from itertools import combinations
from dDR.utils.decoding import compute_dprime
import helpers.tin_helpers as thelp
from settings import RESULTS_DIR
import numpy as np
import pandas as pd

min_trials = 5
n_resamples = 100
db = pd.read_csv(os.path.join(RESULTS_DIR, "db.csv"), index_col=0)
sites = db.site
fs = 10
amask = ["HIT_TRIAL", "CORRECT_REJECT_TRIAL"]
pmask = ["PASSIVE_EXPERIMENT"]

dfs = []
for (i, site) in enumerate(sites):
    print(f"\n{site}, {i}/{len(sites)}")

    area = db.loc[site, "area"]
    uri = os.path.join(RESULTS_DIR, "recordings", db.loc[site, "10hz_uri"])
    rec = recording.load_recording(uri)
    rec['resp'] = rec['resp'].rasterize()
    rec = rec.create_mask(True)
    arec = rec.and_mask(amask)
    prec = rec.and_mask(pmask)

    ref, tars, _ = thelp.get_sound_labels(arec)

    cc = list(combinations(tars, 2))

    for c in cc:
        try:
            ar1 = arec["resp"].extract_epoch(c[0], mask=arec["mask"])[:, :, 1:4].mean(axis=-1)
            ar2 = arec["resp"].extract_epoch(c[1], mask=arec["mask"])[:, :, 1:4].mean(axis=-1)
            pr1 = prec["resp"].extract_epoch(c[0], mask=prec["mask"])[:, :, 1:4].mean(axis=-1)
            pr2 = prec["resp"].extract_epoch(c[1], mask=prec["mask"])[:, :, 1:4].mean(axis=-1)

            # check that all responses meet minimum trial criteria
            if (ar1.shape[1]>=min_trials) & (ar2.shape[1]>=min_trials) & (pr1.shape[1]>=min_trials) & (pr2.shape[1]>=min_trials):
                # compute dprime for each neuron 
                dp = np.zeros((ar1.shape[1], 2))
                cellid = []
                e1 = []
                e2 = []
                cat = []
                null_low_ci = []
                null_high_ci = []
                null_ahigh_ci = []
                null_phigh_ci = []
                for n in range(ar1.shape[1]):
                    adprime = np.sqrt(abs(compute_dprime(ar1[:, [n]].T, ar2[:, [n]].T)))
                    pdprime = np.sqrt(abs(compute_dprime(pr1[:, [n]].T, pr2[:, [n]].T)))
                    cid = arec["resp"].chans[n]

                    # resample to get a "null" distribution of active / passive dprimes
                    # basic idea is, randomly assign a label to each response (active / passive)
                    # then recompute dprime to define the null distribution for that stimulus pair
                    ndraw1 = np.min([ar1.shape[0], pr1.shape[0]])
                    ndraw2 = np.min([ar2.shape[0], pr2.shape[0]])
                    adprime_null = np.zeros(n_resamples)
                    pdprime_null = np.zeros(n_resamples)
                    for rs in range(n_resamples):
                        ss1_all = np.concatenate(
                                [
                                    ar1[np.random.choice(np.arange(ar1.shape[0]), ndraw1, replace=True), [n]],
                                    pr1[np.random.choice(np.arange(pr1.shape[0]), ndraw1, replace=True), [n]]
                                ], axis=0
                        )
                        ss2_all = np.concatenate(
                                [
                                    ar2[np.random.choice(np.arange(ar2.shape[0]), ndraw2, replace=True), [n]],
                                    pr2[np.random.choice(np.arange(pr2.shape[0]), ndraw2, replace=True), [n]]
                                ], axis=0
                        )
                        np.random.shuffle(ss1_all); np.random.shuffle(ss2_all)
                        adprime_null[rs] = np.sqrt(abs(compute_dprime(ss1_all[:ndraw1, np.newaxis].T, ss2_all[:ndraw2, np.newaxis].T)))
                        pdprime_null[rs] = np.sqrt(abs(compute_dprime(ss1_all[ndraw1:, np.newaxis].T, ss2_all[ndraw2:, np.newaxis].T)))

                    # two sided test
                    gg = np.isfinite(adprime_null) & np.isfinite(pdprime_null)
                    dd = adprime_null[gg]-pdprime_null[gg]
                    null_delta_low_ci = np.quantile(dd, 0.025)
                    null_delta_high_ci = np.quantile(dd, 0.975)

                    # second resampling to test if d-prime itself for this pair is significant
                    # idea here is to randomly assign a stimulus ID to each repetition and
                    # compute a null distribution of dprime. Do it separately for active / passive
                    ndraw_active = np.min([ar1.shape[0], ar2.shape[0]])
                    ndraw_passive = np.min([pr1.shape[0], pr2.shape[0]])
                    adprime_null = np.zeros(n_resamples)
                    pdprime_null = np.zeros(n_resamples)
                    for rs in range(n_resamples):
                        ssActive_all = np.concatenate(
                                [
                                    ar1[np.random.choice(np.arange(ar1.shape[0]), ndraw_active, replace=True), [n]],
                                    ar2[np.random.choice(np.arange(ar2.shape[0]), ndraw_active, replace=True), [n]]
                                ], axis=0
                        )
                        ssPassive_all = np.concatenate(
                                [
                                    pr1[np.random.choice(np.arange(pr1.shape[0]), ndraw_passive, replace=True), [n]],
                                    pr2[np.random.choice(np.arange(pr2.shape[0]), ndraw_passive, replace=True), [n]]
                                ], axis=0
                        )
                        np.random.shuffle(ssActive_all); np.random.shuffle(ssPassive_all)
                        adprime_null[rs] = np.sqrt(abs(compute_dprime(ssActive_all[:ndraw_active, np.newaxis].T, ssActive_all[ndraw_active:, np.newaxis].T)))
                        pdprime_null[rs] = np.sqrt(abs(compute_dprime(ssPassive_all[:ndraw_passive, np.newaxis].T, ssPassive_all[ndraw_passive:, np.newaxis].T)))

                    # one sided test
                    null_active_high = np.quantile(adprime_null[np.isfinite(adprime_null)], 0.95)
                    null_passive_high = np.quantile(pdprime_null[np.isfinite(pdprime_null)], 0.95)

                    if ("TAR" in c[0]) & ("TAR" in c[1]):
                        category = "tar_tar"
                    elif ("CAT" in c[0]) & ("CAT" in c[1]):
                        category = "cat_cat"
                    else:
                        category = "tar_cat"

                    dp[n, :] = [adprime, pdprime]
                    cellid.append(cid)
                    e1.append(c[0])
                    e2.append(c[1])
                    cat.append(category)
                    null_low_ci.append(null_delta_low_ci)
                    null_high_ci.append(null_delta_high_ci)
                    null_ahigh_ci.append(null_active_high)
                    null_phigh_ci.append(null_passive_high)

                df = pd.DataFrame(data=dp, columns=["active", "passive"])
                df["cellid"]=cellid; df["e1"]=e1; df["e2"]=e2; df["category"]=cat;df["area"]=area
                df["null_delta_low_ci"] = null_low_ci
                df["null_delta_high_ci"] = null_high_ci
                df["null_active_high_ci"] = null_ahigh_ci
                df["null_passive_high_ci"] = null_phigh_ci
                df["site"] = site
                dfs.append(df)
        except:
            print(f"{c} didn't have matching epochs between passive/active")

df = pd.concat(dfs)
df.to_csv(os.path.join(RESULTS_DIR, "singleNeuronDprime.csv"))