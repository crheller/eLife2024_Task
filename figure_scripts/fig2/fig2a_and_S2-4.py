"""
Population PSTHs in active / passive and their difference
Create for one example site.
"""
# set up python path to access helper functions
import os
rdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
import sys
sys.path.append(rdir)

from settings import RESULTS_DIR
import pandas as pd
import helpers.tin_helpers as thelp
import scipy.ndimage as sf
import numpy as np
import matplotlib.pyplot as plt

from nems.tools import recording

# site = "CRD018d" -- A1 example site 
# site = "CRD010b" -- PEG example site
db = pd.read_csv(os.path.join(RESULTS_DIR, "db.csv"), index_col=0)
fs = 50
for site in db.site:
        try:

                # get area / uri (50 Hz)
                area = db.loc[site, "area"]
                uri = os.path.join(RESULTS_DIR, "recordings", db.loc[site, "50hz_uri"])

                amask = ["HIT_TRIAL", "CORRECT_REJECT_TRIAL"]
                pmask = ["PASSIVE_EXPERIMENT"]

                rec = recording.load_recording(uri)
                rec['resp'] = rec['resp'].rasterize()
                rec = rec.create_mask(True)
                arec = rec.and_mask(amask)
                prec = rec.and_mask(pmask)

                ref, tars, _ = thelp.get_sound_labels(rec)
                freqs = thelp.get_freqs(tars)
                keepfreq = freqs[-1]
                kk = [k for k in range(len(tars)) if freqs[k]==keepfreq]
                tars = np.array(tars)[np.array(kk)]
                target = str(tars[-1])
                catch = str(tars[0])

                atresp = rec["resp"].extract_epoch(target, mask=arec["mask"])
                ptresp = rec["resp"].extract_epoch(target, mask=prec["mask"])
                acresp = rec["resp"].extract_epoch(catch, mask=arec["mask"])
                pcresp = rec["resp"].extract_epoch(catch, mask=prec["mask"])

                sigma = 1.5
                at_psth = sf.gaussian_filter1d(atresp.mean(axis=0)[:, :int(0.5 * fs)], sigma, axis=1)
                pt_psth = sf.gaussian_filter1d(ptresp.mean(axis=0)[:, :int(0.5 * fs)], sigma, axis=1)
                ac_psth = sf.gaussian_filter1d(acresp.mean(axis=0)[:, :int(0.5 * fs)], sigma, axis=1)
                pc_psth = sf.gaussian_filter1d(pcresp.mean(axis=0)[:, :int(0.5 * fs)], sigma, axis=1)

                # normalize according to all the data
                sd = np.concatenate((at_psth, pt_psth, ac_psth, pc_psth), axis=1).std(axis=1)
                sd[sd==0] = 1
                at_psth = ((at_psth.T - at_psth[:, :int(0.1*fs)].mean(axis=1)) / sd).T
                pt_psth = ((pt_psth.T - pt_psth[:, :int(0.1*fs)].mean(axis=1)) / sd).T
                ac_psth = ((ac_psth.T - ac_psth[:, :int(0.1*fs)].mean(axis=1)) / sd).T
                pc_psth = ((pc_psth.T - pc_psth[:, :int(0.1*fs)].mean(axis=1)) / sd).T

                vmin = -5
                vmax = 5
                cmap = "PuOr_r"
                sidx = np.argsort(at_psth.argmax(axis=1))
                f, ax = plt.subplots(2, 3, figsize=(4, 2.5))

                # ax[0, 0].set_title("Active, Target")
                ii=ax[0, 0].imshow(at_psth[sidx, :], cmap=cmap, vmin=vmin, vmax=vmax,
                        aspect="auto", extent=[-0.1, 0.4, 0, at_psth.shape[0]], interpolation="none")
                f.colorbar(ii, ax=ax[0, 0])
                # ax[0, 1].set_title("Active, Catch")
                ii=ax[0, 1].imshow(ac_psth[sidx, :], cmap=cmap, vmin=vmin, vmax=vmax,
                        aspect="auto", extent=[-0.1, 0.4, 0, at_psth.shape[0]], interpolation="none")
                f.colorbar(ii, ax=ax[0, 1])
                ii=ax[0, 2].imshow(at_psth[sidx, :] - ac_psth[sidx, :], cmap="bwr", vmin=-vmax, vmax=vmax, 
                        aspect="auto", extent=[-0.1, 0.4, 0, at_psth.shape[0]], interpolation="none")
                f.colorbar(ii, ax=ax[0, 2])

                # ax[1, 0].set_title("Pasive, Target")
                ii=ax[1, 0].imshow(pt_psth[sidx, :], cmap=cmap, vmin=vmin, vmax=vmax,
                        aspect="auto", extent=[-0.1, 0.4, 0, at_psth.shape[0]], interpolation="none")
                f.colorbar(ii, ax=ax[1, 0])
                # ax[1, 1].set_title("Passive, Catch")
                ii=ax[1, 1].imshow(pc_psth[sidx, :], cmap=cmap, vmin=vmin, vmax=vmax,
                        aspect="auto", extent=[-0.1, 0.4, 0, at_psth.shape[0]], interpolation="none")
                f.colorbar(ii, ax=ax[1, 1])
                ii=ax[1, 2].imshow(pt_psth[sidx, :] - pc_psth[sidx, :], cmap="bwr", vmin=-vmax, vmax=vmax, 
                        aspect="auto", extent=[-0.1, 0.4, 0, at_psth.shape[0]], interpolation="none")
                f.colorbar(ii, ax=ax[1, 2])

                for i in range(2):
                        for j in range(3):
                                if j != 2:
                                        c = "k"
                                else:
                                        c = "k"

                                ax[i, j].axvline(0.0, linestyle="--", color=c)
                                ax[i, j].axvline(0.3, linestyle="--", color=c)
                                ax[i, j].set_yticks([])
                                ax[i, j].set_xticks([])
                                # ax[i, j].set_xlabel("Time (s)")

                f.tight_layout()

        except:
                print(f"failed for {site}. Means that behavior probably was not great for this dataset and didn't have enough trials of a certain stimulus")

plt.show() # show plots for interactive Qt backend