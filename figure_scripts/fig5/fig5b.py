"""
Compare first FA loading to the delta mu axis
for each target (vs. catch)
"""
import os
rdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
import sys
sys.path.append(rdir)

from settings import RESULTS_DIR
from helpers.path_helpers import local_results_file

import scipy.stats as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8 

figpath = "/auto/users/hellerc/code/projects/TBP-ms/figure_files/fig5/"

batch = 324
sqrt = True
db = pd.read_csv(os.path.join(RESULTS_DIR, "db.csv"), index_col=0)
sites = db.site
amodel = 'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR'
pmodel = 'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR'

rra = 0
rrp = 0
dfa = pd.DataFrame(columns=["acos_sim", "e2", "e1", "area", "site"])
dfp = pd.DataFrame(columns=["pcos_sim", "e2", "e1", "area", "site"])
for site in sites:
    d = pd.read_pickle(os.path.join(RESULTS_DIR, site, "FA_perstim_PR.pickle"))
    area = db.loc[site, "area"]
    
    # load decoding results
    ares = pd.read_pickle(local_results_file(RESULTS_DIR, site, amodel, "output.pickle"))
    pres = pd.read_pickle(local_results_file(RESULTS_DIR, site, pmodel, "output.pickle"))
    for e in [k for k in d["active"].keys() if 'CAT' in k]:
        afa = d["active"][e]["components_"][0, :]
        pfa = d["passive"][e]["components_"][0, :]
        try:
            tars = ares[(ares["e1"]==e) & (ares["e2"].str.startswith("TAR"))].e2
            for tar in tars:
                mm = (ares["e1"]==e) & (ares["e2"]==tar)
                _du = ares[mm]["dU"].iloc[0]
                dua = (_du / np.linalg.norm(_du)).dot(ares[mm]["dr_loadings"].iloc[0]).squeeze()
                dfa.loc[rra, :]  = [np.abs(afa.dot(dua)), e, ares[mm]["e2"].iloc[0], area, site]
                rra += 1

            tars = ares[(ares["e1"]==e) & (ares["e2"].str.startswith("TAR"))].e2
            for tar in tars:
                mm = (ares["e1"]==e) & (ares["e2"]==tar)
                _dup = ares[mm]["dU"].iloc[0]
                dup = (_dup / np.linalg.norm(_dup)).dot(ares[mm]["dr_loadings"].iloc[0]).squeeze()
                dfp.loc[rrp, :]  = [np.abs(pfa.dot(dup)), e, ares[mm]["e2"].iloc[0], area, site]
                rrp += 1

        except IndexError:
            print(f"didn't find matching decoding entry for {e}, {site}")
    
# merge 
df = dfa.merge(dfp, on=["e1", "e2", "area", "site"])


f, ax = plt.subplots(1, 2, figsize=(2, 2), sharey=True)

for i, a in enumerate(["A1", "PEG"]):
    y = y = df[df.area==a]["pcos_sim"]
    ax[i].errorbar(0, y.mean(), yerr=y.std()/np.sqrt(len(y)), marker="o",
            capsize=2, markeredgecolor="k", label="passive") 
    y = df[df.area==a]["acos_sim"]
    ax[i].errorbar(1, y.mean(), yerr=y.std()/np.sqrt(len(y)), marker="o",
            capsize=2, markeredgecolor="k", label="active") 
    # ax[i].set_title(a)
    ax[i].set_xticks([])
    ax[i].set_xlim((-0.25, 1.25))
# ax[0].set_ylabel("Cos. similarity (dU vs. FA1)")
# ax[0].legend(frameon=False)
f.tight_layout()

# pvalues
a1pval = ss.wilcoxon(df[df.area=="A1"]["acos_sim"], df[df.area=="A1"]["pcos_sim"])
pegpval = ss.wilcoxon(df[df.area=="PEG"]["acos_sim"], df[df.area=="PEG"]["pcos_sim"])
print(f"a1 alignement pval: {a1pval.pvalue}")
print(f"peg alignement pval: {pegpval.pvalue}")