"""
Cache FA model and metrics for all sites.

Inspired by Umakanthan 2021 neuron paper -- bridging neuronal correlations / dim reduction
Cache script --  
for each site, calculate:
    - % shared variance
    - loading similarity
    - dimensionality
    - also compute dimensionality / components of full space, including stimuli, using PCA (trial-averaged or raw data?)
        - think raw data is okay, but we'll do both. The point we want to make is that variability coflucates in a space with dim < total dim
"""
import os
rdir = os.path.dirname(os.path.dirname(__file__))
import sys
sys.path.append(rdir)

import helpers.fametrics as fhelp
from settings import RESULTS_DIR
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
import helpers.loaders as loaders
import pickle

import logging 

log = logging.getLogger(__name__)

db = pd.read_csv(os.path.join(RESULTS_DIR, "db.csv"), index_col=0)
sites = db.site

modelname = [
    "FA_perstim",
    "FA_perstim_PR"
]

for site in sites:
    shuffle = False
    perstim = False
    regress_pupil = False
    for op in modelname.split("_"):
        if op == "shuff":
            shuffle = True
        if op == "perstim":
            perstim = True
        if op == "PR":
            regress_pupil = True

    # ============ perform analysis ==================

    # load data
    X_active, _ = loaders.load_tbp_for_decoding(site=site, 
                                        wins = 0.1,
                                        wine = 0.4,
                                        collapse=True,
                                        mask=["HIT_TRIAL", "MISS_TRIAL", "CORRECT_REJECT_TRIAL"],
                                        recache=False,
                                        regresspupil=regress_pupil)
    X_passive, _ = loaders.load_tbp_for_decoding(site=site, 
                                        wins = 0.1,
                                        wine = 0.4,
                                        collapse=True,
                                        mask=["PASSIVE_EXPERIMENT"],
                                        recache=False,
                                        regresspupil=regress_pupil)

    # keep only CAT and TAR stims and turn into a 4D matrix (cells x reps x stim x tbins)
    keep = [t for t in X_active.keys() if (("TAR_" in t) | ("CAT_" in t)) & (t in X_passive.keys())]
    X_active = {k: v for k, v in X_active.items() if k in keep}
    X_passive = {k: v for k, v in X_passive.items() if k in keep}

    # FOR SETTING UP FITTING COMMON SPACE ACROSS ALL STIM
    if perstim == False:
        # fit all stim together, after subtracting psth
        # "special" cross-validation -- fitting individual stims doesn't work, not enough data
        # instead, leave-one-stim out fitting to find dims that are shared / stimulus-independent
        nstim = len(keep)
        nCells = X_active[keep[0]].shape[0]
        X_asub = {k: v - v.mean(axis=1, keepdims=True) for k, v in X_active.items()}
        X_psub = {k: v - v.mean(axis=1, keepdims=True) for k, v in X_passive.items()}
        X_ata = {k: v.mean(axis=1, keepdims=True) for k, v in X_active.items()}
        X_pta = {k: v.mean(axis=1, keepdims=True) for k, v in X_passive.items()}

        if shuffle:
            raise NotImplementedError("Not implemented for active / passive shuffling, yet")

        nfold = nstim
        nComponents = 50
        if nCells < nComponents:
            nComponents = nCells

        log.info("\nComputing log-likelihood across models / nfolds")
        LL_active = np.zeros((nComponents, nfold))
        LL_passive = np.zeros((nComponents, nfold))
        for ii in np.arange(1, LL_active.shape[0]+1):
            log.info(f"{ii} / {LL_active.shape[0]}")
            fa = FactorAnalysis(n_components=ii, random_state=0) # init model
            for nf, kk in enumerate(keep):
                fit_keys = [x for x in keep if x != kk]

                fit_afa = np.concatenate([X_asub[k] for k in fit_keys], axis=1).squeeze()
                fit_pfa = np.concatenate([X_psub[k] for k in fit_keys], axis=1).squeeze()
                eval_afa = X_asub[kk].squeeze()
                eval_pfa = X_psub[kk].squeeze()

                # ACTIVE FACTOR ANALYSIS
                fa.fit(fit_afa.T) # fit model
                # Get LL score
                LL_active[ii-1, nf] = fa.score(eval_afa.T)

                # PASSIVE FACTOR ANALYSIS
                fa.fit(fit_pfa.T) # fit model
                # Get LL score
                LL_passive[ii-1, nf] = fa.score(eval_pfa.T)

        log.info("Estimating %sv and loading similarity for the 'best' model")
        # ACTIVE
        active_dim_sem = np.std([fhelp.get_dim(LL_active[:, i]) for i in range(LL_active.shape[1])]) / np.sqrt(LL_active.shape[1])
        active_dim = fhelp.get_dim(LL_active.mean(axis=-1))
        # fit the "best" model over jackknifes
        a_sv = np.zeros(nfold)
        a_loading_sim = np.zeros(nfold)
        a_dim95 = np.zeros(nfold)
        for nf, kk in enumerate(keep):
            fit_keys = [x for x in keep if x != kk]
            x = np.concatenate([X_asub[k] for k in fit_keys], axis=1).squeeze()
            fa_active = FactorAnalysis(n_components=active_dim, random_state=0) 
            fa_active.fit(x.T)
            a_sv[nf] = fhelp.get_sv(fa_active)
            a_loading_sim[nf] = fhelp.get_loading_similarity(fa_active)
            # get n dims needs to explain 95% of shared variance
            a_dim95[nf] = fhelp.get_dim95(fa_active)

        # PASSIVE
        passive_dim_sem = np.std([fhelp.get_dim(LL_passive[:, i]) for i in range(LL_passive.shape[1])]) / np.sqrt(LL_passive.shape[1])
        passive_dim = fhelp.get_dim(LL_passive.mean(axis=-1))
        # fit the "best" model over jackknifes
        p_sv = np.zeros(nfold)
        p_loading_sim = np.zeros(nfold)
        p_dim95 = np.zeros(nfold)
        for nf, kk in enumerate(keep):
            fit_keys = [x for x in keep if x != kk]
            x = np.concatenate([X_psub[k] for k in fit_keys], axis=1).squeeze()
            fa_passive = FactorAnalysis(n_components=passive_dim, random_state=0) 
            fa_passive.fit(x.T)
            p_sv[nf] = fhelp.get_sv(fa_passive)
            p_loading_sim[nf] = fhelp.get_loading_similarity(fa_passive)
            # get n dims needs to explain 95% of shared variance
            p_dim95[nf] = fhelp.get_dim95(fa_passive)


        # final fit with all data to get components
        fa_active = FactorAnalysis(n_components=active_dim, random_state=0) 
        x = np.concatenate([X_asub[k] for k in keep], axis=1).squeeze()
        fa_active.fit(x.T)
        active_sv_all = fhelp.get_sv(fa_active)
        active_ls_all = fhelp.get_loading_similarity(fa_active)
        active_dim95_all = fhelp.get_dim95(fa_active)

        fa_passive = FactorAnalysis(n_components=passive_dim, random_state=0) 
        x = np.concatenate([X_psub[k] for k in keep], axis=1).squeeze()
        fa_passive.fit(x.T)
        passive_sv_all = fhelp.get_sv(fa_passive)
        passive_ls_all = fhelp.get_loading_similarity(fa_passive)
        passive_dim95_all = fhelp.get_dim95(fa_passive)

        # Save results
        results = {
            "active_sv": a_sv.mean(),
            "passive_sv": p_sv.mean(),
            "active_sv_sd": a_sv.std(),
            "passive_sv_sd": p_sv.std(),
            "active_loading_sim": a_loading_sim.mean(),
            "passive_loading_sim": p_loading_sim.mean(),
            "active_loading_sim_sd": a_loading_sim.std(),
            "passive_loading_sim_sd": p_loading_sim.std(),
            "active_dim95": a_dim95.mean(),
            "passive_dim95": p_dim95.mean(),
            "active_dim95_sd": a_dim95.std(),
            "passive_dim95_sd": p_dim95.std(),
            "active_dim": active_dim,
            "passive_dim": passive_dim,
            "active_dim_sem": active_dim_sem,
            "passive_dim_sem": passive_dim_sem,
            "nCells": nCells,
            "nStim": nstim,
            "final_fit": {
                "fa_active.components_": fa_active.components_,
                "fa_passive.components_": fa_passive.components_,
                "fa_active.sigma_shared": fhelp.sigma_shared(fa_active),
                "fa_passive.sigma_shared": fhelp.sigma_shared(fa_passive),
                "fa_active.sigma_ind": fhelp.sigma_ind(fa_active),
                "fa_passive.sigma_ind": fhelp.sigma_ind(fa_passive),
                "fa_active.sigma_full": fhelp.pred_cov(fa_active),
                "fa_passive.sigma_full": fhelp.pred_cov(fa_passive),
                "active_sv_all": active_sv_all,
                "passive_sv_all": passive_sv_all,
                "active_ls_all": active_ls_all,
                "passive_ls_all": passive_ls_all,
                "active_dim95_all": active_dim95_all,
                "passive_dim95_all": passive_dim95_all
            }
        }


    # FIT EACH STIM INDIVIDUALLY
    else:
        # can't do CV here. Instead, just fit model for each stimulus
        nstim = len(keep)
        nCells = X_active[keep[0]].shape[0]
        X_asub = {k: v - v.mean(axis=1, keepdims=True) for k, v in X_active.items()}
        X_psub = {k: v - v.mean(axis=1, keepdims=True) for k, v in X_passive.items()}
        X_ata = {k: v.mean(axis=1, keepdims=True) for k, v in X_active.items()}
        X_pta = {k: v.mean(axis=1, keepdims=True) for k, v in X_passive.items()}

        if shuffle:
            raise NotImplementedError("Not implemented for active / passive shuffling, yet")

        nfold = nstim
        nComponents = 50
        if nCells < nComponents:
            nComponents = nCells

        log.info("\nComputing log-likelihood across models / stimuli")
        LL_active = np.zeros((nComponents, nstim))
        LL_passive = np.zeros((nComponents, nstim))
        rand_jacks = 10
        for ii in np.arange(1, LL_active.shape[0]+1):
            log.info(f"{ii} / {LL_active.shape[0]}")
            fa = FactorAnalysis(n_components=ii, random_state=0) # init model
            for st, kk in enumerate(keep):
                # ACTIVE FACTOR ANALYSIS
                fa.fit(X_asub[kk].squeeze().T[::2, :]) # fit model
                # Get LL score
                LL_active[ii-1, st] = np.mean(fa.score(X_asub[kk].squeeze().T[1::2, :]))

                # PASSIVE FACTOR ANALYSIS
                fa.fit(X_psub[kk].squeeze().T[::2, :]) # fit model
                # Get LL score
                LL_passive[ii-1, st] = fa.score(X_psub[kk].squeeze().T[1::2, :])

        log.info("Estimating %sv and loading similarity for the 'best' model in each state")

        results = {"active": {}, "passive": {}}

        # ACTIVE
        active_dim = [fhelp.get_dim(LL_active[:, i]) for i in range(LL_active.shape[1])]
        # fit the "best" model over jackknifes
        a_sv = np.zeros(nstim)
        a_loading_sim = np.zeros(nstim)
        a_dim95 = np.zeros(nstim)
        for st, kk in enumerate(keep):
            x = X_asub[kk].squeeze()
            fa_active = FactorAnalysis(n_components=active_dim[st], random_state=0) 
            fa_active.fit(x.T)
            a_sv[st] = fhelp.get_sv(fa_active)
            a_loading_sim[st] = fhelp.get_loading_similarity(fa_active)
            # get n dims needs to explain 95% of shared variance
            a_dim95[st] = fhelp.get_dim95(fa_active)

            results["active"][kk] = {}
            results["active"][kk]["sv"] = a_sv[st]
            results["active"][kk]["loading_sim"] = a_loading_sim[st]
            results["active"][kk]["dim"] = a_dim95[st]
            results["active"][kk]["components_"] = fa_active.components_
            results["active"][kk]["sigma_shared"] = fhelp.sigma_shared(fa_active)
            results["active"][kk]["sigma_ind"] = fhelp.sigma_ind(fa_active)
            results["active"][kk]["sigma_full"] = fhelp.pred_cov(fa_active)

        # PASSIVE
        passive_dim = [fhelp.get_dim(LL_passive[:, i]) for i in range(LL_passive.shape[1])]
        # fit the "best" model over jackknifes
        p_sv = np.zeros(nstim)
        p_loading_sim = np.zeros(nstim)
        p_dim95 = np.zeros(nstim)
        for st, kk in enumerate(keep):
            x = X_psub[kk].squeeze()
            fa_passive = FactorAnalysis(n_components=passive_dim[st], random_state=0) 
            fa_passive.fit(x.T)
            p_sv[st] = fhelp.get_sv(fa_passive)
            p_loading_sim[st] = fhelp.get_loading_similarity(fa_passive)
            # get n dims needs to explain 95% of shared variance
            p_dim95[st] = fhelp.get_dim95(fa_passive)

            results["passive"][kk] = {}
            results["passive"][kk]["sv"] = p_sv[st]
            results["passive"][kk]["loading_sim"] = p_loading_sim[st]
            results["passive"][kk]["dim"] = p_dim95[st]
            results["passive"][kk]["components_"] = fa_passive.components_
            results["passive"][kk]["sigma_shared"] = fhelp.sigma_shared(fa_passive)
            results["passive"][kk]["sigma_ind"] = fhelp.sigma_ind(fa_passive)
            results["passive"][kk]["sigma_full"] = fhelp.pred_cov(fa_passive)


    def save(d, path):
        with open(path+f'/{modelname}.pickle', 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return None

    save(results, os.path.join(RESULTS_DIR, site))