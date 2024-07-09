import os
rdir = os.path.dirname(__file__)
import sys
sys.path.append(rdir)

from nems.tools import recording

from settings import RESULTS_DIR
import preprocessing as preproc
import tin_helpers as thelp
import numpy as np
import pandas as pd
import pickle
import logging

log = logging.getLogger(__name__)


db = pd.read_csv(os.path.join(RESULTS_DIR, "db.csv"), index_col=0)


def load_tbp_for_decoding(site, mask, fs=10, wins=0.1, wine=0.4, collapse=True, 
                    recache=False, get_full_trials=False, from_trial_start=False, balance=False, balance_choice=False, 
                    pupexclude=False, regresspupil=False):
    """
    mask is list of epoch categories (e.g. HIT_TRIAL) to include in the returned data

    balance: If true, on a per stimulus basis, make sure there are equal number of 
        active and passive trials (using random subsampling of larger category)

    regresspupil: If true, use linear regression to remove anything that can be explained with first order pupil

    pupexclude: If true and mask=["PASSIVE_EXPERIMENT"], exclude trials where pupil size does
        not match active distribution of pupil size.

    get_full_trials: If true, in the returned dictionary also return full trials, with labels matching the
        respective stim of interest on that trial. For example, if I wanted all full catch trials, by returned
        X dict would have a field called X["TRIAL_<catch token>"].
        Only do this for CAT_ and TAR_ stim, though. Otherwise going to be a crazy big dictionary

    return: 
        X - neuron x rep x time bin (spike counts dictionary for each epoch)
        Xp - 1 x rep x time bin (pupil size dict)
    """     

    if from_trial_start & (get_full_trials==False):
        raise ValueError("To reference from trial start, get_full_trials must be True")

    if fs==10:
        uri = os.path.join(RESULTS_DIR, "recordings", db.loc[site, "10hz_uri"])
    elif fs==50:
        uri = os.path.join(RESULTS_DIR, "recordings", db.loc[site, "50hz_uri"])
    else:
        raise ValueError("fs must be either 10 or 50")
    
    rec = recording.load_recording(uri)
    rec['resp'] = rec['resp'].rasterize()
    if regresspupil:
        rec = preproc.regress_state(rec, 
                        state_sigs=['pupil'],
                        regress=['pupil'])
    rec = rec.create_mask(True)
    rec = rec.and_mask(mask)
    if pupexclude & ("PASSIVE_EXPERIMENT" in mask):
        _r = rec.copy()
        _r = _r.create_mask(True)
        _r = _r.and_mask(["CORRECT_REJECT_TRIAL", "HIT_TRIAL", "MISS_TRIAL"])
        cutoff = rec['pupil']._data[_r["mask"]._data].mean() - (2 * rec['pupil']._data[_r["mask"]._data].std())
        options = {'state': 'big', 'method': 'user_def_value', 'cutoff': cutoff, 'collapse': True, 'epoch': ['REFERENCE', 'TARGET', 'CATCH']}
        pass_big_mask = preproc.create_pupil_mask(rec, **options)['mask']
        rec["mask"] = rec['mask']._modified_copy(pass_big_mask._data)

    # get tbp epochs and create spike matrix using mask
    _, _, all_stim = thelp.get_sound_labels(rec)

    # hack to force pupil to have same epochs as response. Not sure why they're different sometimes though...
    rec["pupil"].epochs = rec["resp"].epochs

    # Get "target" label of each trial. Save this in names_list
    # also get target (or catch) onset for each trial
    names_list = []
    tstarts = []
    if get_full_trials:
        repochs = rec.apply_mask(reset_epochs=True)["resp"].epochs
        trial_epochs = repochs[repochs.name=="TRIAL"]
        for i in range(trial_epochs.shape[0]):
            this_trial = trial_epochs.iloc[i]
            this_target = repochs[(repochs.start>=this_trial.start) & (repochs.end<=this_trial.end) & ((repochs.name.str.startswith("TAR_")) | (repochs.name.str.startswith("CAT_")))]
            if this_target.shape[0]>1:
                log.info("multiple matches for TAR/CAT on this trial. Only keep CAT, in this case")
                mm = np.argwhere(this_target.name.str.contains("CAT_")).squeeze()
                this_epoch_is = this_target.name.iloc[mm]
                this_start = this_target.start.iloc[mm]
            elif this_target.shape[0]==0:
                print(i)
                import pdb; pdb.set_trace()
                raise ValueError("No targets or catches found on this trial")
            else:
                this_epoch_is = this_target.name.iloc[0]
                this_start = this_target.start.iloc[0]

            tstarts.append(this_start - this_trial["start"])
            names_list.append(this_epoch_is)
    
    # overwrite t0 as all 0's (since the default, in the above if condition, is to set tstarts to the start of the target / catch)
    if from_trial_start:
        tstarts = [0 for _ in tstarts]

    # different procedure if we're only working with TRIAL epochs
    if get_full_trials:
        # load data into dictionary
        # need to adjust start time / end for each specific trial to get us into relative time
        bss = ((wins * fs) + np.array(tstarts)*fs).astype(int)
        bee = ((wine * fs) + np.array(tstarts)*fs).astype(int)
        length = int(bee[0] - bss[0])
        r = rec["resp"].extract_epochs(["TRIAL"], mask=rec["mask"])
        p = rec["pupil"].extract_epochs(["TRIAL"], mask=rec["mask"])
        if collapse:
            rnew = {k: v[:, :, [1]] for k, v in r.items()}
            pnew = {k: v[:, :, [1]] for k, v in p.items()}
            for i, (bs, be) in enumerate(zip(bss, bee)):
                if bs >= 0:
                    rnew["TRIAL"][i, :, :] = r["TRIAL"][i, :, bs:be].mean(axis=-1, keepdims=True)
                    pnew["TRIAL"][i, :, :] = p["TRIAL"][i, :, bs:be].mean(axis=-1, keepdims=True)
                else:
                    rnew["TRIAL"][i, :, :] = np.nan
                    pnew["TRIAL"][i, :, :] = np.nan
        else:
            rnew = {k: v[:, :, :length] for k, v in r.items()}
            pnew = {k: v[:, :, :length] for k, v in p.items()}
            for i, (bs, be) in enumerate(zip(bss, bee)):
                if bs>=0:
                    rnew["TRIAL"][i, :, :] = r["TRIAL"][i, :, bs:be]
                    pnew["TRIAL"][i, :, :] = p["TRIAL"][i, :, bs:be]
                else:
                    rnew["TRIAL"][i, :, :] = np.nan
                    pnew["TRIAL"][i, :, :] = np.nan

        r = {k: v.transpose([1, 0, -1]) for k, v in rnew.items()}
        p = {k: v.transpose([1, 0, -1]) for k, v in pnew.items()}
        
        # update the names of TRIAL epoch correctly
        for nn in np.unique(names_list):
            new_name = f"TRIAL_{nn}"
            this_mask = np.argwhere(np.array(names_list)==nn).squeeze()
            resp = r["TRIAL"][:, this_mask, :]
            pupil = p["TRIAL"][:, this_mask, :]
            r[new_name] = resp
            p[new_name] = pupil
        # remover overall "TRIAL" label from dictionary
        _ = r.pop("TRIAL")
        _ = p.pop("TRIAL")
        # remove everything else that doesn't start with TRIAL
        rnew = r.copy()
        pnew = p.copy()
        for k in r.keys():
            if k.startswith("TRIAL")==False:
                _ = rnew.pop(k)
                _ = pnew.pop(k)
        r = rnew.copy()
        p = pnew.copy()
        # redefine the list of "all stim" to look at
        all_stim = [str(s) for s in np.unique(names_list)]

    else:
        # load response dictionary, default situation
        bs = int(wins * fs)
        be = int(wine * fs)
        r = rec["resp"].extract_epochs(all_stim, mask=rec["mask"])
        p = rec["pupil"].extract_epochs(all_stim, mask=rec["mask"])
        if collapse:
            r = {k: v[:, :, bs:be].mean(axis=-1, keepdims=True) for k, v in r.items()}
            p = {k: v[:, :, bs:be].mean(axis=-1, keepdims=True) for k, v in p.items()}
        else:
            r = {k: v[:, :, bs:be] for k, v in r.items()}
            p = {k: v[:, :, bs:be] for k, v in p.items()}
        r = {k: v.transpose([1, 0, -1]) for k, v in r.items()}
        p = {k: v.transpose([1, 0, -1]) for k, v in p.items()}

    if (len(mask) > 1) & ("PASSIVE_EXPERIMENT" in mask) & balance:
        np.random.seed(123)
        # balance active vs. passive trials
        rnew = rec.copy()
        rnew = rnew.create_mask(True)
        rnew = rnew.and_mask(["PASSIVE_EXPERIMENT"])
        for s in all_stim:
            pmask = rnew["mask"].extract_epoch(s, mask=rec["mask"])
            nptrials = pmask[:, 0, 0].sum() 
            natrials = (pmask[:, 0, 0] == False).sum() 
            nktrials = np.min([nptrials, natrials])
            pchoose = np.random.choice(np.argwhere(pmask[:, 0, 0]).squeeze(), nktrials, replace=False)
            achoose = np.random.choice(np.argwhere(pmask[:, 0, 0]==False).squeeze(), nktrials, replace=False)
            choose = np.sort(np.concatenate((achoose, pchoose)))
            r[s] = r[s][:, choose, :]
            p[s] = p[s][:, choose, :]
    
    elif (type(mask)==str) & balance_choice:
        # this is a pretty speciality case, designed for choice decoding
        np.random.seed(123)
        # balance active vs. passive trials
        if mask=="HIT_TRIAL":
            m2 = "MISS_TRIAL"
            all_stim = [s for s in all_stim if ("CAT_" not in s) & ("STIM_" not in s)]
        if mask=="MISS_TRIAL":
            m2 = "HIT_TRIAL"
            all_stim = [s for s in all_stim if ("CAT_" not in s) & ("STIM_" not in s)]
        if mask=="CORRECT_REJECT_TRIAL":
            m2 = "INCORRECT_HIT_TRIAL"
            all_stim = [s for s in all_stim if ("TAR_" not in s) & ("STIM_" not in s)]
        if mask=="INCORRECT_HIT_TRIAL":
            m2 = "CORRECT_REJECT_TRIAL"
            all_stim = [s for s in all_stim if ("TAR_" not in s) & ("STIM_" not in s)]
        rnew = rec.copy()
        rnew = rnew.create_mask(True)
        rnew = rnew.and_mask([m2])
        all_stim = [s for s in all_stim if (rnew.and_mask([s])["mask"]._data.sum()>0) & (rec.and_mask([s])["mask"]._data.sum()>0)]
        for s in all_stim:
            pmask = rnew["mask"].extract_epoch(s)
            nptrials = pmask[:, 0, 0].sum() 
            try:
                natrials = r[s].shape[1]
                nktrials = np.min([nptrials, natrials])
                pchoose = np.random.choice(range(natrials), nktrials, replace=False)
                choose = np.sort(pchoose)
                r[s] = r[s][:, choose, :]
                p[s] = p[s][:, choose, :]
            except KeyError:
                log.info(f"{s} wasn't found in dict. Trying TRIAL_{s} instead...")
                natrials = r[f"TRIAL_{s}"].shape[1]
                nktrials = np.min([nptrials, natrials])
                if nktrials > 1:
                    pchoose = np.random.choice(range(natrials), nktrials, replace=False)
                    choose = np.sort(pchoose)
                    r[f"TRIAL_{s}"] = r[f"TRIAL_{s}"][:, choose, :]
                    p[f"TRIAL_{s}"] = p[f"TRIAL_{s}"][:, choose, :]
                else:
                    # not enough trials to do anything anyways
                    _ = r.pop(f"TRIAL_{s}")
                    _ = p.pop(f"TRIAL_{s}")

    return r, p 



def load_FA_model(site, psth, state, sim=1, rr=None, fa_model="FA", nreps=2000):
    """
    pretty specialized code to load the results of factor analysis model
    and generate data based on this. Since only one psth, if you want to manipulate first
    order (e.g. swap psth for active / passive) that has to happen outside this function.

    psth should be a dictionary with entries of len nCells
    return a dictionary with entries nCells x nreps (simulated)

    generate nreps per stimulus

    state = active or passive

    sim:
        0 = no change (null) model
        1 = change in gain only
        2 = change in indep only (fixing absolute covariance)
        3 = change in indep only (fixing relative covariance - so off-diagonals change)
        4 = change in everything (full FA simulation)
        # extras:
        5 = set off-diag to zero, only change single neuron var.
        6 = set off-diag to zero, fix single neuorn var
        7 = no change (and no correlations at all)
    """
    np.random.seed(123)
    # load the model results
    path = os.path.join(RESULTS_DIR, site)
    filename = f"{fa_model}.pickle"
    with open(path + filename, 'rb') as handle:
        results = pickle.load(handle)

    # if reduced rank, then compute the new, reduced rank shared matrix here (doesn't apply for diag)
    def sigma_shared(components):
        return (components.T @ components)
    if rr is not None:
        active_factors_unique = results["final_fit"]["fa_active.components_"][:rr, :]
        passive_factors_unique = results["final_fit"]["fa_passive.components_"][:rr, :]
        # then, share the rest (big for both)
        factors_shared = results["final_fit"]["fa_active.components_"][rr:, :]
        if factors_shared.shape[0]>0:
            results["final_fit"]["fa_active.sigma_shared"] = sigma_shared(np.concatenate((active_factors_unique, factors_shared), axis=0))
            results["final_fit"]["fa_passive.sigma_shared"] = sigma_shared(np.concatenate((passive_factors_unique, factors_shared), axis=0))
        else:
            results["final_fit"]["fa_active.sigma_shared"] = sigma_shared(active_factors_unique)
            results["final_fit"]["fa_passive.sigma_shared"] = sigma_shared(passive_factors_unique)

    Xsim = dict.fromkeys(psth.keys())
    if sim==0:
        cov_active = results["final_fit"]["fa_active.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
        cov_passive = results["final_fit"]["fa_active.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
    if sim==1:
        cov_active = results["final_fit"]["fa_active.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
        cov_passive = results["final_fit"]["fa_active.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
    elif sim==2:
        # absolute covariance fixed, but fraction shared variance can change
        cov_active = results["final_fit"]["fa_active.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
        cov_passive = results["final_fit"]["fa_passive.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
    elif sim==3:
        # relative covariance fixed, i.e. fraction shared variance can stays the same but absolute covariance can change
        cov_active = results["final_fit"]["fa_active.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
        cov_passive = results["final_fit"]["fa_passive.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
        # force small to have same corr. coef. as cov_active
        norm = np.sqrt(np.diag(cov_active)[:, np.newaxis] @ np.diag(cov_active)[np.newaxis, :])
        corr_active = cov_active / norm # normalize covariance
        var = np.diag(cov_passive) # variance of small pupil          
        rootV = np.sqrt(var[:, np.newaxis] @ var[np.newaxis, :])
        cov_passive = corr_active * rootV # cov small has same (normalized) correlations as cov_active, but variance like cov_passive
    elif sim==4:
        cov_active = results["final_fit"]["fa_active.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
        cov_passive = results["final_fit"]["fa_passive.sigma_ind"] + results["final_fit"]["fa_passive.sigma_shared"]
    elif sim==5:
        # diag matrix, entries change between large and small
        cov_active = results["final_fit"]["fa_active.sigma_ind"]
        cov_passive = results["final_fit"]["fa_passive.sigma_ind"]
    elif sim==6:
        # diag matrix, entries fixed to big pupil between states
        cov_active = results["final_fit"]["fa_active.sigma_ind"]
        cov_passive = results["final_fit"]["fa_active.sigma_ind"]
    elif sim==7:
        cov_active = results["final_fit"]["fa_active.sigma_ind"]
        cov_passive = results["final_fit"]["fa_active.sigma_ind"]

    for s, k in enumerate(psth.keys()):
        _ca = cov_active.copy()            
        _cp = cov_passive.copy()
        if state=="active":
            cov_to_use = _ca
        elif state=="passive":
            cov_to_use = _cp
        Xsim[k] = np.random.multivariate_normal(psth[k], cov=cov_to_use, size=nreps).T

    return Xsim

def load_FA_model_perstim(site, psth, state, sim=1, fa_model="FA_perstim", nreps=2000):
    """
    Only distinct from the above function in that this loads / uses per-stimulus FA results

    pretty specialized code to load the results of factor analysis model
    and generate data based on this. Since only one psth, if you want to manipulate first
    order (e.g. swap psth for active / passive) that has to happen outside this function.

    psth should be a dictionary with entries of len nCells
    return a dictionary with entries nCells x nreps (simulated)

    generate nreps per stimulus

    state = active or passive

    sim:
        0 = no change (null) model
        1 = change in gain only
        2 = change in indep only (fixing absolute covariance)
        3 = change in indep only (fixing relative covariance - so off-diagonals change)
        4 = change in everything (full FA simulation)
        # extras:
        5 = set off-diag to zero, only change single neuron var.
        6 = set off-diag to zero, fix single neuorn var
        7 = no change (and no correlations at all)
    """
    np.random.seed(123)
    # load the model results
    path = os.path.join(RESULTS_DIR, site)
    filename = f"{fa_model}.pickle"
    with open(os.path.join(path, filename), 'rb') as handle:
        results = pickle.load(handle)

    cov_active = dict.fromkeys(psth.keys())
    cov_passive = dict.fromkeys(psth.keys())
    keep_keys = []
    for k in psth.keys():
        try:
            if sim==0:
                cov_active[k] = results["active"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
                cov_passive[k] = results["active"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
            if sim==1:
                cov_active[k] = results["active"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
                cov_passive[k] = results["active"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
            elif sim==2:
                # absolute covariance fixed, but fraction shared variance can change
                cov_active[k] = results["active"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
                cov_passive[k] = results["passive"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
            elif sim==3:
                # relative covariance fixed, i.e. fraction shared variance can stays the same but absolute covariance can change
                cov_active[k] = results["active"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
                cov_passive[k] = results["passive"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
                # force small to have same corr. coef. as cov_active
                norm = np.sqrt(np.diag(cov_active[k])[:, np.newaxis] @ np.diag(cov_active[k])[np.newaxis, :])
                corr_active = cov_active[k] / norm # normalize covariance
                var = np.diag(cov_passive[k]) # variance of small pupil          
                rootV = np.sqrt(var[:, np.newaxis] @ var[np.newaxis, :])
                cov_passive[k] = corr_active * rootV # cov small has same (normalized) correlations as cov_active, but variance like cov_passive
            elif sim==4:
                cov_active[k] = results["active"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
                cov_passive[k] = results["passive"][k]["sigma_ind"] + results["passive"][k]["sigma_shared"]
            elif sim==5:
                # diag matrix, entries change between large and small
                cov_active[k] = results["active"][k]["sigma_ind"]
                cov_passive[k] = results["passive"][k]["sigma_ind"]
            elif sim==6:
                # diag matrix, entries fixed to big pupil between states
                cov_active[k] = results["active"][k]["sigma_ind"]
                cov_passive[k] = results["active"][k]["sigma_ind"]
            elif sim==7:
                cov_active[k] = results["active"][k]["sigma_ind"]
                cov_passive[k] = results["active"][k]["sigma_ind"]

            keep_keys.append(k)
        except KeyError:
            print(f"Missing key {k}. Skip simulation for this epoch")

    Xsim = dict.fromkeys(keep_keys)
    for k in keep_keys:
        _ca = cov_active[k].copy()            
        _cp = cov_passive[k].copy()
        if state=="active":
            cov_to_use = _ca
        elif state=="passive":
            cov_to_use = _cp
        Xsim[k] = np.random.multivariate_normal(psth[k], cov=cov_to_use, size=nreps).T

    return Xsim