"""
Cache final decoding models for all recording sites.

Original analysis ran parallelized on David lab compute cluster.

Reproduced here running all analyses in series. Could take a little while.
"""
import os
rdir = os.path.dirname(os.path.dirname(__file__))
import sys
sys.path.append(rdir)

from settings import RESULTS_DIR
import helpers.loaders as loaders
import helpers.decoding as decoding
from helpers.path_helpers import local_results_file
import pandas as pd
from itertools import combinations
import logging
log = logging.getLogger(__name__)


db = pd.read_csv(os.path.join(RESULTS_DIR, "db.csv"), index_col=0)
sites = db.site

modellist = [
    # shared shape model for visualization in fig 3
    'tbpDecoding_mask.pa_decmask.h.cr.m.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise-sharedSpace',
    # Standard active / passive decoding jobs
    'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise', 
    'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',
    # Standard active / passive decoding jobs with pupil regressed out
    'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR',
    'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR',
    # Factor analysis simulations
    'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.0.PR',
    'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.1.PR',
    'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.3.PR',
    'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.4.PR',
    'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.0.PR',
    'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.1.PR',
    'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.3.PR',
    'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.4.PR',
]

def parse_mask_options(op):
    mask = []
    mask_ops = op.split(".")
    pup_match_active = False
    for mo in mask_ops:
        if mo=="h":
            mask.append("HIT_TRIAL")
        if mo=="cr":
            mask.append("CORRECT_REJECT_TRIAL")
        if mo=="m":
            mask.append("MISS_TRIAL")
        if mo=="fa":
            mask.append("FALSE_ALARM_TRIAL")
        if mo=="pa":
            mask.append("PASSIVE_EXPERIMENT")
        if mo=="paB":
            mask.append("PASSIVE_EXPERIMENT")
            pup_match_active = True
    return mask, pup_match_active


for site in sites:
    print(f"\n\n {site} \n\n")
    for modelname in modellist:
        # for each site / model, run the analysis

        try:
            mask = []
            drmask = []
            decmask = []
            method = "unknown"
            ndims = 2
            noise = "global"
            sharedSpace = False
            factorAnalysis = False
            fa_perstim = False
            sim = None
            pup_match_active = False
            regress_pupil = False
            fa_model = "FA_perstim"
            for op in modelname.split("_"):
                if op.startswith("mask"):
                    mask, pup_match_active = parse_mask_options(op)
                if op.startswith("drmask"):
                    drmask, _ = parse_mask_options(op)
                if op.startswith("decmask"):
                    decmask, _ = parse_mask_options(op)
                if op.startswith("DRops"):
                    dim_reduction_options = op.split(".")
                    for dro in dim_reduction_options:
                        if dro.startswith("dim"):
                            ndims = int(dro[3:])
                        if dro.startswith("ddr"):
                            method = "dDR"
                            ddrops = dro.split("-")
                            for ddr_op in ddrops:
                                if ddr_op == "globalNoise":
                                    noise = "global"
                                elif ddr_op == "targetNoise":
                                    noise = "targets"
                                elif ddr_op == "sharedSpace":
                                    sharedSpace = True
                if op.startswith("PR"):
                    regress_pupil = True
                if op.startswith("FA"):
                    factorAnalysis = True
                    sim_method = int(op.split(".")[1])
                    fa_perstim = op.split(".")[0][2:]=="perstim"
                    try:
                        log.info("Using pupil regressed FA models")
                        if op.split(".")[2]=="PR":
                            fa_model = "FA_perstim_PR"
                    except:
                        log.info("Didn't find a pupil regress FA option")
                        pass


            if decmask == []:
                # default is to compute decoding axis using the same data you're evaluating on
                decmask = mask

            # STEP 3: Load data to be decoded / data to be use for decoding space definition
            X, Xp = loaders.load_tbp_for_decoding(site=site, 
                                                wins = 0.1,
                                                wine = 0.4,
                                                collapse=True,
                                                mask=mask,
                                                recache=False,
                                                pupexclude=pup_match_active,
                                                regresspupil=regress_pupil)

            # for null simulation, load the active PSTH regardless of current state
            X_all, _ = loaders.load_tbp_for_decoding(site=site, 
                                                wins = 0.1,
                                                wine = 0.4,
                                                collapse=True,
                                                mask=["HIT_TRIAL", "CORRECT_REJECT_TRIAL", "MISS_TRIAL", "PASSIVE_EXPERIMENT"],
                                                recache=False,
                                                pupexclude=pup_match_active,
                                                regresspupil=regress_pupil)

            # sim:
            #     0 = no change (null) model
            #     1 = change in gain only
            #     2 = change in indep only (fixing absolute covariance)
            #     3 = change in indep only (fixing relative covariance - so off-diagonals change)
            #     4 = change in everything (full FA simulation)
            #     # extras:
            #     5 = set off-diag to zero, only change single neuron var.
            #     6 = set off-diag to zero, fix single neuorn var
            #     7 = no change (and no correlations at all)
            Xog = X.copy()
            if factorAnalysis:
                # redefine X using simulated data
                if "PASSIVE_EXPERIMENT" in mask:
                    state = "passive"
                else:
                    state = "active"
                if fa_perstim:
                    log.info(f"Loading factor analysis results from {fa_model}")
                    if sim_method==0:
                        log.info("Fixing PSTH between active / passive to active")
                        keep = [k for k in X_all.keys() if ("TAR_" in k) | ("CAT_" in k)]
                        X_all = {k: v for k, v in X_all.items() if k in keep}
                        psth = {k: v.mean(axis=1).squeeze() for k, v in X_all.items()}
                        Xog = {k: v for k, v in X_all.items() if k in X.keys()}
                    else:
                        keep = [k for k in Xog.keys() if ("TAR_" in k) | ("CAT_" in k)]
                        Xog = {k: v for k, v in Xog.items() if k in keep}
                        psth = {k: v.mean(axis=1).squeeze() for k, v in Xog.items()}
                        Xog = {k: v for k, v in Xog.items() if k in X.keys()}

                    log.info("Loading FA simulation using per stimulus results")
                    X = loaders.load_FA_model_perstim(site, psth, state, fa_model=fa_model, sim=sim_method, nreps=2000)
                else:
                    log.info("Loading FA simulation")
                    psth = {k: v.mean(axis=1).squeeze() for k, v in X.items()}
                    X = loaders.load_FA_model(site, psth, state, sim=sim_method, nreps=2000)

            # always define the space with the raw data, for the sake of comparison
            Xd, _ = loaders.load_tbp_for_decoding(site=site, 
                                                wins = 0.1,
                                                wine = 0.4,
                                                collapse=True,
                                                mask=drmask,
                                                balance=True,
                                                regresspupil=regress_pupil)
            Xdec, _ = loaders.load_tbp_for_decoding(site=site, 
                                                wins = 0.1,
                                                wine = 0.4,
                                                collapse=True,
                                                mask=decmask,
                                                balance=True,
                                                regresspupil=regress_pupil)

            # STEP 4: Generate list of stimulus pairs meeting min rep criteria and get the decoding space for each
            stim_pairs = list(combinations(Xog.keys(), 2))
            stim_pairs = [sp for sp in stim_pairs if (Xog[sp[0]].shape[1]>=5) & (Xog[sp[1]].shape[1]>=5) & (Xd[sp[0]].shape[1]>=5) & (Xd[sp[1]].shape[1]>=5)]
            # TODO: Add option to compute a single, fixed space for all pairs. e.g. a generic
            # target vs. catch space.
            decoding_space = decoding.get_decoding_space(Xd, stim_pairs, 
                                                        method=method, 
                                                        noise_space=noise,
                                                        ndims=ndims,
                                                        common_space=sharedSpace)

            if len(decoding_space) != len(stim_pairs):
                raise ValueError

            # STEP 5: Loop over stimulus pairs and perform decoding
            output = []
            for sp, axes in zip(stim_pairs, decoding_space):
                # first, get decoding axis for this stim pair
                # TODO: Add specialty option for generic target vs. catch decoding space.
                _r1 = Xdec[sp[0]][:, :, 0]
                _r2 = Xdec[sp[1]][:, :, 0]
                _result = decoding.do_decoding(_r1, _r2, axes)
                
                r1 = X[sp[0]].squeeze()
                r2 = X[sp[1]].squeeze()
                result = decoding.do_decoding(r1, r2, axes, wopt=_result.wopt)
                pair_category = decoding.get_category(sp[0], sp[1])

                df = pd.DataFrame(index=["dp", "wopt", "evals", "evecs", "evecSim", "dU"],
                                    data=[result.dprimeSquared,
                                        result.wopt,
                                        result.evals,
                                        result.evecs,
                                        result.evecSim,
                                        result.dU]
                                        ).T
                df["class"] = pair_category
                df["e1"] = sp[0]
                df["e2"] = sp[1]
                df["dr_loadings"] = [axes]

                output.append(df)

            output = pd.concat(output)

            dtypes = {
                'dp': 'float32',
                'wopt': 'object',
                'evecs': 'object',
                'evals': 'object',
                'evecSim': 'float32',
                'dU': 'object',
                'e1': 'object',
                'e2': 'object',
                "dr_loadings": "object",
                'class': 'object',
                }
            output = output.astype(dtypes)

            # STEP 6: Save results
            results_file = local_results_file(RESULTS_DIR, site, modelname, "output.pickle")
            output.to_pickle(results_file)

        except:
            print(f"Failed for {site} / {modelname} -- usually means didn't meet min. trial criteria")
