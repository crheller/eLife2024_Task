import numpy as np
from sklearn.linear_model import LinearRegression
import copy
import logging

log = logging.getLogger(__name__)



def regress_state(rec, state_sigs=['behavior', 'pupil'], regress=None):
    """
    Remove first order effects of given state variable(s). Idea is to model all state
    variables, for example, pupil and behavior, then just choose to remove the first order
    effects of one or the other, or both.

    1) Compute compute psth in each bin by averaging over the whole recording
    2) At each time bin, model residuals as linear function of state variable(s)
    3) Subtract off prediction computed using the coef(s) for the "regress" states
    4) Add the corrected residuals back to the overall psth
    """
    if regress is not None:
        log.info(DeprecationWarning('regress argument is deprecated. Always regress out all state signals'))

    r = copy.deepcopy(rec)
    ep = np.unique([ep for ep in r.epochs.name if ('STIM' in ep) | ('TAR_' in ep)]).tolist()
    
    r_st = r['resp'].extract_epochs(ep)
    state_signals = dict.fromkeys(state_sigs)
    for s in state_sigs:
        if s == 'pupil':
            state_signals[s] = r['pupil'].extract_epochs(ep)
        elif s == 'behavior':
            r_beh_mask = r.create_mask(True)
            r_beh_mask = r_beh_mask.and_mask(['ACTIVE_EXPERIMENT'])
            state_signals[s] = r_beh_mask['mask'].extract_epochs(ep)
        elif s == 'lv':
            state_signals[s] = r['lv'].extract_epochs(ep)
        else:
            raise ValueError("No case set up for {}".format(s))

    r_psth = r_st.copy()
    r_new = r_st.copy()
    for e in ep:
        m = r_st[e].mean(axis=0)
        r_psth[e] = np.tile(m, [r_st[e].shape[0], 1, 1])
        # for each stim bin
        for b in range(r_st[e].shape[-1]):
            # for each neuron, regress out state effects
            for n in range(r_psth[e].shape[1]):
                for i, s in enumerate(state_sigs):
                    if i == 0:
                        X = state_signals[s][e][:, :, b]
                    else:
                        X = np.concatenate((X, state_signals[s][e][:, :, b]), axis=-1)

                y = r_st[e][:, n, b] - r_psth[e][:, n, b]
                reg = LinearRegression()

                # zscore if std of state signal not 0
                X = X - X.mean(axis=0)
                nonzero_sigs = np.argwhere(X.std(axis=0)!=0).squeeze()
                if nonzero_sigs.shape != (0,):
                    X = X[:, nonzero_sigs] / X[:, nonzero_sigs].std(axis=0)
                    if len(X.shape) == 1:
                        X = X[:, np.newaxis]
                    if (np.any(np.isnan(y)) | np.any(np.isnan(X))):
                        if n==0:
                            log.info(f"Found nans in data for bin {b}, epoch: {e}. Not regressing out state")
                        model_coefs = np.zeros(X.shape[-1])
                        intercept = 0
                    else:
                        reg.fit(X, y[:, np.newaxis])
                        model_coefs = reg.coef_
                        intercept = reg.intercept_
                    y_pred = np.matmul(X, model_coefs.T) + intercept
                    y_new_residual = y - y_pred.squeeze()
                    r_new[e][:, n, b] = r_psth[e][:, n, b] + y_new_residual
                else:
                    # state signal has 0 std so nothing to regress out
                    r_new[e][:, n, b] = r_psth[e][:, n, b] + y                

    r['resp'] = r['resp'].replace_epochs(r_new)

    return r
