import os

RESULTS_DIR = "/auto/users/hellerc/results/TBP_dryad"

if os.path.isdir(RESULTS_DIR)==False:
    os.mkdir(RESULTS_DIR)

BAD_SITES = [
    "CRD013b", # not enough trials
    "ARM004e", # not enough cells
    "ARM005e", # not enough cells
    "CLT016a", # not enough cells
]