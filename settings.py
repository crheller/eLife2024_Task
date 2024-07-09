import os
rr = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(rr, "eLife2024_data")

if os.path.isdir(RESULTS_DIR)==False:
    os.system(f"mkdir {RESULTS_DIR}")

BAD_SITES = [
    "CRD013b", # not enough trials
    "ARM004e", # not enough cells
    "ARM005e", # not enough cells
    "CLT016a", # not enough cells
]