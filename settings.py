import os
rr = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(rr, "data/elife2024/eLife2024_data")

if os.path.isdir(RESULTS_DIR)==False:
    os.system(f"mkdir {RESULTS_DIR}")

BAD_SITES = [
    "CRD013b", # not enough trials
    "ARM004e", # not enough cells
    "ARM005e", # not enough cells
    "CLT016a", # not enough cells
]

# ================ matplotlib settings =====================
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8
mpl.rcParams['xtick.labelsize'] = 8 
mpl.rcParams['ytick.labelsize'] = 8 

# set inline backend for VS Code using Python extenstion
# mpl.rcParams['backend'] =  'module://matplotlib_inline.backend_inline'

# Interactive plotting is the defualt
mpl.rcParams['backend'] = "QtAgg"
# more information @: https://matplotlib.org/stable/users/explain/figure/backends.html
