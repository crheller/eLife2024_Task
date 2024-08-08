# Task-specific invariant representation in auditory cortex
#### Heller, C.R., Hamersky, G.R., David, S.V.

This repository contains the code to run all analyses presented in [Heller et al, 2023](https://elifesciences.org/reviewed-preprints/89936). The data is publicly available at: [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.z08kprrp4) and should be downloaded as follows, prior to running any code in this repository:

## Download data
* Navigate to our [dataset](https://datadryad.org/stash/dataset/doi:10.5061/dryad.z08kprrp4) which is hosted on Dryad.

* Click the dropdown "version files" button in the upper right corner of the page

* Click on `elife2024.zip` to download the data

* Extract the contents of this downloaded zip file into a new subdirectory of this repository called `data`

* Check that inside the `eLife2024_Task` repository you now have a subdirectory with the structure: `data/elife2024/eLife_2024/<data_files>`

* Note - You can extract the data to whichever location you want. However, if you put it in a different location than the one above, make sure to update the `RESULTS_DIR` variable in `settings.py` accordingly.

## Create conda environment
In a terminal, cd into the cloned repository and run the following commmand:
```
conda env create -f environment.yml
```
This might take a minute or two to collect packages and solve the environment. This is normal. Once the process finished, this will create a conda environment called `elife2024` which should have all the necessary dependencies installed for running the analysis in the repository.

## Reproduce manuscript figures
Using the conda environment you just set up  (e.g., `conda activate elife2024`), you should now be able to run the code that reproduces the manuscript figures. The scripts to produce each figure are located in a figure specific subdirectory of `figure_scripts`. For example, all scripts needed to generate the the contents of Figure 2 (and associated supplementals) are located in `figure_scripts/fig2`.

## `matplotlib` backend
This can be specified in the `settings.py` while and should be used by all figure scripts. The default backend is `QtAgg`.

## Recaching analysis files
Included in the data files downloaded from Dryad are a set of "analysis files" which contain cached results for analysis such as the pairwise stimulus decoding. These can all be regenerated, if desired, by re-running the analysis on your own machine. The scripts for re-caching all analyses are located in `cacheScripts`. At the top of each `.py` cache script you will find a brief description of the analysis that it will perform. 
