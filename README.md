# Task-specific invariant representation in auditory cortex
#### Heller, C.R., Hamersky, G.R., David, S.V.

This repository contains the code to run all analyses presented in [Heller et al, 2023](https://elifesciences.org/reviewed-preprints/89936). The data is publicly available at: [Dryad Link](https://www.dontexist.com) and should be downloaded as follows, prior to running any code in this repository:

## Download data
This will constain instructions on downloading / unzipping the Dryad data into the correct location. Dryad will also contain more information about the specific data files included, and their contents.

Could either force users to put it in this repo under a folder called `eLife2024_data`, or could be more flexbile and just direct users to update the variable `RESULTS_DIR` in `settings.py` to reflect their path to wherever they decided to download the data.

For now, it's just included in the git repository under `eLife2024_data`.

## Create conda environment
In a terminal, run the following commmand:
```
conda env create -f environment.yml
```
This will create a conda environment called `elife2024` which should have all the necessary dependencies installed for running the analysis in the repository.

## Reproduce manuscript figures
Using the conda environment you just set up, you should now be able to run the code that reproduces the manuscript figures. The scripts to produce each figure are located in a figure specific subdirectory of `figure_scripts`. For example, all scripts needed to generate the the contents of Figure 2 (and associated supplementals) are located in `figure_scripts/fig2`.

## `matplotlib` backend
This can be specified in the `settings.py` while and should be used by all figure scripts. The default backend is `QtAgg`.

## Recaching analysis files
Included in the data files downloaded from Dryad are a set of "analysis files" which contain cached results for analysis such as the pairwise stimulus decoding. These can all be regenerated, if desired, by re-running the analysis on your own machine. The scripts for re-caching all analyses are located in `cacheScripts`. At the top of each `.py` cache script you will find a brief description of the analysis that it will perform. 
