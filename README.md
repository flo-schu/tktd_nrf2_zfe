# Case studies to investigate the integration of time resolved RNA data into a GUTS TKTD model

This project studies the integration of gene expression data into a TKTD model at the example of *Nrf2* expression in zebrafish embryos (ZFE) after the exposure to Naproxen, Diuron or Diclofenac. Other observed variables are the concentration inside the organism and lethality.

The data were acquired from 2015 to 2022 at the UFZ in Leipzig. Modeling was done with the modeling platform pymob (github.com/flo-schu/pymob) in Python.
The associated publication is ...

## Instructions

### Prerequisites

Install git (https://git-scm.com/downloads), conda (https://docs.anaconda.com/free/miniconda/) and datalad (https://handbook.datalad.org/en/latest/intro/installation.html)

### Installation

Open a command line utility (cmd, bash, ...) and execute

```bash
git clone git@github.com:flo-schu/tktd_nrf2_zfe
cd tktd_nrf2_zfe
```

Create environment, activate it and install model package. 
```bash
conda create -n tktd_nrf2_zfe
conda activate tktd_nrf2_zfe
conda install python=3.11
pip install -r requirements.txt
```

### Downloading data and pre-computed results

download the input dataset
```bash
datalad clone git@gin.g-node.org:/flo-schu/tktd_nrf2_zfe__data.git data
```

download results to the case study submodules
```bash
datalad clone git@gin.g-node.org:/flo-schu/guts__results.git case_studies/guts/results
datalad clone git@gin.g-node.org:/flo-schu/tktd_rna_pulse__results.git case_studies/tktd_rna_pulse/results
```

The case studies should now be (almost) ready to use. In order to be able to modify the pre-computed results, the datasets must be unlocked first:

```bash
datalad unlock case_studies/tktd_rna_pulse/results
datalad unlock case_studies/guts/results
```

If this does not work, visit the repositories and download the contents directly into
the given locations

- https://gin.g-node.org/flo-schu/tktd_rna_pulse__results > case_studies/tktd_rna_pulse/results
- https://gin.g-node.org/flo-schu/guts__results > case_studies/guts/results
- https://gin.g-node.org/flo-schu/tktd_nrf2_zfe__data > data

### Usage


You may have noticed that `datalad` (https://handbook.datalad.org/en/latest/index.html) is used to obtain data and pre-computed results. Datalad is a data management tool that allows researchers to track the evolution of datasets and results. 

In order to modify saved (and locked from accidental modifcation) states of the results follow this recipe:

```bash
datalad unlock case_studies/tktd_rna_pulse/results
datalad unlock case_studies/guts/results

python scripts/any-analysis-script.py

datalad -d case_studies/tktd_rna_pulse/results save -m "Short description of what I did." 
datalad -d case_studies/guts/results save -m "Short description of what I did." 
```

This recipe is necessary, because datalad (and under the hood git-annex) prevents you from accidentally modifying files. Unlocking results files first, tells datalad that you are aware that you are changing results files and you are okay with that. 

Once the dataset is unlocked, you can do as many modifications of the results as you want. Whenever you want to capture a given status: `datalad save -d case_studies/CASE_STUDY/results` and unlock again.

If a script fails with a **Permission Denied Error**, locked files are the most likely cause for it! See: https://handbook.datalad.org/en/latest/basics/101-110-run2.html#if-outputs-already-exist

#### Interactive jupyter notebooks 

The preferred files to get started with are in the case_studies directory

- case_studies/guts/scripts/tktd_guts_reduced.ipynb
- case_studies/guts/scripts/tktd_guts_scaled_damage.ipynb
- case_studies/guts/scripts/tktd_guts_full_nrf2.ipynb
- case_studies/tktd_rna_pulse/scripts/tktd_rna_3_6c_substance_specific.ipynb
- case_studies/tktd_rna_pulse/scripts/tktd_rna_3_6c_substance_independent.ipynb

Open and interact these files with your preferred development environment. If you
don't have any: `pip install jupyterlab` inside the console with the activated
(conda) environment, navigate to the notebooks and run the notebooks.

#### Command line

there is a command line script provided by the `pymob` package which will directly
run inference accroding to the scenario settings.cfg file provided in the scenario
folder of the respective case study. For details see https://pymob.readthedocs.io/en/latest/case_studies.html

`pymob-infer --case_study guts --scenario guts_scaled_damage --inference_backend numpyro`
`pymob-infer --case_study tktd_rna_pulse --scenario rna_pulse_3_6c_substance_specific --inference_backend numpyro`

The options in the `settings.cfg` are the same as used when preparing the publication

The results will be stored in the results directory of the respective case study 
unless otherwise specified in the settings.cfg

## Scripts to reproduce the analyses of the paper

