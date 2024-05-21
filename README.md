# Case studies to investigate the integration of time resolved RNA data into a GUTS TKTD model

This project studies the integration of gene expression data into a TKTD model at the example of *Nrf2* expression in zebrafish embryos (ZFE) after the exposure to Naproxen, Diuron or Diclofenac. Other observed variables are the concentration inside the organism and lethality.

The data were acquired from 2015 to 2022 at the UFZ in Leipzig. Modeling was done with the modeling platform pymob (github.com/flo-schu/pymob) in Python.
The associated publication is ...

## Instructions

### Prerequisites

Install git (https://git-scm.com/downloads) and conda (https://docs.anaconda.com/free/miniconda/)

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

The case studies should now be ready to use.

### Usage

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

