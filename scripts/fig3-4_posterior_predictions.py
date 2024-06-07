import os
import warnings
import importlib

import pandas as pd
import numpy as np
import xarray as xr
import arviz as az
from matplotlib import pyplot as plt
import matplotlib as mpl

from pymob.utils.plot_helpers import plot_hist
from pymob.utils.store_file import prepare_casestudy
from pymob.inference.analysis import plot_posterior_samples, bic
from method_posterior_analysis import log


def format_parameter(par, subscript_sep="_", superscript_sep="__", textwrap="\\text{}"):
    super_pos = par.find(superscript_sep)
    sub_pos = par.find(subscript_sep)
    
    scripts = sorted(zip([sub_pos, super_pos], [subscript_sep, superscript_sep]))
    scripts = [sep for pos, sep in scripts if pos > -1]


    def wrap_text(substr):
        if len(substr) == 1:
            substring_fmt = f"{substr}"
        else:
            substring_fmt = textwrap.replace("{}", "{{{}}}").format(substr)
    
        return f"{{{substring_fmt}}}"

    formatted_string = "$"
    for i, sep in enumerate(scripts):
        substr, par = par.split(sep, 1)
        substring_fmt = wrap_text(substr=substr)

        math_sep = "_" if sep == subscript_sep else "^"

        formatted_string += substring_fmt + math_sep

    formatted_string += wrap_text(par) + "$"

    return formatted_string

def create_table(posterior, error_metric="hdi", vars={}):
    parameters = list(posterior.data_vars.keys())

    tab = az.summary(posterior, fmt="xarray", kind="stats", stat_focus="mean", hdi_prob=0.94)
    if len(vars) > 0:
        tab = tab[vars.keys()]

    tab = tab.rename(vars)

    if error_metric == "sd":
        arrays = []
        for par in parameters:
            par_formatted = tab.sel(metric=["mean", "sd"])[par]\
                .round(3)\
                .astype(str).str\
                .join("metric", sep=" ±")
            arrays.append(par_formatted)

        formatted_tab = xr.combine_by_coords(arrays).to_dataframe().T

        formatted_parameters = []
        for idx in formatted_tab.index:
            formatted_parameters.append(format_parameter(idx))


        formatted_tab.index = formatted_parameters
        return formatted_tab    
    

    elif error_metric == "hdi":
        stacked_tab = tab.sel(metric=["mean", "hdi_3%", "hdi_97%"])\
            .assign_coords(metric=["mean", "hdi 3%", "hdi 97%"])\
            .stack(col=("substance", "metric"))\
            .round(2)
        formatted_tab = stacked_tab.to_dataframe().T.drop(index=["substance", "metric"])

        return formatted_tab



def filter_not_converged_chains(sim, deviation=1.05):
    idata = sim.inferer.idata
    posterior = idata.posterior
    log_likelihood = idata.log_likelihood
    log_likelihood_summed = log_likelihood.to_array("obs")
    log_likelihood_summed = log_likelihood_summed.sum(("time", "id", "obs"))

    # filter non-converged parameter estimates
    likelihood_mask = (
        # compares the mean of the summed log likelihood of a given chain
        log_likelihood_summed.mean("draw") > 
        # to the maximum of all chain means times a factor
        log_likelihood_summed.mean("draw").max() * deviation
    )
    posterior_filtered = posterior.where(likelihood_mask, drop=True)
    log_likelihood_filtered = log_likelihood.where(likelihood_mask, drop=True)

    idata = az.InferenceData(
        posterior=posterior_filtered, 
        log_likelihood=log_likelihood_filtered,
        observed_data=idata.observed_data,
    )

    return idata

def evaluate_posterior(sim, n_samples=10_000):
    idata = sim.inferer.idata
    # idata.posterior = idata.posterior.chunk(chunks={"draw":100}).load()
    # idata.log_likelihood = idata.log_likelihood.chunk(chunks={"draw":100})
    n_subsample = min(
        int(n_samples / idata.posterior.sizes["chain"]), 
        idata.posterior.sizes["draw"]
    )

    if n_subsample < 250:
        warnings.warn(
            "The number of samples drawn from each chain for the pairplot "
            f"({n_subsample}) may be too small to be representative. "
            "Consider increasing n_samples."
        )

    subsamples = np.random.randint(
        0, idata.posterior.sizes["draw"], n_subsample
    )
    idata.posterior = idata.posterior.sel(draw=subsamples)
    idata.log_likelihood = idata.log_likelihood.sel(draw=subsamples)

    log_likelihood_summed = idata.log_likelihood.to_array("obs")
    log_likelihood_summed = log_likelihood_summed.sum(("time", "id", "obs"))

    az.summary(idata.posterior)
    vars = {"k_i":"k_i", "k_m":"k_m", "z_ci":"z_ci", "v_rt":"v_rt", "r_rt":"r_rt", 
            "r_rd":"k_rd", "k_p":"k_p", "z":"z", "kk":"k_k", "h_b":"h_b", 
            "sigma_cint":"σ_cint", "sigma_nrf2":"σ_Nrf2"}
    table = create_table(
        posterior=idata.posterior, 
        error_metric="hdi",
        vars=vars,
    )
    table_latex = table.to_latex(float_format="%.2f")

    log(table_latex, out=f"{sim.output_path}/parameter_table.tex", mode="w")

    # bic 
    msg, _ = bic(idata)
    log(msg=msg, out=f"{sim.output_path}/bic.md", mode="w")

    fig_param = plot_posterior_samples(
        idata.posterior, 
        col_dim="substance", 
        log=True,
        hist_kwargs = dict(hdi=True, bins=20)
    )
    fig_param.set_size_inches(12, 30)
    fig_param.savefig(f"{sim.output_path}/multichain_parameter_estimates.jpg")
    plt.close()

    def plot_pairs(posterior, likelihood):
        parameters = list(posterior.data_vars.keys())

        N = len(parameters)
        parameters_ = parameters.copy()
        fig = plt.figure(figsize=(3*N, 3*(N+1)))
        gs = fig.add_gridspec(N, N+1, width_ratios=[1]*N+[0.2])
        

        i = 0
        while len(parameters_) > 0:
            par_x = parameters_.pop(0)
            hist_ax = gs[i,i].subgridspec(1, 1).subplots()
            plot_hist(
                posterior[par_x].stack(sample=("chain", "draw")), 
                ax=hist_ax, decorate=False, bins=20
            )
            hist_ax.set_title(par_x)
            for j, par_y in enumerate(parameters_, start=i+1):
                ax = gs[j,i].subgridspec(1, 1).subplots()

                scatter = ax.scatter(
                    posterior[par_x], 
                    posterior[par_y], 
                    c=likelihood, 
                    alpha=0.25,
                    s=10,
                    cmap=mpl.colormaps["plasma_r"]
                )

                if j != len(parameters)-1:
                    ax.set_xticks([])
            
                ax.set_xlabel(par_x)            
                ax.set_ylabel(par_y)

            i += 1

        # ax_colorbar = gs[:,N].subgridspec(1, 1).subplots()
        # fig.colorbar(scatter, cax=ax_colorbar)
        return fig

    for substance in idata.posterior.substance.values:
        az.plot_trace(idata.posterior.sel(substance=substance))
        plt.savefig(f"{sim.output_path}/multichain_pseudo_trace_{substance}.jpg")
        plt.close()

        fig = plot_pairs(
            posterior=idata.posterior.sel(substance=substance), 
            likelihood=log_likelihood_summed,
        )
        fig.savefig(f"{sim.output_path}/multichain_pairs_{substance}.jpg")
        plt.close()


def evaluate_posterior_per_cluster(sim):
    idata_ = sim.inferer.idata.copy()
    observations_ = sim.observations.copy()
    scenario_ = sim.scenario

    clusters_ids = np.unique(idata_.posterior.cluster)
    for cid in clusters_ids:
        print_title(f"Evaluating cluster: {cid}", head=False, line="-")
        
        groups = {}
        for key in ["posterior", "log_likelihood"]:
            data = idata_[key]
            data = data.where(data.cluster==cid, drop=True)
            groups.update({key: data})

        idata_cluster = az.InferenceData(**groups, observed_data=idata_.observed_data)
        sim.inferer.idata = idata_cluster
        sim.config.set("case-study", "scenario", f"{sim.scenario}/cluster_{cid}")
        os.makedirs(sim.output_path, exist_ok=True)
        sim.pyabc_posterior_predictions()
        
        with az.style.context(["arviz-darkgrid", "arviz-plasmish"], after_reset=True):
            evaluate_posterior(sim=sim)

        sim.observations = observations_
        sim.config.set("case-study", "scenario", scenario_)


def print_title(title, head=True, line="="):
    print("\n")
    if head:
        print(line * len(title))
    print(title)
    print(line * len(title), end="\n\n")

if __name__ == "__main__":
    
    scenarios = {
        ("tktd_rna_pulse", "rna_pulse_3_6c_substance_independent_rna_protein_module"): "chains_svi",
        ("tktd_rna_pulse", "rna_pulse_3_6c_substance_specific"): "chains_svi_2",
        ("guts", "guts_rna_2"): "chains_svi",
        ("guts", "guts_scaled_damage"): "chains_svi",
        ("guts", "guts_survival"): "chains_nuts"
    }

    combine_chains = False

    for (case_study, scenario), chain_location in scenarios.items():
        print_title(f"Parameter analysis for {scenario}")
        
        # initialize simulation
        config = prepare_casestudy(
            case_study=(case_study, scenario),
            config_file="settings.cfg",
            pkg_dir="case_studies"
        )

        mmod = importlib.import_module(f"{case_study}.mod")
        msim = importlib.import_module(f"{case_study}.sim")
        mprob = importlib.import_module(f"{case_study}.prob")
        mplot = importlib.import_module(f"{case_study}.plot")
        mdat = importlib.import_module(f"{case_study}.data")

        msim.SingleSubstanceSim2.mod = mmod
        msim.SingleSubstanceSim2.dat = mdat
        msim.SingleSubstanceSim2.prob = mprob
        msim.SingleSubstanceSim2.mplot = mplot
        
        sim = msim.SingleSubstanceSim2(config)

        sim.set_inferer("numpyro")

        # The below is only relevant if multichain evaluations are launched.
        # Otherwise the results can be generated with precomputed filtered
        # posteriors
        if combine_chains:
            # Time-consuming. Only execute when necessary
            idata = sim.inferer.combine_chains(
                chain_location=chain_location, 
                drop_extra_vars=["D", "lethality"],
                cluster_deviation="std"
            )
            idata.to_netcdf(f"{sim.output_path}/numpyro_posterior_{chain_location}.nc")

            # evaluate each cluster separately (this can also be done for any posterior)
            sim.inferer.load_results(f"numpyro_posterior_{chain_location}.nc")
            sim.inferer.idata = filter_not_converged_chains(sim, deviation=1.1)
            evaluate_posterior_per_cluster(sim)

            # evaluate completely with filtered chains
            sim.inferer.load_results(f"numpyro_posterior_{chain_location}.nc")
            sim.inferer.idata = filter_not_converged_chains(sim, deviation=1.025)
            sim.inferer.idata.to_netcdf(f"{sim.output_path}/numpyro_posterior_filtered.nc")
        
        sim.inferer.load_results(f"numpyro_posterior_filtered.nc")
        sim.pyabc_posterior_predictions()
        with az.style.context(["arviz-darkgrid", "arviz-plasmish"], after_reset=True):
            evaluate_posterior(sim=sim)

