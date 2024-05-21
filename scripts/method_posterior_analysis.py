import numpy as np
import matplotlib.pyplot as plt
from pymob.utils.store_file import prepare_casestudy, parse_config_section
from matplotlib import pyplot as plt
import arviz as az
from pymob.utils.plot_helpers import plot_loghist

def log(msg, out, newlines=1, mode="a"):
    with open(out, mode) as f:
        print(msg, file=f, end="\n")
        for _ in range(newlines):
            print("", file=f, end="\n")

def postprocess_posterior(sim):
    idata = sim.inferer.idata
    if not hasattr(idata, "posterior_predictive"):
        # this is just a hot-fix. Make sure that the posterior returned by the
        # backend is correct. End of story
        sim.inferer.idata.add_groups({
            "posterior_predictive": idata.posterior[sim.data_variables]
        })
        idata.posterior = idata.posterior_predictive.drop(
            sim.data_variables + list(sim.coordinates.keys())
        )

    ll = sim.inferer.idata.log_likelihood
    obs = sim.inferer.idata.observed_data
    obs_idx = {f"{k}_obs":np.where(~np.isnan(obs[k])) for k in list(obs.data_vars.keys())}

    log_lik_vars = list(sim.inferer.idata.log_likelihood.data_vars.keys())
    for v in log_lik_vars: 
        ll_new = ll[v].values[:,:,*obs_idx[v]]
        idata.log_likelihood = idata.log_likelihood.drop(v)
        idata.log_likelihood[v] = (("chain", "draw", f"{v}_sample"), ll_new)

    return idata


def bic(idata: az.InferenceData):
    """calculate the BIC for az.InferenceData. The function will average over
    all samples from the markov chain
    """
    log_likelihood = idata.log_likelihood.mean(("chain", "draw")).sum().to_array().sum()
    k = idata.posterior.mean(("chain", "draw")).count().to_array().sum()

    vars = [i.split("_")[0] for i in list(idata.log_likelihood.data_vars.keys())]
    n = (~idata.observed_data[vars].isnull()).sum().to_array().sum()

    bic = float(k * np.log(n) - 2 * log_likelihood)
    msg = str(
        "Bayesian Information Criterion (BIC)"
        "\n===================================="
        f"\nParameters: {int(k)}"
        f"\nData points: {int(n)}"
        f"\nLog-likelihood: {float(log_likelihood)}"
        f"\nBIC: {bic}"
    )

    return msg, bic

# Log Model comparison
if __name__ == "__main__":
    # model = "rna_pulse_3_1"
    # model = "guts_rna"
    model = "guts_scaled_damage"
    out_tab = f"case_studies/reversible_damage/results/model_comparison/{model}.md"
    log(f"# Model comparison for '{model}'", mode="w")

    scenarios = {
        "rna_pulse_3_1": [
            "rna_pulse_3_1_independent", 
            "rna_pulse_3_1_independent_except_rna", 
            "rna_pulse_3_1_combined"
        ],
        "guts_rna": [
            "guts_rna",
            "guts_rna_combined",
        ],
        "guts_scaled_damage": [
            "guts_scaled_damage"
        ],

    }

    idatas = []

    for s in scenarios[model]:
        config = prepare_casestudy(
            case_study=("reversible_damage", s),
            config_file="settings.cfg",
            pkg_dir="case_studies"
        )
        from sim import SingleSubstanceSim
        sim = SingleSubstanceSim(config)

        sim.set_inferer("numpyro")
        sim.inferer.load_results()
        idata = postprocess_posterior(sim)
        msg, bic_value = bic(idata)
        log(f"## Scenario: {s}")

        log("Arviz Summary", newlines=0)
        log("=============")
        log(az.summary(idata))

        log(msg)

        log("Leave-one-out (LOO) cross validation", newlines=0)
        log("==============================")
        logliks = list(idata.log_likelihood.data_vars.keys())
        for v in logliks:
            log(f"### LOO: {v}")
            loo = az.loo(idata, var_name=v)
            log(loo)

        idatas.append(idata)

