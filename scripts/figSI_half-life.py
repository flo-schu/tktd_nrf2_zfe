import numpy as np
import arviz as az
from scipy.stats import lognorm
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from pymob.utils.store_file import prepare_casestudy

from method_posterior_analysis import log

halflife = lambda y0, r, t: y0 * np.exp(-r * t)

scenario = "rna_pulse_3_6c_substance_independent_rna_protein_module"
# initialize simulation
config = prepare_casestudy(
    case_study=("tktd_rna_pulse", scenario),
    config_file="settings.cfg",
    pkg_dir="case_studies"
)
from sim import SingleSubstanceSim2
sim = SingleSubstanceSim2(config)
sim.set_inferer("numpyro")

sim.inferer.load_results(f"numpyro_posterior_filtered.nc")


# half life RNA
rna_decay = sim.inferer.idata.posterior.r_rd
rna_decay

log(az.hdi(rna_decay), "results/log_rna_halflife_hdi.txt" )

y0 = 10
r = 1
t = np.linspace(0, 10, 1000)

fig, (ax, ax_hist) = plt.subplots(2,1, sharex=True)

HL = []
i = 0
for d in rna_decay.draw:
    for c in rna_decay.chain:
        r = rna_decay.sel(chain=c, draw=d)
        y = halflife(y0, float(r), t)
        ax.plot(t, y, alpha=.01, color="black")
        hl = t[np.argmin(np.abs(y - y0/2))]
        HL.append(hl)
        
        i += 1
        # if i > 1000:
        #     break

ax.set_ylabel("RNA level")
ax.set_yticks([0, 5, 10])
ax.set_yticklabels(["0%", "50%", "100%"])
ax.set_ylim(0,y0)


lnpars = lognorm.fit(HL, floc=0)
p_hl = lognorm(*lnpars).pdf(t)

log(f"Mode Half-life: {t[np.argmax(p_hl)] * 60} min", "results/log_rna_halflife_hdi.txt", mode="a")


_ = ax_hist.plot(t, p_hl, color="black")
hist_, bins_, _ = ax_hist.hist(HL, bins=90, density=True)
ax_hist.vlines(20/60, 0, p_hl.max() *1.1, color="tab:red")
ax_hist.set_xlabel("")
ax_hist.set_xticks([ 20/60, 1, 2, 3, 4, 5])
ax_hist.set_xticklabels([ "20 min", "1 hour", "2 hours","","4 hours","" ])
ax_hist.set_xlim(0,3)
ax_hist.set_ylim(0, p_hl.max() *1.1)
ax.set_title("RNA expression level")
ax_hist.set_title("RNA Half life")
ax_hist.set_ylabel("Pr(Half-life)")
fig.savefig("results/fig_si_halflife_complete.png")

# half life Protein
k_d = sim.inferer.idata.posterior.k_p
k_d

log(az.hdi(k_d), "results/log_protein_halflife_hdi.txt", mode="w")

y0 = 10
r = 1
t = np.linspace(0, 300, 1000)

fig, (ax, ax_hist) = plt.subplots(2,1, sharex=True)

HL = []
i = 0
for d in k_d.draw:
    for c in k_d.chain:
        r = k_d.sel(chain=c, draw=d)
        y = halflife(y0, float(r), t)
        ax.plot(t, y, alpha=.01, color="black")
        hl = t[np.argmin(np.abs(y - y0/2))]
        HL.append(hl)
        
        i += 1
        # if i > 1000:
        #     break

ax.set_ylabel("Protein level")
ax.set_yticks([0, 5, 10])
ax.set_yticklabels(["0%", "50%", "100%"])
ax.set_ylim(0,y0)


lnpars = lognorm.fit(HL, floc=0)
p_hl = lognorm(*lnpars).pdf(t)

log(f"Mode Half-life: {t[np.argmax(p_hl)]} h", "results/log_protein_halflife_hdi.txt", mode="a")

hist_, bins_, _ = ax_hist.hist(HL, bins=90, density=True)
y_lim = p_hl.max() *1.1
ax_hist.plot(t, p_hl, color="black")
p = ax_hist.add_patch(Rectangle((20, 0), 46-20, y_lim, fill=True, facecolor="tab:red", alpha=0.3))
ax_hist.set_xlabel("")
ax_hist.set_xticks([24, 48, 96])
ax_hist.set_xticklabels([ "24 hours", "48 hours", "96 hours" ])
ax_hist.set_xlim(0,120)
ax_hist.set_ylim(0, y_lim)
ax.set_title("Protein level")
ax_hist.set_title("Protein Half life")
ax_hist.set_ylabel("Pr(Half-life)")
fig.savefig("results/fig_si_halflife_complete_protein.png")