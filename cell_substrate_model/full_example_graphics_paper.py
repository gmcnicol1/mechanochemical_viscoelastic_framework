import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# -------------------------------
# Load solver output
# -------------------------------

Tall = np.load("Tsol_cell_substrate_interaction.npy")
Yall = np.load("Ysol_cell_substrate_interaction.npy")
xd = np.load("xd.npy")

# -------------------------------
# Domain / discretisation
# -------------------------------
N=len(xd)-1

# -------------------------------
# Extract variables from Yall
# -------------------------------

i=0
cG_all=Yall[:, i:i+N+1]; i += N+1
cF1_all=Yall[:, i:i+N+1]; i += N+1
cF2_all=Yall[:, i:i+N+1]; i += N+1
cSp_all=Yall[:, i:i+N+1]; i += N+1
cM_all=Yall[:, i:i+N+1]; i += N+1
cMp_all=Yall[:, i:i+N+1]; i += N+1
nf_all=Yall[:, i:i+N+1]; i += N+1
nh_all=Yall[:, i:i+N+1]; i += N+1
nb1_all=Yall[:, i:i+N+1]; i += N+1
nb2_all=Yall[:, i:i+N+1]; i += N+1
nA1_all=Yall[:, i:i+N+1]; i += N+1
nA2_all=Yall[:, i:i+N+1]; i += N+1
ns1_all=Yall[:, i:i+N+1]; i += N+1
ns2_all=Yall[:, i:i+N+1]; i += N+1
cR_all=Yall[:, i:i+N+1]; i += N+1
cRA_all=Yall[:, i:i+N+1]; i += N+1
cMLCP_all=Yall[:, i:i+N+1]; i += N+1
cMLCPP_all=Yall[:, i:i+N+1]; i += N+1
cMLCK_all=Yall[:, i:i+N+1]; i += N+1
cMLCKP_all=Yall[:, i:i+N+1]; i += N+1
cC_all=Yall[:, i:i+N+1]; i += N+1
cCP_all=Yall[:, i:i+N+1]; i += N+1

u_all=Yall[:, i:i+N+3]; i += N+3
w_all=Yall[:, i:i+N+3]; i += N+3
sigma_cell_all=Yall[:, i:i+N+1]; i += N+1
sigma_ECM_all=Yall[:, i:i+N+1]

# -------------------------------
# Convert time back to dimensional seconds
# -------------------------------

from driver_code import params
tchar=params["tchar"]

Tall_dim=Tall*tchar

# -------------------------------
# Times to plot (seconds)
# -------------------------------

times_to_plot = [0, 250, 500, 1000, 2500, 5000]

# Interpolants in time
u_interp = interp1d(Tall_dim, u_all[:,1:-1], axis=0, kind="linear", fill_value="extrapolate")
sigma_interp = interp1d(Tall_dim, sigma_cell_all, axis=0, kind="linear", fill_value="extrapolate")
nA_interp = interp1d(Tall_dim, nA1_all + nA2_all, axis=0, kind="linear", fill_value="extrapolate")
cSp_interp = interp1d(Tall_dim, cSp_all, axis=0, kind="linear", fill_value="extrapolate")

## -------------------------------
# 4-panel figure: cell–substrate model 
# -------------------------------

fig, axs = plt.subplots(2, 2, figsize=(8, 5), sharex=True)

title_fs = 14
label_fs = 18
legend_fs = 8

# Fix a colour per time (Matplotlib default cycle)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
time_colors = {t: color_cycle[i % len(color_cycle)] for i, t in enumerate(times_to_plot)}

# -------------------------------
# (a) Cell deformation
# -------------------------------
for t in times_to_plot:
    axs[0, 0].plot(xd, u_interp(t), color=time_colors[t], label=rf"$t={t}\,\mathrm{{s}}$")

axs[0, 0].set_ylabel(r"$u(x,t)$", fontsize=label_fs)
axs[0, 0].set_title(r"(a)", fontsize=title_fs)
axs[0, 0].legend(loc="upper right", fontsize=legend_fs,
                 frameon=True, framealpha=1.0, edgecolor="black", ncol=2)
axs[0, 0].set_xlim(xd[0], xd[-1])

# -------------------------------
# (b) Cell stress
# -------------------------------
for t in times_to_plot:
    axs[0, 1].plot(xd, sigma_interp(t), color=time_colors[t])

axs[0, 1].set_ylabel(r"$\sigma_{c}(x,t)$", fontsize=label_fs)
axs[0, 1].set_title(r"(b)", fontsize=title_fs)
axs[0, 1].set_xlim(xd[0], xd[-1])

# -------------------------------
# (c) Focal adhesions
# -------------------------------
for t in times_to_plot:
    axs[1, 0].plot(xd, nA_interp(t), color=time_colors[t])

axs[1, 0].set_xlabel(r"$x$", fontsize=label_fs)
axs[1, 0].set_ylabel(r"$n_A(x,t)$", fontsize=label_fs)
axs[1, 0].set_title(r"(c)", fontsize=title_fs)
axs[1, 0].set_xlim(xd[0], xd[-1])

# -------------------------------
# (d) Stress fibres
# -------------------------------
for t in times_to_plot:
    axs[1, 1].plot(xd, cSp_interp(t), color=time_colors[t])

axs[1, 1].set_xlabel(r"$x$", fontsize=label_fs)
axs[1, 1].set_ylabel(r"$c_{S}^{+}(x,t)$", fontsize=label_fs)
axs[1, 1].set_title(r"(d)", fontsize=title_fs)
axs[1, 1].set_xlim(xd[0], xd[-1])

# Tick label size (match KV panels)
for ax in axs.flat:
    ax.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig("cell_substrate_spatial_panels.pdf", dpi=300, bbox_inches="tight")
plt.show()
