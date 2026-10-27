import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from driver_code import params

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# -------------------------------
# Load solver output
# -------------------------------
Tall=np.load("Tsol_cell_substrate_interaction.npy") # dimensionless time
Yall=np.load("Ysol_cell_substrate_interaction.npy") # full solution
xd=np.load("xd.npy") # dimensionless space

# -------------------------------
# Parameters
# -------------------------------
# Number of spatial points
N=params['N']
tchar=params['tchar']
Tall_dim=Tall*tchar # convert to seconds

# -------------------------------
# Indices in state vector
# -------------------------------
i=0
cG=Yall[:, i:i+N+1]; i += N+1
cF1=Yall[:, i:i+N+1]; i += N+1
cF2=Yall[:, i:i+N+1]; i += N+1
cSp=Yall[:, i:i+N+1]; i += N+1
cM=Yall[:, i:i+N+1]; i += N+1
cMp=Yall[:, i:i+N+1]; i += N+1
nf=Yall[:, i:i+N+1]; i += N+1
nh=Yall[:, i:i+N+1]; i += N+1
nb1=Yall[:, i:i+N+1]; i += N+1
nb2=Yall[:, i:i+N+1]; i += N+1
nA1=Yall[:, i:i+N+1]; i += N+1
nA2=Yall[:, i:i+N+1]; i += N+1
ns1=Yall[:, i:i+N+1]; i += N+1
ns2=Yall[:, i:i+N+1]; i += N+1
cR=Yall[:, i:i+N+1]; i += N+1
cRA=Yall[:, i:i+N+1]; i += N+1
cMLCP=Yall[:, i:i+N+1]; i += N+1
cMLCPP=Yall[:, i:i+N+1]; i += N+1
cMLCK=Yall[:, i:i+N+1]; i += N+1
cMLCKP=Yall[:, i:i+N+1]; i += N+1
cC=Yall[:, i:i+N+1]; i += N+1
cCP=Yall[:, i:i+N+1]; i += N+1
u=Yall[:, i:i+N+3]; i += N+3
w=Yall[:, i:i+N+3]; i += N+3
sigma_cell=Yall[:, i:i+N+1]

nb=nb1+nb2 # bound integrins
nA=nA1+nA2 # focal adhesions

# -------------------------------
# Strain from displacement (ghost nodes at 0 and N+2)
# -------------------------------
dx=xd[1]-xd[0]

def strain_from_u(u):

    Nt=u.shape[0]
    du_dx=np.zeros((Nt, N+1))

    du_dx[:, 1:-1]=(u[:, 3:N+2]-u[:, 1:N])/(2*dx) # interior nodes
    du_dx[:,0]=(u[:,2]-u[:,1])/dx # left physical boundary (i=1)
    du_dx[:, -1]=(u[:,N+1]-u[:,N])/dx # right physical boundary (i=N+1)

    return du_dx

strain=strain_from_u(u)

# -------------------------------
# Spatial interpolants (physical domain only)
# -------------------------------
u_interp=interp1d(xd, u[:,1:N+2], axis=1, bounds_error=False, fill_value="extrapolate")
strain_interp=interp1d(xd, strain, axis=1, bounds_error=False, fill_value="extrapolate")
nb_interp=interp1d(xd, nb, axis=1, bounds_error=False, fill_value="extrapolate")
nA_interp=interp1d(xd, nA, axis=1, bounds_error=False, fill_value="extrapolate")
cRA_interp=interp1d(xd, cRA, axis=1, bounds_error=False, fill_value="extrapolate")
cSp_interp=interp1d(xd, cSp, axis=1, bounds_error=False, fill_value="extrapolate")

# -------------------------------
# Plotting
# -------------------------------
x_plot=[0.25, 0.5]
linestyles={0.25: "--", 0.5: "-"}

fig, axs = plt.subplots(2, 2, figsize=(8, 5), sharex=True)

# Font sizes (match other figure)
title_fs=14
label_fs=18
legend_fs=8
tick_fs=16

# ===============================
# (a) displacement
# ===============================
colors_a = {0.25: "tab:blue", 0.5: "tab:blue"}

for xloc in x_plot:
    axs[0,0].plot(Tall_dim, u_interp(xloc),
                  linestyle=linestyles[xloc],
                  color=colors_a[xloc],
                  label=rf"$u(x={xloc})$")

axs[0,0].set_ylabel(r"$u(x,t)$", fontsize=label_fs)
axs[0,0].set_title(r"(a)", fontsize=title_fs)
axs[0,0].set_xlim(0, 5000)
axs[0,0].legend(fontsize=legend_fs, frameon=True, edgecolor="black")

# ===============================
# (b) bound integrins + focal adhesions
# ===============================
colors_b = {"nb": "tab:red", "nA": "tab:green"}

for xloc in x_plot:
    axs[0,1].plot(Tall_dim, nb_interp(xloc),
                  linestyle=linestyles[xloc],
                  color=colors_b["nb"],
                  label=rf"$n_b(x={xloc})$")

    axs[0,1].plot(Tall_dim, nA_interp(xloc),
                  linestyle=linestyles[xloc],
                  color=colors_b["nA"],
                  label=rf"$n_A(x={xloc})$")

axs[0,1].set_ylabel(r"$n_b(x,t),\, n_A(x,t)$", fontsize=label_fs)
axs[0,1].set_title(r"(b)", fontsize=title_fs)
axs[0,1].set_xlim(0, 5000)
axs[0,1].legend(fontsize=legend_fs, frameon=True, edgecolor="black", ncol=2)

# ===============================
# Early-time inset (0-250 seconds)
# ===============================
t_zoom=250.0

# Inset position [left, bottom, width, height]
# Coordinates are relative to panel (b)
axins=axs[0,1].inset_axes([0.60, 0.32, 0.35, 0.36])

early_values=[]

for xloc in x_plot:

    nb_values=np.asarray(nb_interp(xloc))
    nA_values=np.asarray(nA_interp(xloc))

    # Extract solver output during the first 250 seconds
    mask=Tall_dim<=t_zoom

    t_early=Tall_dim[mask]
    nb_early=nb_values[mask]
    nA_early=nA_values[mask]

    if t_early[-1]<t_zoom:

        t_early=np.append(t_early, t_zoom)

        nb_early=np.append(nb_early, np.interp(t_zoom, Tall_dim, nb_values))

        nA_early=np.append(nA_early, np.interp(t_zoom, Tall_dim, nA_values))

    # Bound integrins
    axins.plot(t_early, nb_early, linestyle=linestyles[xloc], color=colors_b["nb"], linewidth=1.1)

    # Mature focal adhesions
    axins.plot(t_early, nA_early, linestyle=linestyles[xloc], color=colors_b["nA"], linewidth=1.1)

    early_values.extend([nb_early, nA_early])

ymin=min(np.min(v) for v in early_values)
ymax=max(np.max(v) for v in early_values)
yrange=max(ymax-ymin, 1e-8)

axins.set_xlim(0, t_zoom)
axins.set_ylim(min(0,ymin)-0.05*yrange, ymax+0.10*yrange)

axins.set_xticks([0, 125, 250])
axins.tick_params(axis="both", labelsize=7, direction="in", pad=2)

axins.set_facecolor("white")
axins.set_zorder(5)

for spine in axins.spines.values():
    spine.set_linewidth(0.9)
    spine.set_color("black")

# ===============================
# (c) activated ROCK
# ===============================
colors_c = {0.25: "tab:purple", 0.5: "tab:purple"}

for xloc in x_plot:
    axs[1,0].plot(Tall_dim, cRA_interp(xloc),
                  linestyle=linestyles[xloc],
                  color=colors_c[xloc],
                  label=rf"$x={xloc}$")

axs[1,0].set_xlabel(r"$t\;(\mathrm{s})$", fontsize=label_fs)
axs[1,0].set_ylabel(r"$c_{R}^{+}(x,t)$", fontsize=label_fs)
axs[1,0].set_title(r"(c)", fontsize=title_fs)
axs[1,0].set_xlim(0, 5000)
axs[1,0].legend(fontsize=legend_fs, frameon=True, edgecolor="black")

# ===============================
# (d) stress fibres
# ===============================
colors_d = {0.25: "tab:olive", 0.5: "tab:olive"}

for xloc in x_plot:
    axs[1,1].plot(Tall_dim, cSp_interp(xloc),
                  linestyle=linestyles[xloc],
                  color=colors_d[xloc],
                  label=rf"$x={xloc}$")

axs[1,1].set_xlabel(r"$t\;(\mathrm{s})$", fontsize=label_fs)
axs[1,1].set_ylabel(r"$c_{S}^{+}(x,t)$", fontsize=label_fs)
axs[1,1].set_title(r"(d)", fontsize=title_fs)
axs[1,1].set_xlim(0, 5000)
axs[1,1].legend(fontsize=legend_fs, frameon=True, edgecolor="black")

# -------------------------------
# Main axis tick formatting
# -------------------------------
for ax in axs.flat:
    ax.tick_params(labelsize=tick_fs)

# -------------------------------
# Formatting
# -------------------------------
plt.tight_layout()

# -------------------------------
# Save figure
# -------------------------------
plt.savefig("cell_substrate_temporal_panels.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()
