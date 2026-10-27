import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from driver_code import params

# Graphics driver for example model

# -------------------------------
# Load solver output
# -------------------------------

Tall_KV=np.load("Tall_KV.npy") # full time vector
Yall_KV=np.load("Yall_KV.npy") # full solution array
Tall_SLS_Kelvin=np.load("Tall_SLS_Kelvin.npy") 
Yall_SLS_Kelvin=np.load("Yall_SLS_Kelvin.npy") 
Tall_SLS_Maxwell=np.load("Tall_SLS_Maxwell.npy") 
Yall_SLS_Maxwell=np.load("Yall_SLS_Maxwell.npy") 
Tall_Maxwell=np.load("Tall_Maxwell.npy") 
Yall_Maxwell=np.load("Yall_Maxwell.npy") 
xd=np.load("xd.npy") # spatial grid

# Number of spatial points
N=params['N']

# Separate cell density and ECM displacement
ca_all_KV=Yall_KV[:,0:N+1]
cb_all_KV=Yall_KV[:,N+1:2*N+2]
u_all_KV=Yall_KV[:,2*N+2:3*N+3]
sigma_all_KV=Yall_KV[:,3*N+3:4*N+4]

ca_all_SLS_Kelvin=Yall_SLS_Kelvin[:,0:N+1]
cb_all_SLS_Kelvin=Yall_SLS_Kelvin[:,N+1:2*N+2]
u_all_SLS_Kelvin=Yall_SLS_Kelvin[:,2*N+2:3*N+3]
sigma_all_SLS_Kelvin=Yall_SLS_Kelvin[:,3*N+3:4*N+4]

ca_all_SLS_Maxwell=Yall_SLS_Maxwell[:,0:N+1]
cb_all_SLS_Maxwell=Yall_SLS_Maxwell[:,N+1:2*N+2]
u_all_SLS_Maxwell=Yall_SLS_Maxwell[:,2*N+2:3*N+3]
sigma_all_SLS_Maxwell=Yall_SLS_Maxwell[:,3*N+3:4*N+4]

ca_all_Maxwell=Yall_Maxwell[:,0:N+1]
cb_all_Maxwell=Yall_Maxwell[:,N+1:2*N+2]
u_all_Maxwell=Yall_Maxwell[:,2*N+2:3*N+3]
sigma_all_Maxwell=Yall_Maxwell[:,3*N+3:4*N+4]

plt.rcParams.update({"text.usetex":True,"font.family":"serif","axes.labelsize":24,"xtick.labelsize":20,"ytick.labelsize":20,
                     "legend.fontsize":12,"figure.figsize":(6,4),"axes.titlesize":12})

# -------------------------------
# Fixed spatial locations to track
# -------------------------------

u_space_interp_KV=interp1d(xd,u_all_KV,axis=1,kind="linear",fill_value="extrapolate") # interpolate to desired domain locations
sigma_space_interp_KV=interp1d(xd,sigma_all_KV,axis=1,kind="linear",fill_value="extrapolate")
ca_space_interp_KV=interp1d(xd,ca_all_KV,axis=1,kind="linear",fill_value="extrapolate")
cb_space_interp_KV=interp1d(xd,cb_all_KV,axis=1,kind="linear",fill_value="extrapolate")

u_space_interp_SLS_Maxwell=interp1d(xd,u_all_SLS_Maxwell,axis=1,kind="linear",fill_value="extrapolate") # interpolate to desired domain locations
sigma_space_interp_SLS_Maxwell=interp1d(xd,sigma_all_SLS_Maxwell,axis=1,kind="linear",fill_value="extrapolate")
ca_space_interp_SLS_Maxwell=interp1d(xd,ca_all_SLS_Maxwell,axis=1,kind="linear",fill_value="extrapolate")
cb_space_interp_SLS_Maxwell=interp1d(xd,cb_all_SLS_Maxwell,axis=1,kind="linear",fill_value="extrapolate")

u_space_interp_SLS_Kelvin=interp1d(xd,u_all_SLS_Kelvin,axis=1,kind="linear",fill_value="extrapolate") # interpolate to desired domain locations
sigma_space_interp_SLS_Kelvin=interp1d(xd,sigma_all_SLS_Kelvin,axis=1,kind="linear",fill_value="extrapolate")
ca_space_interp_SLS_Kelvin=interp1d(xd,ca_all_SLS_Kelvin,axis=1,kind="linear",fill_value="extrapolate")
cb_space_interp_SLS_Kelvin=interp1d(xd,cb_all_SLS_Kelvin,axis=1,kind="linear",fill_value="extrapolate")


u_space_interp_Maxwell=interp1d(xd,u_all_Maxwell,axis=1,kind="linear",fill_value="extrapolate") # interpolate to desired domain locations
sigma_space_interp_Maxwell=interp1d(xd,sigma_all_Maxwell,axis=1,kind="linear",fill_value="extrapolate")
ca_space_interp_Maxwell=interp1d(xd,ca_all_Maxwell,axis=1,kind="linear",fill_value="extrapolate")
cb_space_interp_Maxwell=interp1d(xd,cb_all_Maxwell,axis=1,kind="linear",fill_value="extrapolate")


rheologies=["KV", "Maxwell", "SLS_Kelvin", "SLS_Maxwell"]
colors={"KV":"tab:red", "Maxwell":"tab:blue", "SLS_Kelvin":"tab:green", "SLS_Maxwell":"tab:purple"}
x_plot=[0.5, 0.9]
linestyles={0.5: "--", 0.9: "-"}

# -------------------------------
# Strain function
# -------------------------------
def strain_from_u(u, dx):
    du_dx = np.zeros_like(u)
    du_dx[1:-1]=(u[2:]-u[:-2])/(2*dx)
    du_dx[0]=(u[1]-u[0])/dx
    du_dx[-1]=(u[-1]-u[-2])/dx
    return du_dx

dx=xd[1]-xd[0]

# -------------------------------
# Build spatial interpolants per rheology
# -------------------------------
interp_dict = {}
Tall_dict = {}

for rheo in rheologies:
    u_all = globals()[f"u_all_{rheo}"]       # shape (Nt, Nx)
    sigma_all = globals()[f"sigma_all_{rheo}"]
    ca_all = globals()[f"ca_all_{rheo}"]
    cb_all = globals()[f"cb_all_{rheo}"]
    Tall = globals()[f"Tall_{rheo}"]

    # Create interpolators in space for each variable
    interp_dict[rheo] = {
        "u": interp1d(xd, u_all.T, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate"),
        "sigma": interp1d(xd, sigma_all.T, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate"),
        "c_a": interp1d(xd, ca_all.T, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate"),
        "c_b": interp1d(xd, cb_all.T, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate"),
    }
    Tall_dict[rheo] = Tall

# -------------------------------
# 4-panel figure
# -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
title_fs, label_fs, legend_fs = 14, 16, 8

# (a) displacement
for rheo in rheologies:
    for xloc in x_plot:
        axs[0,0].plot(Tall_dict[rheo], interp_dict[rheo]["u"](xloc), color=colors[rheo], linestyle=linestyles[xloc])
axs[0,0].set_ylabel(r"$u(x,t)$", fontsize=label_fs)
axs[0,0].set_title(r"(a)", fontsize=title_fs)
axs[0, 0].set_xlim(0, 10)

# (b) strain
for rheo in rheologies:
    u_all = globals()[f"u_all_{rheo}"]
    times = Tall_dict[rheo]
    for xloc in x_plot:
        eps_t = [np.interp(xloc, xd, strain_from_u(u_all[ti,:], dx)) for ti in range(len(times))]
        axs[0,1].plot(times, eps_t, color=colors[rheo], linestyle=linestyles[xloc])
axs[0,1].set_ylabel(r"$\partial_x u(x,t)$", fontsize=label_fs)
axs[0,1].set_title(r"(b)", fontsize=title_fs)
axs[0, 0].set_xlim(0, 10)

# -------------------------------
# Create a legend for rheology colors only
# -------------------------------
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="tab:red", lw=2, label="Kelvin-Voigt"),
    Line2D([0], [0], color="tab:blue", lw=2, label="Maxwell"),
    Line2D([0], [0], color="tab:green", lw=2, label="SLS (Kelvin)"),
    Line2D([0], [0], color="tab:purple", lw=2, label="SLS (Maxwell)")
]

# Place legend in bottom left of panel (b)
axs[0,1].legend(handles=legend_elements, loc="upper left", fontsize=8, frameon=True, framealpha=1.0, edgecolor="black")


# (c) stress
for rheo in rheologies:
    for xloc in x_plot:
        axs[1,0].plot(Tall_dict[rheo], interp_dict[rheo]["sigma"](xloc), color=colors[rheo], linestyle=linestyles[xloc])
axs[1,0].set_xlabel(r"$t$", fontsize=label_fs)
axs[1,0].set_ylabel(r"$\sigma(x,t)$", fontsize=label_fs)
axs[1,0].set_title(r"(c)", fontsize=title_fs)
axs[0, 0].set_xlim(0, 10)

# (d) chemical B
for rheo in rheologies:
    for xloc in x_plot:
        axs[1,1].plot(Tall_dict[rheo], interp_dict[rheo]["c_b"](xloc), color=colors[rheo], linestyle=linestyles[xloc])
axs[1,1].set_xlabel(r"$t$", fontsize=label_fs)
axs[1,1].set_ylabel(r"$c_b(x,t)$", fontsize=label_fs)
axs[1,1].set_title(r"(d)", fontsize=title_fs)
axs[0, 0].set_xlim(0, 10)

for ax in axs.flat:
    ax.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig("Comparison.pdf", dpi=300, bbox_inches="tight")
plt.show()
