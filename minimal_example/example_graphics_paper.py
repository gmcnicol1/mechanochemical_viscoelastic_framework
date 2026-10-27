import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from driver_code import params

# Graphics driver for example model

# -------------------------------
# Load solver output
# -------------------------------

mechanics_model="KV"  #  choose the mechanical model to post-process


Tall=np.load(f"Tall_{mechanics_model}.npy")   # full time vector
Yall=np.load(f"Yall_{mechanics_model}.npy")   # full solution array
xd=np.load("xd.npy") # spatial grid

# Number of spatial points
N=params['N']

# Separate cell density and ECM displacement
ca_all=Yall[:,0:N+1]
cb_all=Yall[:,N+1:2*N+2]
u_all=Yall[:,2*N+2:3*N+3]
sigma_all=Yall[:,3*N+3:4*N+4]

plt.rcParams.update({"text.usetex":True,"font.family":"serif","axes.labelsize":24,"xtick.labelsize":20,"ytick.labelsize":20,
                     "legend.fontsize":12,"figure.figsize":(6,4),"axes.titlesize":12})

# -------------------------------
# Define exact times to plot
# -------------------------------

times_to_plot=[0.5,1,2.5,5,10]

# Time interpolants for smooth plotting
ca_interp=interp1d(Tall,ca_all,axis=0,kind="linear",fill_value="extrapolate")
cb_interp=interp1d(Tall,cb_all,axis=0,kind="linear",fill_value="extrapolate")
u_interp=interp1d(Tall,u_all,axis=0,kind="linear",fill_value="extrapolate")
sigma_interp=interp1d(Tall,sigma_all,axis=0,kind="linear",fill_value="extrapolate")


# -------------------------------
# 4-panel figure: u, strain, sigma, c_a & c_b (consistent colours by time)
# -------------------------------

# Compute strain from displacement (central difference)
dx=xd[1]-xd[0]
def strain_from_u(u):
    du_dx=np.zeros_like(u)
    du_dx[1:-1]=(u[2:]-u[:-2])/(2*dx)
    du_dx[0]=(u[1]-u[0])/dx
    du_dx[-1]=(u[-1]-u[-2])/dx
    return du_dx

fig, axs = plt.subplots(2, 2, figsize=(7, 4), sharex=True)

title_fs=14
label_fs=16
legend_fs=8

# Fix a colour per time (Matplotlib default cycle)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
time_colors = {t: color_cycle[i % len(color_cycle)] for i, t in enumerate(times_to_plot)}

# (a) displacement u(x,t)
for t in times_to_plot:
    axs[0, 0].plot(xd, u_interp(t), color=time_colors[t], label=rf"$t={t}$")
axs[0, 0].set_ylabel(r"$u(x,t)$", fontsize=label_fs)
axs[0, 0].set_title(r"(a)", fontsize=title_fs)
axs[0, 0].legend(loc="upper left", fontsize=legend_fs,
                 frameon=True, framealpha=1.0, edgecolor="black", ncol=2)
axs[0, 0].set_xlim(0, 1)
axs[0, 0].set_ylim(-0.05, 0.3)

# (b) strain du/dx
for t in times_to_plot:
    eps = strain_from_u(u_interp(t))
    axs[0, 1].plot(xd, eps, color=time_colors[t])
axs[0, 1].set_ylabel(r"$\partial_x u(x,t)$", fontsize=label_fs)
axs[0, 1].set_title(r"(b)", fontsize=title_fs)
axs[0, 1].set_xlim(0, 1)

# (c) stress sigma(x,t)
for t in times_to_plot:
    axs[1, 0].plot(xd, sigma_interp(t), color=time_colors[t])
axs[1, 0].set_xlabel(r"$x$", fontsize=label_fs)
axs[1, 0].set_ylabel(r"$\sigma(x,t)$", fontsize=label_fs)
axs[1, 0].set_title(r"(c)", fontsize=title_fs)
axs[1, 0].set_xlim(0, 1)

# (d) concentrations c_a (solid) and c_b (dashed)
for t in times_to_plot:
    axs[1, 1].plot(xd, ca_interp(t), color=time_colors[t], linestyle="-")
    axs[1, 1].plot(xd, cb_interp(t), color=time_colors[t], linestyle="--")
axs[1, 1].set_xlabel(r"$x$", fontsize=label_fs)
axs[1, 1].set_ylabel(r"$c_a(x,t),\, c_b(x,t)$", fontsize=label_fs)
axs[1, 1].set_title(r"(d)", fontsize=title_fs)
axs[1, 1].set_xlim(0, 1)

# Tick label size
for ax in axs.flat:
    ax.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig("KV_panels.pdf", dpi=300, bbox_inches="tight")
plt.show()
