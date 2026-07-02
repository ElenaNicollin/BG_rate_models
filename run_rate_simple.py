import numpy as np
import matplotlib.pyplot as plt

from rate_model import *
from rate_analysis import *


def run_and_plot(G_dict, skip=0, *args, **kwargs):

    rates = run_rate(*args, G_dict=G_dict, return_all=True, **kwargs)

    ### PLOTS ###
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(f"species: {species}, noise: {noise_method}, all_to_all: {all_to_all}, t_sim: {t_sim}")

    ax11 = fig.add_subplot(3,3,7)
    ax11 = plot_table(G_dict)
    ax12 = fig.add_subplot(3,1,3)
    ax12 = plot_full_table(G_dict, pops)

    axrate = fig.add_subplot(3,1,1)
    plot_rate(rates, t_sim, dt=1e-4, skip=skip, ax=axrate)

    axfft = fig.add_subplot(3,3,(4,5))
    plot_fft({k:np.mean(v, axis=0) for k,v in rates.items()}, dt, skip=skip, window_size=2, ax=axfft)
    axfft.set_xlim(0,100)
    
    pops_sublist = ["Proto", "Arky", "STN", "D1", "D2"]
    axphase = fig.add_subplot(3,3,6, projection='polar')
    plot_relative_phase_angles(rates, pops_sublist, "Proto", dt, window_size=1, skip=skip, n_pairs=2000, freq_method="fft", normalize=False, ax=axphase)

    plt.tight_layout()
    return fig


#####################################

pops = ["Th", "Ctx", "FSI", "D1", "D2", "STN", "Arky", "Proto", "GPi"]

species = "rat"
params_file = f"/home/elena/Documents/code/params/{species}_pop_params.json"
n_model = 1000
skip = 0.1 if species=="rat" else 0.5
t_sim = 2+skip
dt = 1e-4
n_steps = int(round(t_sim/dt, 0))
all_to_all = False
noise_method = "Ornstein-Uhlenbeck" #can be None, "Gaussian", or "Ornstein-Uhlenbeck"
noise_variance = "auto"

G_default = dict(G_Proto_to_STN=7.26, G_STN_to_Proto=0.36, G_Proto_to_Proto=0.6,
                 G_Proto_to_Arky=0.73, G_Arky_to_D2=0.1, G_D2_to_Proto=10,
                 G_Proto_to_FSI=1.73, G_FSI_to_D2=1.35,
                 G_FSI_to_D1=0.5, G_Ctx_to_STN=0.5, G_Ctx_to_FSI=0.5, G_Ctx_to_D1=0.5, G_Ctx_to_D2=0.5,
                 G_STN_to_GPi=0.5, G_D1_to_GPi=0.5, G_Proto_to_GPi=0.5,
                 G_GPi_to_Th=0.5, G_Th_to_Ctx=0.5)


fig = run_and_plot(G_default, skip, params_file, pops, t_sim, 
                   noise_variance=noise_variance, n_model=n_model, dt=dt, all_to_all=all_to_all, noise_method=noise_method)
plt.show()