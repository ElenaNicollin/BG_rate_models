import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import argparse

from utils import *
from LIF_model import *
from rate_analysis import *


parser = argparse.ArgumentParser(
        prog='Fig. 5 - LIF rat model, individual generators',
        description='Activities, spikes, power spectrum and phase angle distributions')
parser.add_argument('-o', '--outfile', required=True)
args = parser.parse_args()


full_network = ["Th", "Ctx", "FSI", "D1", "D2", "STN", "Proto", "Arky", "GPi"]
G_default = dict(G_Proto_to_STN=-0.2, G_STN_to_Proto=0.2, G_Proto_to_Proto=-0.2,
                G_Proto_to_Arky=-0.2, G_Arky_to_D2=-0.2, G_Arky_to_D1=-1, G_D2_to_Proto=-0.2,
                G_Proto_to_FSI=-0.2, G_FSI_to_D1=-1, G_FSI_to_D2=-0.2,
                G_Ctx_to_STN=0.2, G_Ctx_to_FSI=1, G_Ctx_to_D1=1, G_Ctx_to_D2=1,
                G_STN_to_GPi=0.2, G_D1_to_GPi=-1, G_Proto_to_GPi=-1,
                G_GPi_to_Th=-0.2, G_Th_to_Ctx=0.2)


G_edits = [{}, #first round = steady state, no oscill
           {"G_Proto_to_STN": -3, "G_STN_to_Proto": 3},
           {"G_Proto_to_FSI": -5, "G_FSI_to_D2": -5, "G_D2_to_Proto": -5},
           {"G_Proto_to_Arky": -3, "G_Arky_to_D2": -3, "G_D2_to_Proto": -4},
           {"G_Ctx_to_STN": 1.8, "G_STN_to_GPi": 1.8, "G_GPi_to_Th": -1.8, "G_Th_to_Ctx": 1.8}] #last round = steady state, no oscill

n_model = 1000
skip = 0.1
t_sim = 3+skip
dt = 1e-4
n_steps = int(round(t_sim/dt, 0))
all_to_all = False
noise_method = "Ornstein-Uhlenbeck" #can be None, "Gaussian", or "Ornstein-Uhlenbeck"
noise_variance = "auto"
state = "DD" #can be "Ctrl" (control) or "DD" (dopamine-depleted): changes population FR (see params)


pops_sublist = ["Ctx", "STN", "Proto", "Arky", "D2"] #for phase angles

loops = ["steady", "STN loop", "FSI loop", "Arky loop", "hyperdirect loop"]


os.makedirs("outputs/LIF", exist_ok=True)
with PdfPages(f"outputs/LIF/{args.outfile}.pdf") as pdf:


    params_file = "params/LIF_model/rat_pop_params.json"
    input_params = load_params(params_file)
    data = preprocess(input_params)

    connectivity_params, K_values_sim = calculate_connectivity(full_network, data, n_model, all_to_all)

    all_pops = []
    for pop in full_network:
        properties = data[pop]["properties"]
        all_pops.append(create_pop(id=properties["id"], nb_neurons=n_model, n_steps=n_steps, syn_type=properties["type"], state=state,
                                   mean_v_rest=properties["mean_v_rest"], sd_v_rest=properties["sd_v_rest"], range_v_rest=properties["range_v_rest"],
                                   mean_v_th=properties["mean_v_th"], sd_v_th=properties["sd_v_th"],
                                   mean_tau=properties["mean_tau"], sd_tau=properties["sd_tau"], range_tau=properties["range_tau"],
                                   mean_FR=properties[f"mean_FR_{state}"], sd_FR=properties[f"sd_FR_{state}"], range_FR=properties["range_FR"], nonlinearity_thresh=properties["nonlinearity_thresh"],
                                   I_ext_noise_method=noise_method, noise_variance=noise_variance, extra_stim_dict=dict(),
                                   a_adapt = 0, b_adapt = 0, tau_adapt = 1))
    connect_all_pops(all_pops, connectivity_params, K_values_sim, dt)

    for i_loop, loop_name in enumerate(loops):
        fig = plt.figure()
        fig.set_figwidth(15)
        ax0 = plt.subplot(131)
        ax1 = plt.subplot(132)
        ax2 = plt.subplot(133, projection='polar')
        ax = [ax0, ax1, ax2]
        fig.suptitle(f"{loop_name}", fontsize=20)

        print(loop_name)

        G_dict = G_default.copy()
        for k,v in G_edits[i_loop].items():
            G_dict[k] = v
        data = edit_W_values(G_dict, data)

        connectivity_params, K_values_sim = calculate_connectivity(full_network, data, n_model, all_to_all)
        reconnect(all_pops, connectivity_params, dt)

        spikes = simulate_network(all_pops, t_sim, dt)

        rates = spikes_to_smoothed_rates(spikes, n_steps=int(round(t_sim/dt, 0)), dt=dt)
        mean_rates = {k:np.mean(v, axis=0) for k,v in rates.items()}

        plot_rate({k: v[0] for k,v in rates.items()}, t_sim, dt, skip, xaxis_len = 0.25, ax=ax[0])
        plot_fft(rates, dt, skip=skip, ax=ax[1])
        plot_relative_phase_angles_from_spikes(spikes, pops_sublist, "Proto", dt, t_sim, skip=skip, ax=ax[2])

        # plt.show()
        # ax[0].get_legend().remove()
        fig.tight_layout()
        pdf.savefig(fig)

        
        fig_raster = plot_raster({k:spikes[k] for k in pops_sublist}, t_sim, dt, skip=skip, n_model=n_model)
        fig_raster.suptitle(loop_name)
        pdf.savefig(fig_raster)
