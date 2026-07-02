import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import argparse

from utils import *
from rate_model import *
from rate_analysis import *


parser = argparse.ArgumentParser(
        prog='Fig. 4 - Individual generators',
        description='Power spectra and phase angle distributions')
parser.add_argument('-o', '--outfile', required=True)
args = parser.parse_args()


loops = json.load(open("params/graphical_params.json"))["loops"]
species_list = ["rat", "monkey"]

full_network = ["Th", "Ctx", "FSI", "D1", "D2", "STN", "Proto", "Arky", "GPi"]
G_default = dict(G_Proto_to_STN=-0.2, G_STN_to_Proto=0.2, G_Proto_to_Proto=-0.2,
                G_Proto_to_Arky=-0.2, G_Arky_to_D2=-0.2, G_Arky_to_D1=-1, G_D2_to_Proto=-0.2,
                G_Proto_to_FSI=-0.2, G_FSI_to_D1=-1, G_FSI_to_D2=-0.2,
                G_Ctx_to_STN=0.2, G_Ctx_to_FSI=1, G_Ctx_to_D1=1, G_Ctx_to_D2=1,
                G_STN_to_GPi=0.2, G_D1_to_GPi=-1, G_Proto_to_GPi=-1,
                G_GPi_to_Th=-0.2, G_Th_to_Ctx=0.2)

n_model = 1000
skip = 0.5
t_sim = 4+skip
dt = 1e-4
n_steps = int(round(t_sim/dt, 0))
all_to_all = True
noise_method = "Ornstein-Uhlenbeck" #can be None, "Gaussian", or "Ornstein-Uhlenbeck"

fig_fft, ax_fft = plt.subplots(1,2, squeeze=True, sharex=True)
fig_fft.set_figwidth(6.4*2)
fig_main, ax_main = plt.subplots(5, 2, subplot_kw=dict(projection='polar'))
fig_main.set_figheight(4.8*5)
fig_main.set_figwidth(6.4*2)


with PdfPages(f"outputs/rate/{args.outfile}.pdf") as pdf:
    for s, species in enumerate(species_list):
        ax_fft[s].set_title(species.upper(), fontsize=20)
        ax_fft[s].set_xlabel("Frequency (Hz)")

        params_file = f"params/{species}_pop_params.json"
        input_params = load_params(params_file)
        data = preprocess(input_params)

        connectivity_params, K_values_sim = calculate_connectivity(full_network, data, n_model, all_to_all)
        
        all_pops = []
        for pop in full_network:
            properties = data[pop]["properties"]
            all_pops.append(create_pop(id=properties["id"], nb_neurons=n_model, n_steps=n_steps, syn_type=properties["type"], mean_FR=properties["mean_FR"], sd_FR=properties["sd_FR"], range_FR=properties["range_FR"], I_ext_noise_method=noise_method, noise_variance="auto", thresh=properties["threshold"]))
        connect_all_pops(all_pops, connectivity_params, K_values_sim, dt)

        for i_loop, (loop_name, loop_params) in enumerate(loops.items()):
            fig = plt.figure()
            fig.set_figwidth(15)
            ax0 = plt.subplot(131)
            ax1 = plt.subplot(132)
            ax2 = plt.subplot(133, projection='polar')
            ax = [ax0, ax1, ax2]
            fig.suptitle(f"{species.upper()} - {loop_name}", fontsize=20)

            print(loop_name)
            nuclei = loop_params["nuclei"]
            loop_network = np.unique(nuclei).tolist()

            # get G* for current loop
            _, G = get_loop_G_critical(data, loop_params)

            # set G values such that this loop will drive oscillations
            new_G = (np.abs(G))**(1/len(loop_network))
            #if loop_name=="hyperdirect":
            #    new_G -= 0.1
            #    G_dict["G_STN_to_Proto"] = 0.5
            print(new_G)
            G_dict = G_default.copy()
            loop_meanFR = np.mean([data[pop]["properties"]["mean_FR"] for pop in loop_network])
            for a,b in combinations(nuclei, 2):
                for k in [f"G_{a}_to_{b}", f"G_{b}_to_{a}"]:
                    if k in G_dict.keys():
                        # FR1 = data[k.split("_")[1]]["properties"]["mean_FR"]
                        # FR2 = data[k.split("_")[-1]]["properties"]["mean_FR"]
                        # sign = G_dict[k]/abs(G_dict[k])
                        # G_dict[k] = sign *  np.abs(new_G * (FR2-loop_meanFR)/(FR1-loop_meanFR))
                        G_dict[k] = new_G +0.1*(len(loop_network)-1) #G_dict[k] = new_G +0.3
                        #G_dict[k] = new_G #default

            data = edit_G_values(G_dict, data)

            connectivity_params, K_values_sim = calculate_connectivity(full_network, data, n_model, all_to_all)
            reconnect(all_pops, connectivity_params)

            rates = simulate_network(all_pops, t_sim, dt, return_all=True)
            pops_sublist = ["Proto", "Arky", "STN", "D2", "Ctx"]

            plot_rate({k: v[0] for k,v in rates.items()}, t_sim, dt, skip, xaxis_len = 0.25, ax=ax[0])
            plot_mean_coherences(rates, pops_sublist, "Proto", dt, window_size=1, ax=ax[1])
            ax[1].set_xlim(0,60)
            plot_relative_phase_angles(rates, pops_sublist, "Proto", dt, skip=skip, loop_name=loop_name, normalize=False, n_pairs=2000, ax=ax[2])
            plot_relative_phase_angles(rates, pops_sublist, "Proto", dt, skip=skip, loop_name=loop_name, normalize=False, n_pairs=2000, ax=ax_main[i_loop, s])
            # plt.show()
            ax[0].get_legend().remove()
            fig.tight_layout()
            pdf.savefig(fig)

            #f_Proto, Pxx_Proto = np.mean([compute_welch_of_signal(rate, dt) for rate in rates["Proto"]], axis=0)
            f_Proto, fft_Proto = compute_fft_of_signal(np.mean(rates["Proto"], 0)[int(round(skip/dt, 0)):], dt, window_size=1)
            area = np.sum(fft_Proto)
            ax_fft[s].plot(f_Proto, fft_Proto/area, color=loop_params["color"], label=loop_name)
            ax_fft[s].spines[['right', 'top']].set_visible(False)

    ax_fft[0].set_ylabel("normalized FFT power")
    ax_fft[0].set_xlim(0,70)
    #ax_main[0,1].set_xlim(0,70)
    ax_fft[1].legend(bbox_to_anchor=(1,1),loc="upper left")
    fig_fft.tight_layout()
    fig_main.tight_layout()
    pdf.savefig(fig_fft)
    pdf.savefig(fig_main)