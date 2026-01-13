import numpy as np, pandas as pd
import scipy.fft as fft, scipy.signal as sig
from scipy.stats import truncnorm, norm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

# import time

from rate_model import *
from rate_analysis import *
from utils import *


parser = argparse.ArgumentParser(
        prog='Heatmap',
        description='Plot frequency heatmap and subfigures')
parser.add_argument('-i', '--infile', default=None)
parser.add_argument('-o', '--outfile', required=True)
parser.add_argument('-s', '--species', required=True, choices=["rat", "monkey"])
args = parser.parse_args()


if args.species=="rat":
    G_list = dict(G_Proto_to_Proto=[0.5, 1, 0],
                  G_STN_to_Proto=[1.75, 1, 1.5],
                  G_D2_to_Proto=[3, 5, 3],
                  G_GPi_to_Th=[1, 0, 2])
else:
    G_list = dict(G_Proto_to_Proto=[0.5, 0, 0],
                  G_STN_to_Proto=[0.5, 1.25, 0.75],
                  G_D2_to_Proto=[1, 3, 4],
                  G_GPi_to_Th=[0, 1, 2])



def run_and_plot_insets_only(fig, n_rows, row, label, G_dict, skip=0, *args, **kwargs):
    # rates = run_rate(*args, G_dict=G_dict, return_all=True, gaussian_delay=False, **kwargs)
    rates, std = run_rate(*args, G_dict=G_dict, **kwargs)


    ### PLOTS ###
    n_col = 10
    k = n_col*row

    axwelch = fig.add_subplot(n_rows, n_col, (k+1, k+2))
    # plot_fft(rates, dt, skip=skip, window_size=2, ax=axwelch)
    plot_welch({k:rates[k] for k in ["Proto", "STN", "D2"]}, dt, noise_method, window_size=2, skip=skip, ax=axwelch)
    axwelch.axvspan(12, 30, color="#ebebeb", zorder=-1)
    # axwelch.axvline(x=18, ls="dashed", color="dimgray", zorder=-1, lw=0.8)
    axwelch.set_xlim(9,70)
    axwelch.get_legend().remove()
    axwelch.grid(False)

    axrate = fig.add_subplot(n_rows, n_col, (k+3,k+5))
    plot_rate(rates, t_sim, dt=1e-4, skip=skip, ax=axrate)
    axrate.set_xlim(0.05, 0.55)
    axrate.get_legend().remove()
    axrate.set_title("")
    if row!=n_rows-1:
        axrate.set_xlabel("")
        axwelch.set_xlabel("")
        axrate.set_ylabel("")
        axwelch.set_ylabel("")
    
    axphase = fig.add_subplot(n_rows, n_col, (k+6, k+7), projection='polar')
    plot_relative_phase_angle_simplified(rates, ["Proto", "STN", "Arky", "D2"], "Proto", dt, skip=skip, ax=axphase)
    
    axloopsG = fig.add_subplot(n_rows, n_col, (k+8, k+9))
    axloopsG = plot_full_table(G_dict, pops)

    # fig.text(0.01, 0.91-0.31*row, label, va='center', fontsize=14, weight="bold")
    
    plt.tight_layout()
    # plt.show()
    return fig


#####################################


pops = ["Th", "Ctx", "FSI", "D1", "D2", "STN", "Arky", "Proto", "GPi"]

species = args.species
params_file = f"params/{species}_pop_params.json"
n_model = 100
skip = 0.5 if species=="rat" else 0.5
t_sim = 2+skip
dt = 1e-4
n_steps = int(round(t_sim/dt, 0))
all_to_all = True
noise_method = None #can be None, "Gaussian", or "Ornstein-Uhlenbeck"
noise_variance = "auto"


G_default = dict(G_Proto_to_STN=1, G_STN_to_Proto=0, G_Proto_to_Proto=2,
                 G_Proto_to_Arky=1, G_Arky_to_D2=1, G_D2_to_Proto=5,
                 G_Proto_to_FSI=1, G_FSI_to_D1=1, G_FSI_to_D2=1,
                 G_Ctx_to_STN=1, G_Ctx_to_FSI=1, G_Ctx_to_D1=1, G_Ctx_to_D2=1,
                 G_STN_to_GPi=1, G_D1_to_GPi=1, G_Proto_to_GPi=1,
                 G_GPi_to_Th=1.5, G_Th_to_Ctx=1)


draw_heatmap = args.infile!=None
subfig_letter = {"rat":["A","B"], "monkey":["C","D"]}

with PdfPages(f"outputs/rate/{args.outfile}.pdf") as pdf:
    if draw_heatmap:
        f_values = np.load(f"outputs/rate/{args.infile}")
        print(f_values.shape)
        G_Proto_to_Proto_values = np.linspace(0, 1.5, 4)
        G_STN_to_Proto_values = np.linspace(0, 3.5, 15) if species=="rat" else np.linspace(0, 2.5, 11)
        G_D2_to_Proto_values = np.linspace(0, 5, 11) #if species=="rat" else np.linspace(0, 6, 13)
        G_GPi_to_Th_values = np.linspace(0, 3, 4)
        G_values = [G_Proto_to_Proto_values, G_STN_to_Proto_values, G_D2_to_Proto_values, G_GPi_to_Th_values]
        
        n_tot = len(G_Proto_to_Proto_values) * len(G_STN_to_Proto_values) * len(G_D2_to_Proto_values) *len(G_GPi_to_Th_values)
        n_max = np.max([len(G_Proto_to_Proto_values), len(G_STN_to_Proto_values), len(G_D2_to_Proto_values), len(G_GPi_to_Th_values)])
        
        fig_imshow, ax_imshow = plot_4D_frequency_heatmap(f_values, G_Proto_to_Proto_values, G_STN_to_Proto_values, G_D2_to_Proto_values, G_GPi_to_Th_values)
        fig_imshow.text(0.03, 0.95, subfig_letter[species][0], ha='left', va='top', fontsize=28, weight="bold")

    n = len([v for v in G_list.values()][0])
    labels = ["a", "b", "c", "d", "e", "f", "g", "h"]

    fig_insets = plt.figure(figsize=(9, 0.5+1.8*n), dpi=100)
    fig_insets.suptitle(f"species: {species}, noise: {noise_method}, all_to_all: {all_to_all}, t_sim: {t_sim}")
    # fig_insets.text(0.01, 0.99, subfig_letter[species][1], ha='left', va='top', fontsize=28, weight="bold")


    for i in range(n):
        print(i)
        G_dict = G_default.copy()
        for k,v in G_list.items():
            G_dict[k] = v[i]
        if draw_heatmap:
            coords = [G_dict[k] for k in G_list.keys()]
            idx_coords = [int(np.where(G_x_to_x==coords[idx])[0][0]) for idx, G_x_to_x in enumerate(G_values)]
            print(coords)
            print(idx_coords)

            ax_imshow[idx_coords[3], idx_coords[0]].text(coords[1], coords[2], labels[i], ha="center", va="center",
                                                         bbox = dict(boxstyle=f"circle,pad=0.1", fc="white", alpha=0.6))

        run_and_plot_insets_only(fig_insets, n, i, labels[i], G_dict, skip, params_file, pops, t_sim,
        noise_variance=noise_variance, n_model=n_model, dt=dt, all_to_all=all_to_all, noise_method=noise_method)


    if draw_heatmap:
        pdf.savefig(fig_imshow, dpi=300)
    pdf.savefig(fig_insets, dpi=300)
