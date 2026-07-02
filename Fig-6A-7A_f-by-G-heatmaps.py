import numpy as np, pandas as pd
import scipy.signal as sig
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time
import argparse

from rate_model import *
from rate_analysis import *
from utils import *

parser = argparse.ArgumentParser(
        prog='Fig. 6A-7A: Heatmap',
        description='Run simulations, extract peak frequency, plot heatmap')
parser.add_argument('-o', '--outfile', required=True)
parser.add_argument('-s', '--species', required=True, choices=["rat", "monkey"])
parser.add_argument('-b', '--bin_size', default=None, type=float)
args = parser.parse_args()


print(f"outputs/rate/{args.outfile}")

def set_outfile_names(outfile):
    if outfile[-4:] == ".pdf":
        return outfile, str(outfile.rstrip("pdf")+"npy")
    return f"{outfile}.pdf", f"{outfile}_freqs.npy"


def run_and_get_f(all_pops, t_sim, dt, skip=0):
    rates, _sd = simulate_network(all_pops, t_sim, dt)
    signal = rates["Proto"]
    trunc_signal = signal[int(skip/dt):]
    oscill_state = is_stable_oscillation(trunc_signal)[0]
    if not oscill_state:
        return signal, np.nan
    peak_f = get_peak_f_of_signal(trunc_signal, dt, window_size=1)
    return signal, peak_f

#####################################


outfile_pdf, outfile_npy = set_outfile_names(args.outfile)

n_model = 100
network = ["Th", "Ctx", "FSI", "D1", "D2", "STN", "Proto", "Arky", "GPi"]
species = args.species
params_file = f"params/{species}_pop_params.json"
skip = 0.5 if species=="rat" else 1
t_sim = 2+skip
dt = 1e-4
n_steps = int(round(t_sim/dt, 0))
all_to_all = True
noise_method = None #can be None, "Gaussian", or "Ornstein-Uhlenbeck"

loops = json.load(open("params/graphical_params.json"))["loops"]


#    G_Proto_to_Proto_values = np.array([0.5, 2])
#    G_STN_to_Proto_values = np.array([0.5, 2, 3.5]) if species=="rat" else np.array([0.5, 3])
#    G_D2_to_Proto_values = np.array([0.5, 3]) if species=="rat" else np.array([0.5, 2])
#    G_GPi_to_Th_values = np.array([0.5, 2.5])
G_Proto_to_Proto_values = np.linspace(0, 1.5, 4)
G_STN_to_Proto_values = np.linspace(0, 3.5, 15) if species=="rat" else np.linspace(0, 2.5, 11)
G_D2_to_Proto_values = np.linspace(0, 5, 11) #if species=="rat" else np.linspace(0, 6, 13)
G_GPi_to_Th_values = np.linspace(0, 2, 5)
G_values = [G_Proto_to_Proto_values, G_STN_to_Proto_values, G_D2_to_Proto_values, G_GPi_to_Th_values]

n_tot = len(G_Proto_to_Proto_values) * len(G_STN_to_Proto_values) * len(G_D2_to_Proto_values) *len(G_GPi_to_Th_values)
n_max = np.max([len(G_Proto_to_Proto_values), len(G_STN_to_Proto_values), len(G_D2_to_Proto_values), len(G_GPi_to_Th_values)])
f_values = np.zeros((n_max,n_max,n_max,n_max), dtype=float)


input_params = load_params(params_file)
data = preprocess(input_params)

G_default = dict(G_Proto_to_STN=1, G_STN_to_Proto=1, G_Proto_to_Proto=1,
                G_Proto_to_Arky=1, G_Arky_to_D2=1, G_D2_to_Proto=1,
                G_Proto_to_FSI=1, G_FSI_to_D1=1, G_FSI_to_D2=1,
                G_Ctx_to_STN=1, G_Ctx_to_FSI=1, G_Ctx_to_D1=1, G_Ctx_to_D2=1,
                G_STN_to_GPi=1, G_D1_to_GPi=1, G_Proto_to_GPi=1,
                G_GPi_to_Th=1, G_Th_to_Ctx=1)

G_dict = G_default.copy()

data = edit_G_values(G_dict, data)

all_pops = []

# create populations
for pop in network:
    properties = data[pop]["properties"]
    all_pops.append(create_pop(id=properties["id"], nb_neurons=n_model, syn_type=properties["type"], n_steps=n_steps, mean_FR=properties["mean_FR"], sd_FR=properties["sd_FR"], range_FR=properties["range_FR"], I_ext_noise_method=noise_method, noise_variance = "auto", thresh=properties["threshold"]))

# connect populations
connectivity_params, K_values_sim = calculate_connectivity(network, data, n_model, all_to_all)
connect_all_pops(all_pops, connectivity_params, K_values_sim, dt)


i=1
start=time.time()
for a, G_Proto_to_Proto in enumerate(G_Proto_to_Proto_values):
    connectivity_params["Proto"]["Proto"]["G"] = G_Proto_to_Proto
    for b, G_STN_to_Proto in enumerate(G_STN_to_Proto_values):
        connectivity_params["D2"]["Proto"]["G"] = G_STN_to_Proto
        for c, G_D2_to_Proto in enumerate(G_D2_to_Proto_values):
            connectivity_params["Proto"]["Proto"]["G"] = G_Proto_to_Proto
            for d, G_GPi_to_Th in enumerate(G_GPi_to_Th_values):
                connectivity_params["Ctx"]["STN"]["G"] = G_D2_to_Proto
                print(f"progress {i}/{n_tot}")
                
                reconnect(all_pops, connectivity_params)
                _trash, f_values[a][b][c][d] = run_and_get_f(all_pops, t_sim, dt, skip=skip)
            
                reset(all_pops)

                stop=time.time()
                t_remain = time.strftime('%H:%M:%S', time.gmtime((stop-start)/i * (n_tot-i)))
                print("freq:", f_values[a][b][c][d])
                print("estimated time remaining:", t_remain)
                i+=1

np.save(f"outputs/rate/{outfile_npy}", f_values)
#f_values = np.random.randint(0, 80, (n_y,n_x))


# _, G_eq_PS = get_loop_G_critical(data, loops["Proto-STN"])
# _, G_eq_PFD = get_loop_G_critical(data, loops["Proto-FSI-D2"])
# print(G_eq_PS, G_eq_PFD)


pad=10
with PdfPages(f"outputs/rate/{outfile_pdf}") as pdf:
    fig, ax = plt.subplots(nrows=len(G_GPi_to_Th_values), ncols=len(G_Proto_to_Proto_values), figsize=(12,12), sharex=True, sharey=True)
    print("subplots:", np.shape(ax))
    for icol, G_Proto_to_Proto in enumerate(G_Proto_to_Proto_values):
        print("col:", icol, G_Proto_to_Proto)
        for irow, G_GPi_to_Th in enumerate(G_GPi_to_Th_values):
            print("row:", irow, G_GPi_to_Th)
            frequencies = f_values[icol, :len(G_STN_to_Proto_values), :len(G_D2_to_Proto_values), irow]
            print(frequencies)
            axi = ax[irow, icol]
            ylab = ""
            xlab = ""
            if icol==0:
                ylab="|G D2-Proto|"
                axi.annotate(G_GPi_to_Th, xy=(0, 0.5), xytext=(-axi.yaxis.labelpad - pad, 0), xycoords=axi.yaxis.label, textcoords='offset points', fontsize=14, ha='right', va='center')
            if irow==0:
                axi.annotate(G_Proto_to_Proto, xy=(0.5,1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', fontsize=14, ha='center', va='baseline')
            elif irow==len(G_GPi_to_Th_values)-1:
                xlab="|G STN-Proto|"
            img = plot_frequency_heatmap(frequencies.T, xvalues=G_STN_to_Proto_values, xlabel=xlab,
                                            yvalues=G_D2_to_Proto_values, ylabel=ylab, ax=axi)

    fig.text(0.44, 0.92, "|G Proto-Proto|", ha='center', fontsize=14)
    fig.text(0, 0.5, "|G Ctx-STN|", va='center', rotation='vertical', fontsize=14)

    #for axi, G_Proto_to_Proto in zip(ax[0], G_Proto_to_Proto_values):
    #for axi, G_GPi_to_Th in zip(ax[:,0], G_GPi_to_Th_values):


    clb = fig.colorbar(img, shrink=0.4, ax = ax[:,:])
    clb.set_ticks([0, 10, 20, 30, 40, 50, 60, 70])
    clb.set_ticklabels([0, 10, 20, 30, 40, 50, 60, 70])
    clb.set_label("frequency (Hz)", labelpad=20, y=0.5, rotation=-90, fontsize=14)
    #clb.ax.tick_params(labelsize=clb_tick_size, length = tick_length)
    #set_max_dec_tick(ax, n_decimal = n_decimal)
    clb.ax.yaxis.tick_right()

    #fig.tight_layout()
    pdf.savefig(fig)
