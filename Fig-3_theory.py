import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sympy import *

from utils import *
from rate_model import *
from rate_analysis import *

parser = argparse.ArgumentParser(
        prog='Fig. 3 - Theoretical properties',
        description='Plot Glim values & achievable frequencies')
parser.add_argument('-o', '--outfile', default=None)
args = parser.parse_args()


def get_delays(connectivity_params, nuclei):
    sum_delays=0
    for A,B in combinations(nuclei, 2):
        if B in connectivity_params[A].keys():
            sum_delays += connectivity_params[A][B]["delay"]
        if A!=B and A in connectivity_params[B]:
            sum_delays += connectivity_params[B][A]["delay"]
    return round(sum_delays, 5)

def get_taus(connectivity_params, nuclei):
    list_taus=[]
    for A,B in combinations(nuclei, 2):
        if B in connectivity_params[A].keys():
            list_taus.append(connectivity_params[A][B]["tau_decay"])
        if A!=B and A in connectivity_params[B]:
            list_taus.append(connectivity_params[B][A]["tau_decay"])
    return list_taus

def interpolate_delay(v, x, y):
    idx = np.where(np.subtract(x,v)>0)[0][0]
    x1 = x[idx-1]
    x2 = x[idx]
    y1 = y[idx-1]
    y2 = y[idx]
    return linear_interpolation(v, x1, x2, y1, y2)




species_list = ["rat", "monkey"]
linestyles={"rat":"dotted", "monkey":"dashed"}
markers={"rat":"o", "monkey":"s"}
loops = json.load(open("params/graphical_params.json"))["loops"]


fig, (axGlim, _ax, axC1, axC2, axC3, axC4) = plt.subplots(1, 6, gridspec_kw={'width_ratios': [1.2, 0.3, 1,1,1,1]})
# fig = plt.figure(figsize=(6.92, 3.5), dpi=300)
fig.dpi = 300
fig.set_figwidth(6.92)
fig.set_figheight(3.5)
_ax.remove()
axC2.sharex(axC1)
axC2.sharey(axC1)
axC3.sharex(axC1)
axC3.sharey(axC1)
axC4.sharex(axC1)
axC4.sharey(axC1)

list_axC = [axC1, axC2, axC3, axC4]

plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.family': "Arial"})


### Fig. 3B
Glim_values={}
for x, species in enumerate(species_list):
    Glim_values[species] = []
    params_file = f"params/{species}_pop_params.json"
    input_params = load_params(params_file)
    data = preprocess(input_params)

    for loop_name, loop_params in loops.items():
        root_f, G = get_loop_G_critical(data, loop_params)
        print(species, loop_name, np.round(G, 2), np.round(root_f/(2*3.14), 2))
        Glim_values[species].append(abs(G))
    
for i,(loop_name, loop_params) in enumerate(loops.items()):
    axGlim.plot([0,1],[ Glim_values["rat"][i], Glim_values["monkey"][i] ],
                color=loop_params["color"], marker="x", ls='--', lw=0.6, label=loop_name)
axGlim.set_ylabel("Glim")
axGlim.set_xticks([0,1])
axGlim.set_xticklabels(["Rat", "Monkey"])
axGlim.set_xlim(-0.5,1.5)
axGlim.spines[['right', 'top']].set_visible(False)


func_dict = {"1": theory_1_nuclei, "2": theory_2_nuclei, "3": theory_3_nuclei, "4": theory_4_nuclei}
for species in species_list:
    ### Fig. 3C
    params_file = f"params/{species}_pop_params.json"
    input_params = load_params(params_file)
    data = preprocess(input_params)
    bin_size = 0.5
    sum_delays_values = np.arange(4,26.5,bin_size)
    ls = linestyles[species]
    mk = markers[species]
    lw = 0.9
    ms = 4
    for i, (loop_name, loop_params) in enumerate(loops.items()):
        nuclei = loop_params["nuclei"]
        network = np.unique(nuclei)
        n = len(network)
        theory_freqs = np.zeros_like(sum_delays_values)
        Proto_to_itself = True if n==1 else False

        connectivity_params, _ = calculate_connectivity(network, data, 100, True, Proto_to_itself=Proto_to_itself)
        #get real sum delays value
        real_sum_delay = np.sum(get_delays(connectivity_params, nuclei))


        x0_lims = loop_params[f"x0_delay_{species}"]
        x0 = np.linspace(x0_lims[0], x0_lims[1], len(sum_delays_values))
        
        for j,sum_delays in enumerate(sum_delays_values):
            # calculate theoretical frequency
            list_taus = get_taus(connectivity_params, nuclei)
            root, G = func_dict[f"{n}"] (x0[j], list_taus, sum_delays/1000)
            theory_freqs[j] = root / (2*np.pi)
            
        list_axC[n-1].plot(sum_delays_values, theory_freqs, color=loop_params["color"], ls=ls, lw=lw)
        list_axC[n-1].plot(real_sum_delay*1000, interpolate_delay(real_sum_delay*1000, sum_delays_values, theory_freqs), marker=mk, ms=ms, ls=ls, lw=lw, color=loop_params["color"])
        #for legend purpose
        list_axC[n-1].plot(100, 100, ls=ls, lw=lw, marker=mk, ms=ms, color="k", label=species, )

for axCi in list_axC:
    axCi.axhspan(12, 30, color="lightgrey", alpha=0.4, zorder=-1, ec=None)
    axCi.axhline(y=18, ls="dashed", color="dimgray", zorder=-1, lw=0.75)
    # axi.set_xlabel("sum synaptic delays (ms)", fontsize=14)
    axCi.tick_params('y', labelleft=False)
    axCi.spines[['right', 'top']].set_visible(False)
axC1.tick_params('y', labelleft=True)
axC1.set_yticks(range(0,81,10))
axC1.set_yticklabels([0,"","","",40,"","","",80])
axC1.set_xlim(sum_delays_values[0], sum_delays_values[-1])
axC1.set_ylim(0,80)
axC1.set_ylabel("Frequency (Hz)")
axC1.set_xlabel('sum of loop synaptic delays (ms)', loc="left")
axC4.legend(bbox_to_anchor=(0.5,0.7),loc="upper left", frameon=False)
axC4.set_zorder(-1)

####

# fig.text(0.5, 0.02, 'sum of loop synaptic delays (ms)', ha='center', fontsize=14)
# ax[0,1].get_legend().remove()
axGlim.legend(bbox_to_anchor=(4.9,1),loc="upper left", frameon=False)
# ax[0,1].legend(bbox_to_anchor=(1,1),loc="upper right", framealpha=1, )
# ax[1,1].legend(bbox_to_anchor=(1,1),loc="upper right", framealpha=1)
#ax[2,1].legend(bbox_to_anchor=(1.24,1),loc="upper right")
# fig.tight_layout(h_pad=2, w_pad=-5)
fig.tight_layout()

# plt.show()
outfile = "Fig-3_theory_output" if args.outfile==None else args.outfile
with PdfPages(f"outputs/rate/{outfile}.pdf") as pdf:
    pdf.savefig(fig, bbox_inches='tight', dpi=300)