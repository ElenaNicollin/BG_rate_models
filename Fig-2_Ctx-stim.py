import numpy as np
from scipy.stats import normaltest
import matplotlib.pyplot as plt
from matplotlib import colormaps as mcmaps
from matplotlib.backends.backend_pdf import PdfPages
import argparse
# import time

from rate_model import *
from rate_analysis import *


parser = argparse.ArgumentParser(
        prog='Fig. 2 - Cortex stim',
        description='Simulate cortical stimulation and compare latencies to experimental')
parser.add_argument('-o', '--outfile', default=None)
parser.add_argument('-n', '--n_repeats', type=int, required=True)
parser.add_argument('-c', '--color_mode', choices=["simple", "multiple"], default="simple")
args = parser.parse_args()


def get_pop_latencies(rates, stim_starts, dt, bin_size=1e-3, which_stim="Ctx"):
    n_cells = len(rates)
    EE_all = np.zeros(n_cells, float)
    SI_all = np.zeros(n_cells, float)
    LE_all = np.zeros(n_cells, float)

    for i_cell, rate in enumerate(rates):
        EE_repeats = np.zeros(len(stim_starts), float)
        SI_repeats = np.zeros(len(stim_starts), float)
        LE_repeats = np.zeros(len(stim_starts), float)

        for i_stim, stim_t in enumerate(stim_starts):
            stim_idx = int(round(stim_t/dt, 0))
            start_idx = stim_idx - int(0.2/dt)  #200ms pre stim for baseline
            end_idx = stim_idx + int(0.1/dt)   #post stim: crop to 100ms
            
            prestim_rate = rate[start_idx:stim_idx]
            prestim_psth = rate_to_psth(prestim_rate, dt, bin_size)
            CI95_low, CI95_high = stat.t.interval(0.95, len(prestim_psth)-1, loc=np.mean(prestim_psth), scale=np.std(prestim_psth))

            stim_rate = rate[stim_idx:end_idx]
            psth = rate_to_psth(stim_rate, dt, bin_size)

            exc_onsets = get_events_onsets(np.where(psth>CI95_high)[0], bin_size)
            inh_onsets = get_events_onsets(np.where(psth<CI95_low)[0], bin_size)


            ## handling various possible outcomes
            if np.isnan(exc_onsets[0]): # if no excitations found
                EE_onset = np.nan
                LE_onset=np.nan
                SI_onset = np.nan if np.isnan(inh_onsets[0]) else inh_onsets[0]
            else:
                EE_onset = exc_onsets[0]
                LE_onset = exc_onsets[1]
                for idx, inh in enumerate(inh_onsets): # looping until we get an inhibition onset happening after the early excitation (avoid starting with inhibition)
                    if inh>EE_onset or np.isnan(inh):
                        SI_onset = inh
                        next_inh = inh_onsets[idx+1] if idx<len(inh_onsets)-1 else np.nan
                        break
                if not np.isnan(next_inh) and LE_onset>next_inh:
                    LE_onset = np.nan

                if SI_onset > LE_onset: # check short inhibition happens before late excitation, else ignore
                    SI_onset = np.nan

            EE_repeats[i_stim] = round(EE_onset*1000, 0)
            SI_repeats[i_stim] = round(SI_onset*1000, 0)
            LE_repeats[i_stim] = round(LE_onset*1000, 0)
        EE_all[i_cell] = np.nanmean(EE_repeats)
        SI_all[i_cell] = np.nanmean(SI_repeats)
        LE_all[i_cell] = np.nanmean(LE_repeats)
    
    latencies = np.array([EE_all.copy(), SI_all.copy(), LE_all.copy()])
    return latencies


def run_and_get_latencies(stim_starts, params_file, pops, t_sim, n_model, dt, all_to_all, noise, G_dict, stim_dict, noise_variance, pops_subgroup):

    rates = run_rate(params_file, pops, t_sim, n_model, dt, all_to_all, noise, G_dict, stim_dict, noise_variance=noise_variance, return_all=True, gaussian_delay=False, adaptation=False)

    latencies = {}
    rates_for_plotting = {}
    for pop in pops_subgroup:
        pop_rates = rates[pop]
        latencies[pop] = get_pop_latencies(pop_rates, stim_starts, dt)
    for pop in rates.keys():
        pop_mean_rates = np.mean(rates[pop], axis=0)
        rates_for_plotting[pop] = np.zeros(int(round(0.12/dt, 0)))
        for stim_t in stim_starts:
            rates_for_plotting[pop] += pop_mean_rates[int(round((stim_t-0.02)/dt, 0)) : int(round((stim_t+0.1)/dt, 0))]
        rates_for_plotting[pop] /= len(stim_starts)
    del rates
    return latencies, rates_for_plotting


pops = ["Th", "Ctx", "FSI", "D1", "D2", "STN", "Proto", "Arky", "GPi"]
Kita_pops = {"GPe": ["Proto"], "STN": ["STN"], "GPi": ["GPi"], "STR": ["D1","D2"]}
sublist_pops = [x for l in Kita_pops.values() for x in l]

n_model = 1000
dt = 1e-4
all_to_all = True
noise_method = "Ornstein-Uhlenbeck" #can be None, "Gaussian", or "Ornstein-Uhlenbeck"
noise_variance = "auto"

G_default = dict(G_Proto_to_STN=-1, G_STN_to_Proto=1, G_Proto_to_Proto=-1,
                 G_Proto_to_Arky=-1, G_Arky_to_D2=-1, G_Arky_to_D1=-1, G_D2_to_Proto=-1,
                 G_Proto_to_FSI=-1, G_FSI_to_D1=-1, G_FSI_to_D2=-1,
                 G_Ctx_to_STN=1, G_Ctx_to_FSI=1, G_Ctx_to_D1=1, G_Ctx_to_D2=1,
                 G_STN_to_GPi=1, G_D1_to_GPi=-1, G_Proto_to_GPi=-1,
                 G_GPi_to_Th=-1, G_Th_to_Ctx=1)



# onset values are (mean, sd), ordered by [early excitation, short inhibition, late excitation]

rat_experimental_onsets = dict(
    GPe = {"Kita 2011": [(7.4, 2.2), (13.4, 1.5), (27.0, 4.9)]},
    STN = {"Kita 2011": [(5.5, 1.3), (10.4, 2.3), (19.1, 3.8)]},
    GPi = {"Kita 2011": [(6.0, 0.9), (13.4, 1.5), (24.3, 4.2)]},
    STR = {"Kita 2011": [(10.5, 2.1)]}
)

monkey_experimental_onsets = dict(
    GPe = {
        "Nambu 2000": [(9.2, 3.8), (16.9, 4.4), (25.8, 2.6)],
        "Kita 2004": [(8.7, 1.3), (16.9, 2.0), (30.8, 1.9)],
        "Iwamuro 2017": [(9.2, 2.0), (16.7, 3.0), (28.1, 4.4)]},
    STN = {
        "Nambu 2000": [(5.8, 4.5), (np.nan, np.nan), (19.8, 5.3)],
        "Iwamuro 2009": [(7.0, 1.7), (np.nan, np.nan), (16.7, 2.8)],
        "Iwamuro 2017": [(6.7, 1.8), (np.nan, np.nan), (17.8, 3.8)]},
    GPi = {
        "Nambu 2000": [(7.8, 2.4), (20.9, 5.0), (29.9, 6.5)],
        "Tachibana 2008": [(9.2, 2.2), (19.4, 3.0), (32.6, 8.7)]},
    STR = {
        "Turner 2000": [(8.5, 5.9)], "Nambu 2002": [(10.2, 2.5)]})


refs_ordered=["Nambu 2000", "Turner 2000", "Nambu 2002", "Kita 2004", "Tachibana 2008", "Iwamuro 2009", "Kita 2011", "Iwamuro 2017"]
experimental_data = {"rat": rat_experimental_onsets, "monkey": monkey_experimental_onsets}

color_mode = args.color_mode

if color_mode == "multiple":
    n_refs = len(refs_ordered)
    cmap = mcmaps["plasma"]
    ref_colors = cmap(np.linspace(0, 1, n_refs))
    colors = {"Model": "#26a01b"}
    for i,ref in enumerate(refs_ordered):
        colors[ref] = ref_colors[i]

elif color_mode == "simple":
    experimental_data["rat"] = {k: {"Reference": experimental_data["rat"][k]["Kita 2011"]} for k in ["GPe", "STN", "GPi", "STR"]}
    experimental_data["monkey"] = {k: {"Reference": experimental_data["monkey"][k]["Nambu 2000"]} for k in ["GPe", "STN", "GPi"]}
    colors = {"Model": "#c28419", "Reference": "#4270a1"}


fig, ax = plt.subplots(2,2, sharex='row') #Fig A & B
fig.set_figwidth(12)
fig.set_figheight(7)

fig2, ax2 = plt.subplots(1,2, squeeze=True) #Supp Fig -> Table with latencies
fig2.set_figwidth(11)

fig3, ax3 = plt.subplots(1,4, squeeze=True, sharex=True, sharey=True) #Fig C
fig3.set_figwidth(12)
fig3.set_figheight(2)


for col, species in enumerate(["rat", "monkey"]):
    print(species)
    exp_pops = [k for k in experimental_data[species].keys()]
    params_file = f"params/{species}_pop_params.json"
    
    n_repeats = args.n_repeats
    stim_interval=1 if species=="rat" else 1.5
    stim_starts = np.linspace(stim_interval, n_repeats*stim_interval, n_repeats)
    stim_duration = 3e-4
    
    t_sim = round(stim_starts[-1]+0.1, 4)
    n_steps = int(round(t_sim/dt, 0))

    stim_dict = {"Ctx": np.zeros(n_steps)}
    for stim_t in stim_starts:
        stim_dict["Ctx"][int(round((stim_t)/dt, 0)) : int(round((stim_t+stim_duration)/dt, 0))] = 1800

    G_dict = {k:v for k,v in G_default.items()}
    if species=="monkey":
        for g in ["G_Proto_to_Arky", "G_Proto_to_FSI", "G_FSI_to_D2", "G_Arky_to_D2", "G_Proto_to_Proto"]:
            G_dict[g] = 0.8     #Avoid spontaneous oscillations from striatopallidal loops
    

    rates_to_plot = {k:np.zeros(n_steps) for k in pops}

    latencies, mean_rates = run_and_get_latencies(stim_starts, params_file, pops, t_sim, n_model, dt, all_to_all, noise_method, G_dict, stim_dict, noise_variance, sublist_pops)
    if "STR" in exp_pops:
        STR_concat = np.concatenate((latencies["D1"][0].copy(), latencies["D2"][0].copy())) # concatenate D1 and D2, early excitation only
        latencies["STR"] = np.array([STR_concat])

    del latencies["D1"], latencies["D2"]
    latencies["GPe"] = latencies.pop("Proto")

    if species=="monkey":
        statistic, pvalue = normaltest(latencies["GPe"][-1], nan_policy='omit')
        print("pvalue:", pvalue)

    mean_latencies = {k:np.nanmean(v, axis=1) for k,v in latencies.items()}
    sd_latencies = {k:np.nanstd(v, axis=1) for k,v in latencies.items()}
    print(mean_latencies)
    print(sd_latencies)
    formatted_latencies = {}
    for k in exp_pops:
        if k=="STR":
            formatted_latencies[k] = [f"{np.round(mean_latencies[k][0], 1)} ± {np.round(sd_latencies[k][0], 1)}", "", ""]
            continue
        formatted_latencies[k] = [f"{np.round(mean_latencies[k][i], 1)} ± {np.round(sd_latencies[k][i], 1)}" for i in range(3)]


    markers = ["^", "v", "^"]       #excitation inhibition excitation
    y_offset = [0.02, -0.02, 0.02]  #to avoid visual overlaps of the std bars
    pre_rate_t = 0.02

    plot_rate(mean_rates, 0.1+pre_rate_t, dt, zero_time=pre_rate_t, ax=ax[0,col])
    ax[0,col].set_xlim(-pre_rate_t,0.1)
    ax[0,col].set_xlabel("Time from cortex stimulation (ms)")
    ax[0,col].set_ylim(0,120)
    ax[0,col].set_ylabel("Population activity (spk/s)")
    ax[0,col].set_title(species.upper(), fontsize=20)
    ax[0,col].spines[['right', 'top']].set_visible(False)
    
    yticks=[]
    for y,k in enumerate(exp_pops[::-1]):
        print(k)
        yticks.append(k)
        for m,x in enumerate(mean_latencies[k]):            
            #model latency onsets
            ax[1,col].plot(x, y+0.2, ls=None, marker=markers[m], color = colors["Model"])
            ax[1,col].plot([x-sd_latencies[k][m], x+sd_latencies[k][m]], [y+0.2, y+0.2], color = colors["Model"])
            
            #experimental latency onsets
            for z, (ref, exp_onsets) in enumerate(experimental_data[species][k].items()):
                exp_x, exp_sd = exp_onsets[m]
                ax[1,col].plot(exp_x, y+y_offset[m]-z*0.15, ls=None, marker=markers[m], color = colors[ref])
                ax[1,col].plot([exp_x-exp_sd, exp_x+exp_sd], [y+y_offset[m]-z*0.15, y+y_offset[m]-z*0.15], color = colors[ref])
    ax[1,col].set_yticks(range(y+1))
    ax[1,col].set_yticklabels(yticks)
    ax[1,col].set_ylim(-0.5, y+0.5)
    ax[1,col].set_xlabel("Response latency from cortex stimulation (ms)")
    ax[1,col].spines[['right', 'top']].set_visible(False)

    ax2[col].set_title(species.upper(), fontsize=20)
    ax2[col].axis('off')
    ax2[col].axis('tight')
    ax2[col].table(cellText=np.array([v for v in formatted_latencies.values()]).T,
                colLabels=[k for k in formatted_latencies.keys()],
                rowLabels=["Early Exc.", "Short Inh.", "Late Exc."],
                loc='center').auto_set_column_width(col=list(range(len(yticks))))


    #### Fig C
    ## Inhib STN
    for i in range(2):
        ax3[2*col+i].plot(np.arange(0, (0.1+pre_rate_t), dt)-pre_rate_t, mean_rates["GPi"], color="k", ls="-", alpha=0.4)

    n_repeats = args.n_repeats
    stim_interval=1 if species=="rat" else 1.5
    stim_starts = np.linspace(stim_interval, n_repeats*stim_interval, n_repeats)
    stim_duration = 3e-4
    
    t_sim = round(stim_starts[-1]+0.1, 4)
    n_steps = int(round(t_sim/dt, 0))
    
    stims_with_pharmaco = {
        "STN blockade": {"Ctx": np.zeros(n_steps), "STN": np.full((n_steps, n_model), -1000)},
        "GPe blockade": {"Ctx": np.zeros(n_steps), "Proto": np.zeros((n_steps, n_model)), "Arky": np.zeros((n_steps, n_model))}
    }
    for stim_dict in stims_with_pharmaco.values():
        for stim_t in stim_starts:
            stim_dict["Ctx"][int(round((stim_t)/dt, 0)) : int(round((stim_t+stim_duration)/dt, 0))] = 1800
    prop_GPe_blocked = 0.7
    stims_with_pharmaco["GPe blockade"]["Proto"][:, :int(n_model*prop_GPe_blocked)] = -1000


    G_dict = {k:v for k,v in G_default.items()}



    for i, (title, stim_dict) in enumerate(stims_with_pharmaco.items()):
        mean_rates, _ = run_rate(params_file, pops, t_sim, n_model, dt, all_to_all, noise_method, G_dict, stim_dict, noise_variance=noise_variance, return_all=False, gaussian_delay=True, adaptation=False)
        
        rate_to_plot = {"GPi": np.zeros(int(round(0.12/dt, 0))), "Proto": np.zeros(int(round(0.12/dt, 0)))}
        for stim_t in stim_starts:
            rate_to_plot["GPi"] += mean_rates["GPi"][int(round((stim_t-0.02)/dt, 0)) : int(round((stim_t+0.1)/dt, 0))]
            rate_to_plot["Proto"] += mean_rates["Proto"][int(round((stim_t-0.02)/dt, 0)) : int(round((stim_t+0.1)/dt, 0))]
        rate_to_plot["GPi"] /= len(stim_starts)
        rate_to_plot["Proto"] /= len(stim_starts)
        

        axi = 2*col+i
        ax3[axi].plot(np.arange(0, (0.1+pre_rate_t), 1e-3)-pre_rate_t, rate_to_psth(rate_to_plot["GPi"], dt, 1e-3), color="k", ls="-")
        # ax3[axi].plot(np.arange(0, (0.1+pre_rate_t), dt)-pre_rate_t, rate_to_plot["Proto"], color="#1F77B4", ls="-")
        ax3[axi].set_title(title)
        ax3[axi].spines[['right', 'top']].set_visible(False)
    ax3[0].set_xlabel("Time from cortex stimulation (s)")
    ax3[0].set_ylabel("GPi activity (spk/s)")


### legend handling
ax[0,0].get_legend().remove()
# ax[0,1].get_legend().remove()
ax[0,1].legend(bbox_to_anchor=(1.24,1),loc="upper right")

ax[1,1].plot(3, -10, ls=None, marker="^", color="k", label="Exc.")
ax[1,1].plot(3, -10, ls=None, marker="v", color="k", label="Inh.")
# ax[1,1].plot(3, -10, color=colors["model"], label="Model")
for ref, color in colors.items():
    ax[1,1].plot(3, -10, color=color, label=ref)
ax[1,1].legend(bbox_to_anchor=(1.24,1), loc="upper right")

#plt.show()
with PdfPages(f"outputs/rate/{args.outfile}_fig.pdf") as pdf:
    pdf.savefig(fig)
with PdfPages(f"outputs/rate/{args.outfile}_table.pdf") as pdf:
    pdf.savefig(fig2)
