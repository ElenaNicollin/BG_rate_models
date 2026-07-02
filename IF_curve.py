import numpy as np
import json
import os
import warnings
warnings.simplefilter("always", RuntimeWarning)

from LIF_model import *

output_folder = "../data/FI_data"  #if changed here, also change in LIF_model.py, otherwise step3 won't run properly
os.makedirs(output_folder, exist_ok=True)


### Warning: this takes a long time to run ###
### Must be run before using the LIF_model ###

reps = 3

n_model=1000
pops = ["Proto", "STN", "D1", "D2", "FSI", "Arky", "Ctx", "GPi", "Th"]
species = "rat"
params_file = f"/home/elena/Documents/code/params/LIF_model/{species}_pop_params.json"
skip = 0.1
t_sim = 2+skip
dt=1e-4
n_steps=int(round(t_sim/dt,0))

input_params = load_params(params_file)
data = preprocess(input_params, species, rate=False)


###########################################
# STEP 1 - create IF-curves without noise #
###########################################

I_ext_values = {
    "Proto": np.concatenate([[0,5], np.arange(6,20,0.5)]),
    "STN": np.concatenate([[0,5], np.arange(6.5,10,0.05)]),
    "D1": np.concatenate([[0,15], np.arange(16,30,0.5)]),
    "D2": np.concatenate([[0,15], np.arange(16,30,0.5)]),
    "FSI": np.concatenate([[0,20], np.arange(22,29,0.1)]),
    "Arky": np.concatenate([[0,9], np.arange(10,20,0.5)]),
    "Ctx": np.concatenate([[0,23], np.arange(24,40,0.5)]),
    "GPi": np.concatenate(([0,5], np.arange(6,20,0.5))),
    "Th": np.concatenate(([0,5], np.arange(6,20,0.5)))
}


for pop in pops:
    print(pop)
    connectivity_params, K_values_sim = calculate_connectivity([pop], data, n_model, True, Proto_to_itself=False)  #nothing to connect really

    properties = data[pop]["properties"]

    rate_values = []
    for I_ext in I_ext_values[pop]:
        rate_values_each_rep = []
        for r in range(reps):
            print(I_ext, r)
            all_pops = []
            all_pops.append(create_pop(id=properties["id"], nb_neurons=n_model, n_steps=n_steps, state="Ctrl",
                                    mean_v_rest = properties["mean_v_rest"], sd_v_rest = properties["sd_v_rest"], range_v_rest = properties["range_v_rest"],
                                    mean_v_th = properties["mean_v_th"], sd_v_th = properties["sd_v_th"],
                                    mean_tau = properties["mean_tau"], sd_tau = properties["sd_tau"], range_tau = properties["range_tau"],
                                    mean_FR=properties[f"mean_FR_Ctrl"], sd_FR=properties[f"sd_FR_Ctrl"], range_FR=properties["range_FR"], nonlinearity_thresh=properties["nonlinearity_thresh"],
                                    I_ext_noise_method=None, noise_variance = {pop:0}, extra_stim_dict=dict(),
                                    a_adapt=0,b_adapt=0,tau_adapt=1))
            connect_all_pops(all_pops, connectivity_params, K_values_sim, dt, I_ext={pop: I_ext/1000})

            spikes = simulate_network(all_pops, t_sim, dt)
            rates = spikes_to_mean_rate(spikes, int(round(t_sim/dt, 0)), dt)
            rate_values_each_rep.append(np.mean(rates[pop]))

        rate_values.append(np.mean(rate_values_each_rep))
        print(rate_values[-1])
    
    with open(f"{output_folder}/FI_data_{pop}.json", 'w') as f:
        json.dump({"I_ext": I_ext_values[pop].tolist(), pop: rate_values}, f)


#################################################
# STEP 2 - infer noise level as 1/1000 of I_ext #
#################################################


states = ["Ctrl", "DD"]
noise_vars = {state: {} for state in states}

for state in states:
    for pop in pops:
        properties = data[pop]["properties"]
        pop_class = create_pop(id=properties["id"], nb_neurons=n_model, n_steps=n_steps, state=state,
                                    mean_v_rest=properties["mean_v_rest"], sd_v_rest=properties["sd_v_rest"], range_v_rest=properties["range_v_rest"],
                                    mean_v_th=properties["mean_v_th"], sd_v_th=properties["sd_v_th"],
                                    mean_tau=properties["mean_tau"], sd_tau=properties["sd_tau"], range_tau=properties["range_tau"],
                                    mean_FR=properties[f"mean_FR_{state}"], sd_FR=properties[f"sd_FR_{state}"], range_FR=properties["range_FR"], nonlinearity_thresh=properties["nonlinearity_thresh"],
                                    I_ext_noise_method=None, noise_variance="auto", extra_stim_dict = dict(),
                                    a_adapt=0,b_adapt=0,tau_adapt=1)
        pop_class.calculate_I_ext_mean(dt)
        noise_vars[state][pop] = pop_class.noise_variance


for k,v in noise_vars.items():
    print(k)
    print(v)


############################################
# STEP 3 - create IF-curves with O-U noise #
############################################

I_ext_values = {
    "Proto": np.arange(0,20,0.5),
    "STN": np.concatenate([[0,1], np.arange(2,10,0.05)]),
    "D1": np.concatenate([[0,10], np.arange(11,30,0.5)]),
    "D2": np.concatenate([[0,10], np.arange(11,30,0.5)]),
    "FSI": np.concatenate([[0,10], np.arange(12,30,0.1)]),
    "Arky": np.concatenate([[0,5], np.arange(5.5,20,0.5)]),
    "Ctx": np.concatenate([[0,19], np.arange(20,40,0.5)]),
    "GPi": np.concatenate([[0,5], np.arange(6,20,0.5)]),
    "Th": np.concatenate([[0,10], np.arange(11,20,0.5)])
}


for state in states:
    for pop in pops:
        print(pop)
        connectivity_params, K_values_sim = calculate_connectivity([pop], data, n_model, True, Proto_to_itself=False)  #nothing to connect really

        properties = data[pop]["properties"]

        rate_values = []
        for I_ext in I_ext_values[pop]:
            rate_values_each_rep = []
            for r in range(reps):
                print(I_ext, r)
                all_pops = []
                all_pops.append(create_pop(id=properties["id"], nb_neurons=n_model, n_steps=n_steps, state=state,
                                        mean_v_rest = properties["mean_v_rest"], sd_v_rest = properties["sd_v_rest"], range_v_rest = properties["range_v_rest"],
                                        mean_v_th = properties["mean_v_th"], sd_v_th = properties["sd_v_th"],
                                        mean_tau = properties["mean_tau"], sd_tau = properties["sd_tau"], range_tau = properties["range_tau"],
                                        mean_FR=properties[f"mean_FR_{state}"], sd_FR=properties[f"sd_FR_{state}"], range_FR=properties["range_FR"], nonlinearity_thresh=properties["nonlinearity_thresh"],
                                        I_ext_noise_method=None, noise_variance = noise_vars[state][pop], extra_stim_dict=dict(),
                                        a_adapt=0,b_adapt=0,tau_adapt=1))
                connect_all_pops(all_pops, connectivity_params, K_values_sim, dt, I_ext={pop: I_ext/1000})

                all_pops[0].I_ext_noise_method = "Ornstein-Uhlenbeck"

                spikes = simulate_network(all_pops, t_sim, dt)
                rates = spikes_to_mean_rate(spikes, int(round(t_sim/dt, 0)), dt)
                rate_values_each_rep.append(np.mean(rates[pop]))

            rate_values.append(np.mean(rate_values_each_rep))
            print(rate_values[-1])
        
        with open(f"{output_folder}/FI_data_noisy_{state}_{pop}.json", 'w') as f:
            json.dump({"I_ext": I_ext_values[pop].tolist(), pop: rate_values}, f)