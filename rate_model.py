import numpy as np
import json
#from scipy.stats import truncnorm, norm

from utils import *

class Network:
    def __init__(self, id, nb_neurons, n_steps, syn_type, mean_FR, sd_FR, range_FR, I_ext_noise_method, noise_variance="auto", noise_amplitude = 1, noise_tau = 0.01, extra_stim_dict = None, thresh=0.1, adaptation = False, b_adapt = 0, tau_adapt = 0):
        self.id = id
        self.size = nb_neurons
        self.syn_type = syn_type
        self.set_syn_sign()

        self.I_ext_noise_method = I_ext_noise_method
        self.I_ext_noise_method_dict = {None: no_noise, "Gaussian": Gaussian_noise, "Ornstein-Uhlenbeck": Ornstein_Uhlenbeck_noise}
        #self.I_ext_sd = sd_I_ext
        self.noise_mode = noise_variance
        if noise_variance == "auto":
            self.set_noise(0, noise_amplitude, noise_tau)
        else:
            self.set_noise(noise_variance[self.id], noise_amplitude, noise_tau)
        self.noise = np.zeros(self.size, dtype=float)
        
        if extra_stim_dict is None or self.id not in extra_stim_dict.keys():
            self.extra_stim_values = np.zeros(n_steps)
        else:
            self.extra_stim_values = extra_stim_dict[self.id]

        #self.m_recordings = np.zeros((self.size, n_steps))
        self.m_recordings = {}
        self.m_start = gaussian_trunc_sample(mean_FR, sd_FR, min(range_FR), max(range_FR), self.size) if sd_FR !=0 else np.full(self.size, mean_FR)
        self.next_m = {}
        self.thresh=thresh

        
        self.tau_decay = {}
        self.delay= {}
        self.I_syn = 0
        self.A = np.zeros(nb_neurons, dtype=float)
        self.A_recordings = np.zeros((self.size, n_steps), dtype=float)
        self.list_receiving_from = []
        self.K_connections = {}
        self.G_values = {}
        self.connections = {}
        self.adaptation = adaptation
        self.b = np.full(self.size, b_adapt, dtype=float)
        self.tau_adapt = np.full(self.size, tau_adapt, dtype=float)
        self.w = np.zeros(self.size, dtype=float)


    def set_syn_sign(self):
        if self.syn_type == 1 or self.syn_type.casefold() in ["e".casefold(), "exc".casefold(), "excitation".casefold(), "excitatory".casefold()]:
            self.syn_sign = 1
        elif self.syn_type == -1 or self.syn_type.casefold() in ["i".casefold(), "inh".casefold(), "inhibition".casefold(), "inhibitory".casefold()]:
            self.syn_sign = -1
        else:
            raise ValueError("Unknown synaptic type for", self.id)

    def add_receiving_from(self, alpha_pop):
        self.list_receiving_from.append(alpha_pop)

    def set_K_connections(self, alpha_pop_id, K_sim):
        self.K_connections[alpha_pop_id] = K_sim

    def set_delay_params_distrib(self, dt, alpha_name, mean_delay):
        raw_delay = gaussian_trunc_sample(mean_delay, mean_delay/10, 0, 2*mean_delay, self.size)
        #raw_delay = np.full(self.size, mean_delay)
        self.delay[alpha_name] = dt*np.around(raw_delay/dt)   # to match delay precision with dt

    def set_delay_params(self, dt, alpha_name, delay):
        self.delay[alpha_name] = dt*np.around(delay/dt)
    
    def set_tau_decay_params(self, alpha_name, tau_decay):
        self.tau_decay[alpha_name] = tau_decay

    def set_m(self, alpha, alpha_name):
        self.m_recordings[alpha_name] = np.zeros_like(alpha.A_recordings)
        self.m_recordings[alpha_name][:,0] = alpha.m_start.copy()
        self.next_m[alpha_name] = alpha.A.copy()

    def set_connectivity_matrix(self, alpha_pop_id, alpha_syn_sign, n_alpha_neurons, G, same_pop = False):
        connection_prob = np.random.rand(n_alpha_neurons, self.size)
        G_real = alpha_syn_sign * np.abs(G)         #set inhibitory (<0) or excitatory (>0) gain
        self.G_values[alpha_pop_id] = G_real
        if same_pop:
            np.fill_diagonal(connection_prob, 0)    #avoid autapses when connecting population to itself by setting prob to 0
        
        K_sim = self.K_connections[alpha_pop_id]
        prob = K_sim/self.size
        J_matrix = np.where(connection_prob >= 1-prob, 1, 0) #G is normalized by population size

        if G_real==0:
            G_matrix = np.zeros((n_alpha_neurons, self.size))
        else:
            # G_matrix = lognormal_sample(m=G_real, size=np.shape(J_matrix))      # lognormally distributed G
            G_matrix = np.full((n_alpha_neurons, self.size), G_real/K_sim)            # fixed G for all cells
        connectivity_matrix = np.multiply(J_matrix, G_matrix)

        self.connections[alpha_pop_id] = connectivity_matrix

    def set_noise(self, variance, amplitude=1, tau=0.01):
        self.noise_variance = variance
        self.noise_std = np.sqrt(variance)
        self.noise_amplitude = amplitude
        self.noise_tau = tau

    def set_I_ext(self):
        I_ext_tot = self.m_start + self.thresh
        I_syn=0
        if self.list_receiving_from:
            I_syn_by_alpha = [np.sum(self.connections[alpha.id],1) * np.mean(alpha.m_start) for alpha in self.list_receiving_from]
            I_syn = np.sum(I_syn_by_alpha, 0)
        self.I_ext = I_ext_tot - I_syn
        if self.noise_mode == "auto":
            self.set_noise(0.5*np.abs(np.mean(I_ext_tot)), self.noise_amplitude, self.noise_tau)


    def I_ext_with_noise(self, dt):
        self.noise = self.I_ext_noise_method_dict[self.I_ext_noise_method] (self.noise_amplitude, self.noise_std, self.size, dt, np.sqrt(dt), self.noise_tau, self.noise)
        return self.I_ext + self.noise
        #return gaussian_sample(self.I_ext_mean, self.I_ext_sd, self.size)

    def calculate_and_update_m(self, t, dt):
        for alpha in self.list_receiving_from:
            self.m_recordings[alpha.id][:,t] = self.next_m[alpha.id].copy()
            dm = ((-self.m_recordings[alpha.id][:,t] + alpha.A) / self.tau_decay[alpha.id])*dt
            self.next_m[alpha.id] += dm
    
    def adaptation_current(self, t, dt):
        dw = dt/self.tau_adapt * \
        (-self.w + self.b* self.tau_adapt * self.A)
        return self.w + dw
        #da = (-a[t-1] + b * r[t-1]) / tau_a

    def transfer_f(self, t, dt):
        if self.adaptation:
            self.w = self.adaptation_current(t, dt)
        self.A_recordings[:,t] = self.A.copy()
        total_I_ext = self.I_ext_with_noise(dt) + self.extra_stim_values[t]
        self.A = np.where(self.I_syn+total_I_ext-self.w > self.thresh, self.I_syn+total_I_ext -self.w - self.thresh, 0)

    def calculate_total_synaptic_input(self, t, dt): # t as index not seconds
        I_syn = np.zeros(self.size, dtype=float)
        for alpha in self.list_receiving_from:
            t_prev = t - self.delay[alpha.id]/dt
            t_prev = np.where(t_prev<0, 0, t_prev).astype(np.int64)
            I_syn += self.connections[alpha.id] @ self.m_recordings[alpha.id][np.arange(alpha.size) , t_prev]
        self.I_syn = I_syn.copy()


def load_params(params_file):
    if not isinstance(params_file, str):
        raise TypeError("input_params must be a path to json file")
    import json
    f = open(params_file)
    return json.load(f)


def preprocess(data):
    for pop in data.keys():
        if "threshold" not in data[pop]["properties"].keys():
            data[pop]["properties"]["threshold"] = 0.1
        if "range_FR" not in data[pop]["properties"].keys():
            max_FR = data[pop]["properties"]["mean_FR"] + 2*data[pop]["properties"]["sd_FR"]
            data[pop]["properties"]["range_FR"] = [0, max_FR]
    return data

def create_pop(id, nb_neurons: int, n_steps, syn_type, mean_FR, sd_FR, range_FR, I_ext_noise_method, noise_variance="auto", extra_stim_dict = None, thresh=0.1, adaptation=False, b_adapt=0, tau_adapt=1):
    # Checking arguments
    if not type(nb_neurons) is int:
        raise TypeError("nb_neurons: Number of neurons must be int, got %s instead" % (type(nb_neurons)))
    if nb_neurons<=0:
        raise ValueError("nb_neurons: Number of neurons must be positive, got %s instead" % (nb_neurons))
    
    network = Network(id, nb_neurons, n_steps, syn_type, mean_FR, sd_FR, range_FR, I_ext_noise_method, noise_variance, extra_stim_dict=extra_stim_dict, thresh=thresh, adaptation=adaptation, b_adapt=b_adapt, tau_adapt=tau_adapt)
    return network


def calculate_k(N_alpha_real, N_alpha_sim, K_real):
    k = 1/(1/K_real - 1/N_alpha_real + 1/N_alpha_sim)
    return k


def calculate_connectivity(network, data, N_sim, all_to_all=False, Proto_to_itself=True):
    connectivity_params = {}
    K_values_sim = {}
    for alpha in network:
        connectivity_params[alpha] = {}
        K_values_sim[alpha] = {}
        for beta,params in data[alpha]["outgoing_connections"].items():
            if beta not in network:
                continue
            if beta==alpha and not Proto_to_itself:
                continue
            
            connectivity_params[alpha][beta] = {"G": params["mean_W"], "delay": params["mean_delay"], "tau_decay": params["mean_tau_decay"]}
            if all_to_all:
                K_values_sim[alpha][beta] = N_sim
            elif "k_real" in params.keys():
                K_values_sim[alpha][beta] = calculate_k(int(np.prod(data[alpha]["properties"]["n_real"])), N_sim, params["k_real"])
            elif "k_sim" in params.keys():
                K_values_sim[alpha][beta] = params["k_sim"]
    return connectivity_params, K_values_sim


def connect_all_pops(all_pops, connectivity_params, K_values_sim, dt, gaussian_delay=False):
    for alpha in all_pops:
        alpha_name = alpha.id
        if alpha_name in connectivity_params.keys():
            for beta in all_pops:
                beta_name = beta.id
                if beta_name in connectivity_params[alpha_name].keys():
                    same_pop = True if alpha_name == beta_name else False
                    K_sim = K_values_sim[alpha_name][beta_name]
                    connect_two_pops(alpha, beta, K_sim, connectivity_params[alpha_name][beta_name], same_pop, dt, gaussian_delay)
    for pop in all_pops:
        pop.set_I_ext()


def connect_two_pops(alpha, beta, K_sim, params, same_pop, dt, gaussian_delay=False):
    alpha_name = alpha.id

    beta.add_receiving_from(alpha)
    beta.set_K_connections(alpha_name, K_sim)
    if gaussian_delay:
        beta.set_delay_params_distrib(dt, alpha_name, params["delay"])
    else:
        beta.set_delay_params(dt, alpha_name, params["delay"])
    beta.set_tau_decay_params(alpha_name, params["tau_decay"])
    beta.set_m(alpha, alpha_name)
    beta.set_connectivity_matrix(alpha_name, alpha.syn_sign, alpha.size, params["G"], same_pop)


def simulate_network(all_pops, t_sim, dt, return_all=False):
    
    for t in range(int(round(t_sim/dt, 0))):
        for pop in all_pops:
            pop.calculate_total_synaptic_input(t, dt)
            pop.calculate_and_update_m(t, dt)
            pop.transfer_f(t, dt)
    
    if return_all:
        rates_by_pop = {}
        for pop in all_pops:
            rates_by_pop[pop.id] = pop.A_recordings.copy()
        return rates_by_pop

    else:
        average_rates_by_pop = {}
        std_rates_by_pop = {}
        for pop in all_pops:
            average_rates_by_pop[pop.id] = np.mean(pop.A_recordings, axis=0)
            std_rates_by_pop[pop.id] = np.std(pop.A_recordings, axis=0)
        return average_rates_by_pop, std_rates_by_pop


def reset(all_pops):
    for pop in all_pops:
        pop.noise = np.zeros_like(pop.noise, dtype=float)
        pop.A_recordings = np.zeros_like(pop.A_recordings, dtype=float)
        pop.A = np.zeros_like(pop.A, dtype=float)
    for pop in all_pops:
        for alpha in pop.list_receiving_from:
            pop.set_m(alpha, alpha.id)


def reconnect(all_pops, connectivity_params):
    for alpha in all_pops:
        alpha_name = alpha.id
        if alpha_name in connectivity_params.keys():
            for beta in all_pops:
                beta_name = beta.id
                if beta_name in connectivity_params[alpha_name].keys():
                    same_pop = True if alpha_name == beta_name else False
                    beta.set_tau_decay_params(alpha_name, connectivity_params[alpha_name][beta_name]["tau_decay"])
                    beta.set_connectivity_matrix(alpha_name, alpha.syn_sign, alpha.size, connectivity_params[alpha_name][beta_name]["G"], same_pop)
    for pop in all_pops:
        pop.set_I_ext()


def edit_G_values(G_dict: dict, data: dict):
    for k,v in G_dict.items():
        k_split = k.split('_')
        pre=k_split[1]
        post=k_split[-1]
        if pre in data.keys() and post in data[pre]["outgoing_connections"].keys():
            data[pre]["outgoing_connections"][post]["mean_W"] = v
    return data


def get_loop_G_critical(data, loop_params):
    func_dict = {"1": theory_1_nuclei, "2": theory_2_nuclei, "3": theory_3_nuclei, "4": theory_4_nuclei}
    loop_network = np.unique(loop_params["nuclei"]).tolist()
    n = len(loop_network)
    Proto_to_itself = True if n==1 else False

    loop_connectivity_params, _loop_K_values_sim = calculate_connectivity(loop_network, data, N_sim=1000, all_to_all=True, Proto_to_itself=Proto_to_itself)

    sum_delays = np.sum([loop_connectivity_params[alpha][beta]["delay"] for alpha in loop_network for beta in loop_connectivity_params[alpha].keys()])
    list_taus = [loop_connectivity_params[alpha][beta]["tau_decay"] for alpha in loop_network for beta in loop_connectivity_params[alpha].keys()]
    return func_dict[f"{n}"] (loop_params["x0"], list_taus, sum_delays)


def run_rate(params_file, network, t_sim, n_model=1000, dt=1e-4, all_to_all=False, noise_method=None, G_dict=None, extra_stim_dict=dict(), Proto_to_itself=True, noise_variance = "auto", return_all=False, gaussian_delay=False, adaptation=False):

    input_params = load_params(params_file)
    data = preprocess(input_params)

    n_steps = int(round(t_sim/dt, 0))

    all_pops = []


    if G_dict != None:
        data = edit_G_values(G_dict, data)

    # create populations
    for pop in network:
        properties = data[pop]["properties"]
        all_pops.append(create_pop(id=properties["id"], nb_neurons=n_model, n_steps=n_steps, syn_type=properties["type"],
                                   mean_FR=properties["mean_FR"], sd_FR=properties["sd_FR"], range_FR=properties["range_FR"],
                                   I_ext_noise_method=noise_method, noise_variance = noise_variance,
                                   extra_stim_dict=extra_stim_dict, thresh=properties["threshold"],
                                   adaptation=adaptation, b_adapt = properties.get("b", 0), tau_adapt = properties.get("tau_adapt", 1)))

    # connect populations
    connectivity_params, K_values_sim = calculate_connectivity(network, data, n_model, all_to_all, Proto_to_itself)
    connect_all_pops(all_pops, connectivity_params, K_values_sim, dt, gaussian_delay)
    
    # run simulation
    return simulate_network(all_pops, t_sim, dt, return_all)