import numpy as np
import scipy.sparse as sparse
from scipy.ndimage import gaussian_filter1d
import json
import matplotlib.pyplot as plt
#from scipy.stats import truncnorm, norm

from utils import *

# Classes

class Network:
    def __init__(self, id, nb_neurons, n_steps, state, mean_v_rest, sd_v_rest, range_v_rest, mean_v_th, sd_v_th, mean_tau, sd_tau, range_tau,
                 mean_FR, sd_FR, range_FR, nonlinearity_thresh, I_ext_noise_method, noise_variance,
                 noise_amplitude = 1, noise_tau = 0.01, refractory_period = 0.002, extra_stim_dict = dict(), a_adapt = 0, b_adapt = 0, tau_adapt = 0):
        self.id = id                                            # population name
        self.size = nb_neurons                                  # number of neurons
        self.state = state

        # Constants
        self.v_rest = gaussian_trunc_sample(m=mean_v_rest, sd=sd_v_rest, lower_bound=min(range_v_rest), upper_bound = max(range_v_rest), size=self.size)  # 1-D array filled with ranged random normal values
        self.v_th = gaussian_sample(m=mean_v_th, sd=sd_v_th, size=self.size)                      # 1-D array filled with random normal values
        self.tau = gaussian_trunc_sample(mean_tau, sd_tau, min(range_tau), max(range_tau), self.size)              # 1-D array filled with ranged random normal values
        # self.v_rest = np.full(self.size, mean_v_rest)
        # print("HOMOGENEOUS V")
        # self.v_th = np.full(self.size, mean_v_th)
        # self.tau = np.full(self.size, mean_tau)
        
        self.mean_FR = mean_FR
        self.sd_FR = sd_FR
        self.range_FR = range_FR
        self.basal_firing = gaussian_trunc_sample(mean_FR, sd_FR, min(range_FR), max(range_FR), size=self.size) if sd_FR!=0 else np.full(self.size, mean_FR)    #Hz
        # self.basal_firing = np.full(self.size, mean_FR)
        # print("HOMOGENEOUS FR")



        self.nonlinearity_thresh = nonlinearity_thresh      #basal_firing value below which the IF-curve is not linear
        self.extra_stim = extra_stim_dict.get(self.id, np.zeros(n_steps))
        #self.extra_stim = np.zeros(n_steps) if extra_stim==None else extra_stim
        self.I_ext_noise_method = I_ext_noise_method
        self.I_ext_noise_method_dict = {None: no_noise, "Gaussian": Gaussian_noise, "Ornstein-Uhlenbeck": Ornstein_Uhlenbeck_noise}
        #self.I_ext_sd = sd_I_ext

        self.set_noise(noise_variance, noise_amplitude)
        self.noise_tau = noise_tau
        self.noise = np.zeros(self.size)
        self.refractory_period = refractory_period
        self.list_receiving_from = []                           # List of presyn populations (Network class)
        self.K_connections = {}                                 # Dict with presyn population name as keys and int as values
        self.tau_rise = {}                                      # Dict with presyn population name as keys and np.array of float as values (filled with random.normal)
        self.tau_decay = {}                                     # ^ idem
        self.delay = {}                                         # ^ idem
        self.connections = {}                                   # Dict with presyn population name as keys and 2-D connectivity matrix as values (rows=alpha, cols=beta)
        self.a = np.full(self.size, a_adapt, dtype=float)
        self.b = np.full(self.size, b_adapt, dtype=float)
        self.tau_adapt = np.full(self.size, tau_adapt, dtype=float)
        
        self.warnings=0

        # Changing
        self.v = uniform_sample(self.v_rest, self.v_th, self.size)      # 1-D array, between v_rest and v_th for init
        #self.v_hist = np.zeros((self.size, n_steps), dtype=float)
        self.next_v = self.v.copy()                                     # ^idem
        self.spikes = sparse.lil_array((self.size, n_steps), dtype=int) # sparse matrix, rows=neurons, cols=time steps in simulation 
        self.t_last_spike = np.full(self.size, np.inf)                  # 1-D array, all values increase by 1 dt at each dt, individual values reset to 0 at spike times
        self.I_syn = {}                                                 # Dict with presyn population name as keys and np.array as values (empty at init)
        # self.I_syn_hist = np.zeros(n_steps, dtype=float)
        self.I_rise = {}                                                # ^ idem
        self.I_syn_tot = np.zeros(self.size, dtype=float)
        self.w = np.zeros(self.size, dtype=float)


    def set_noise(self, noise_variance, noise_amplitude = 1):
        self.noise_variance = noise_variance
        self.noise_std = np.sqrt(noise_variance)
        self.noise_amplitude = noise_amplitude

    def add_receiving_from(self, alpha_pop):
        self.list_receiving_from.append(alpha_pop)

    def set_K_connections(self, alpha_pop_id, K_sim):
        self.K_connections[alpha_pop_id] = K_sim

    def set_tau_rise_params(self, alpha_pop_id, mean_tau_rise, sd_tau_rise, range_tau_rise):
        self.tau_rise[alpha_pop_id] = gaussian_trunc_sample(mean_tau_rise, sd_tau_rise, min(range_tau_rise), max(range_tau_rise), self.size)
        #self.tau_rise[alpha_pop_id] = np.full(self.size, mean_tau_rise)
        
    def set_tau_decay_params(self, alpha_pop_id, mean_tau_decay, sd_tau_decay, range_tau_decay):
        self.tau_decay[alpha_pop_id] = gaussian_trunc_sample(mean_tau_decay, sd_tau_decay, min(range_tau_decay), max(range_tau_decay), self.size)
        #self.tau_decay[alpha_pop_id] = np.full(self.size, mean_tau_decay)
        
    def set_delay_params(self, alpha_pop_id, mean_delay, sd_delay, range_delay, dt):
        raw_delay = gaussian_trunc_sample(mean_delay, sd_delay, min(range_delay), max(range_delay), self.size)
        #raw_delay = np.full(self.size, mean_delay)
        self.delay[alpha_pop_id] = dt*np.around(raw_delay/dt)   # to match delay precision with dt

    def set_I_syn(self, alpha_pop_id): #alpha_pop_id = id of presynaptic population
        self.I_syn[alpha_pop_id] = np.zeros(self.size)
    
    def set_I_rise(self, alpha_pop_id):
        self.I_rise[alpha_pop_id] = np.zeros(self.size)
    
    def set_connectivity_matrix(self, dt, alpha_pop_id, n_alpha_neurons, alpha_FR, mean_W, sd_G, same_pop = False):
        random_prob_matrix = np.random.rand(n_alpha_neurons, self.size)
        if same_pop:
            np.fill_diagonal(random_prob_matrix, 0)    #avoid autapses when connecting pop to itself by setting prob to 0

        K_sim = self.K_connections[alpha_pop_id]
        prob = K_sim/self.size
        
        #J_matrix = np.where(random_prob_matrix >= 1-prob, 1,0)
        
        if prob!=1:
            prob_vector = 1 - gaussian_sample(m = prob, sd = 0.03, size = self.size) #average prob = Ksim/N, but all beta cells have a different K (normally distributed)
        else:
            prob_vector = np.zeros(self.size)
        J_matrix = np.where(random_prob_matrix>=prob_vector, 1, 0)



        if mean_W==0:
            G_matrix = np.zeros((n_alpha_neurons, self.size))
        else:
            mean_G = np.mean(mean_W * (self.v_th - self.v_rest) / (1+mean_W*self.b*self.tau_adapt) / (K_sim*dt)) #this is if mean_W is a "ratio" weight (like in the rate model)
            # mean_G = mean_W / dt        # this is if mean_W is already in Volt
            reverse_W = np.mean(mean_G*K_sim*dt/(self.v_th - self.v_rest))
            print(f"meanG {alpha_pop_id} to {self.id} = {mean_G} = {np.round(reverse_W,2)}")
            G_matrix = lognormal_sample(m=mean_G, size=np.shape(J_matrix))      # lognormally distributed G
            # G_matrix = np.full((n_alpha_neurons, self.size), mean_G)                        # fixed G for all cells
        connectivity_matrix = np.multiply(J_matrix, G_matrix)
        print(np.sum(connectivity_matrix))
        self.connections[alpha_pop_id] = connectivity_matrix

    def calculate_I_ext_mean(self, dt):
        I_syn=0
        if self.list_receiving_from:
            I_syn_by_alpha = [np.sum(self.connections[alpha.id],1) * alpha.mean_FR * dt for alpha in self.list_receiving_from] * self.tau
            I_syn = np.sum(I_syn_by_alpha, 0)
            print(np.mean(I_syn))
        if self.mean_FR < self.nonlinearity_thresh:
            if self.I_ext_noise_method == "Ornstein-Uhlenbeck":
                if self.id == "STN":
                    with open(f"/home/elena/Documents/data/FI_data/test_noisy5_{self.id}.json", 'r') as f:
                        FI_data = json.load(f)
                else:
                    with open(f"/home/elena/Documents/data/FI_data/FI_data_noisy_{self.state}_{self.id}.json", 'r') as f:
                        FI_data = json.load(f)
            else:
                print("calculate I_ext no noise")
                with open(f"/home/elena/Documents/data/FI_data/FI_data_{self.id}.json", 'r') as f:
                    FI_data = json.load(f)
            I_ext_tot = self.calculate_I_ext_low(FI_data)
        else:
            print("high")
            I_ext_tot = self.calculate_I_ext_high()
        I_ext = I_ext_tot - I_syn
        print(self.id, ":", np.mean(I_ext), "for", self.mean_FR)
        return I_ext

    #########################################################

    def calculate_I_ext_low(self, FI_data):
        I_ext_list = FI_data["I_ext"]
        FR_list = FI_data[self.id]
        idx = [np.where(np.subtract(FR_list, basal_FR)>0)[0][0] for basal_FR in self.basal_firing]
        FR_upper_bound, FR_lower_bound = np.array([FR_list[i] for i in idx]), np.array([FR_list[i-1] for i in idx])
        I_ext_upper_bound, I_ext_lower_bound = np.array([I_ext_list[i] for i in idx]), np.array([I_ext_list[i-1] for i in idx])
        
        # linear_interp = interp1d(I_ext_list, FR_list)
        # linear_results = linear_interp(interpolation_time)

        # cubic_interp = interp1d(time_points, datapoints, kind='cubic')
        # cubic_results = cubic_interp(interpolation_time)



        # plot = sns.scatterplot(x=time_points, y=datapoints, label="Data points", color="green")
        # plot = sns.lineplot(x=interpolation_time, y=cubic_results, label = "Cubic interpolation")
        # plot = sns.lineplot(x=interpolation_time, y=linear_results, label = "Linear interpolation", color="darkorange")
        
        I_ext = linear_interpolation(self.basal_firing, FR_lower_bound, FR_upper_bound, I_ext_lower_bound, I_ext_upper_bound)
        if np.mean(I_ext) > 1: I_ext /= 1000  #in case I_ext is in mV, convert to V
        return I_ext

    def calculate_I_ext_high(self):
        return (self.v_th - self.v_rest)/(1 - np.exp(-(1/self.basal_firing - self.refractory_period) / self.tau))

    def sum_synaptic_inputs(self, t, dt):
        for alpha_pop in self.list_receiving_from:
            alpha_pop_id = alpha_pop.id

            sum_inputs = self.connections[alpha_pop_id] @ \
                alpha_pop.spikes[np.arange(alpha_pop.size), t - self.delay[alpha_pop_id]/dt].toarray().reshape(-1)

            dI_rise = dt * (-self.I_rise[alpha_pop_id] + sum_inputs * self.tau) / self.tau_rise[alpha_pop_id]
            dI_syn = dt * (-self.I_syn[alpha_pop_id] + self.I_rise[alpha_pop_id]) / self.tau_decay[alpha_pop_id]
            self.I_rise[alpha_pop_id] += dI_rise
            self.I_syn[alpha_pop_id] += dI_syn


            """
            dI_rise = 0.5 * dt * ((-self.I_rise[alpha_pop_id] + sum_inputs * self.tau)) / self.tau_rise[alpha_pop_id]
            dI_syn = 0.5 * dt * ((-self.I_syn[alpha_pop_id] + self.I_rise[alpha_pop_id]) / self.tau_decay[alpha_pop_id])
            self.I_rise[alpha_pop_id] += dI_rise
            self.I_syn[alpha_pop_id] += dI_syn
            
            dI_rise = 0.5 * dt * ((-self.I_rise[alpha_pop_id] + sum_inputs * self.tau)) / self.tau_rise[alpha_pop_id]
            dI_syn = 0.5 * dt * ((-self.I_syn[alpha_pop_id] + self.I_rise[alpha_pop_id]) / self.tau_decay[alpha_pop_id])
            self.I_rise[alpha_pop_id] += dI_rise
            self.I_syn[alpha_pop_id] += dI_syn
            """
        
        if bool(self.I_syn)==True:
            self.I_syn_tot = np.sum([v for v in self.I_syn.values()], axis=0)
        # self.I_syn_hist[t] = np.mean(self.I_syn_tot)

    def I_ext_with_noise(self, dt):
        self.noise = self.I_ext_noise_method_dict[self.I_ext_noise_method] (self.noise_amplitude, self.noise_std, self.size, dt, np.sqrt(dt), self.noise_tau, self.noise)
        return self.I_ext + self.noise

    def calculate_next_v(self, t, dt):
        self.w = self.adaptation_current(t, dt)
        self.next_v = self.v.copy()
        I_ext = self.I_ext_with_noise(dt) + self.extra_stim[t]
        ind = self.cells_not_in_refractory_period()
        #dv = dt * ((-(self.v - self.v_rest) + self.I_syn_tot + I_ext) / self.tau)
        #self.next_v[ind] = self.v[ind] + dv[ind]
        next_v = Runge_Kutta_2nd_order(dt, self.v[ind], self.v_rest[ind], self.tau[ind], self.I_syn_tot[ind], I_ext[ind], self.w[ind])
        self.next_v[ind] = next_v

    def adaptation_current(self, t, dt):
        dw = dt/self.tau_adapt * \
            (self.a * (self.v - self.v_rest) \
            - self.w \
            + self.b * self.tau_adapt * self.spikes[:, [t-1]].toarray().reshape(-1) )
        return self.w + dw

    def cells_not_in_refractory_period(self):
        return self.t_last_spike > self.refractory_period

    def cells_above_spiking_threshold(self):
        return self.next_v > self.v_th

    def record_spikes_and_reset(self, t, dt):
        ind = self.cells_above_spiking_threshold()
        self.spikes[ind, t] = 1
        self.t_last_spike[ind] = 0
        #self.next_v[ind] = self.v_rest[ind].copy()
        self.next_v[ind] = Hansel_linear_interpolation(dt, self.v[ind], self.next_v[ind], self.v_rest[ind], self.v_th[ind], self.tau[ind])
        ##checking if any cells were reset above their v_th: if so, count how many and manually set them to v_th
        ind2 = self.cells_above_spiking_threshold()
        if True in ind2:
            self.warnings += np.count_nonzero(ind2 == True)
            self.next_v[ind2] = self.v_th[ind2]
        self.v = self.next_v.copy()
        self.t_last_spike += dt
 
##############################################

def load_params(params_file):
    if not isinstance(params_file, dict):
        if not isinstance(params_file, str):
            raise TypeError("input_params must be a dict or a path to json file")
        import json
        f = open(params_file)
        return json.load(f)

def preprocess(data, species, rate=False):
    for pop in data.keys():
        if "threshold" not in data[pop]["properties"].keys():
            data[pop]["properties"]["threshold"] = 0.1
    if rate and species=="rat":
        data["STN"]["outgoing_connections"]["Proto"]["mean_tau_decay"] = 6e-3
        print("rate?")
    return data

def edit_W_values(W_dict: dict, data: dict):
    for k,v in W_dict.items():
        k_split = k.split('_')
        pre=k_split[1]
        post=k_split[-1]
        # print(pre, post)
        data[pre]["outgoing_connections"][post]["mean_W"] = v
    return data

def create_pop(id, nb_neurons: int, n_steps, state, mean_v_rest, sd_v_rest, range_v_rest, mean_v_th, sd_v_th, mean_tau, sd_tau, range_tau, mean_FR, sd_FR, range_FR, nonlinearity_thresh, I_ext_noise_method, noise_variance, extra_stim_dict, a_adapt, b_adapt, tau_adapt):
    # Checking arguments
    if not type(nb_neurons) is int:
        raise TypeError("nb_neurons: Number of neurons must be int, got %s instead" % (type(nb_neurons)))
    if nb_neurons<0:
        raise ValueError("nb_neurons: Number of neurons must be positive, got %s instead" % (nb_neurons))
    
    network = Network(id, nb_neurons, n_steps, state, mean_v_rest, sd_v_rest, range_v_rest, mean_v_th, sd_v_th, mean_tau, sd_tau, range_tau, mean_FR, sd_FR, range_FR, nonlinearity_thresh, I_ext_noise_method, noise_variance, extra_stim_dict=extra_stim_dict, a_adapt=a_adapt, b_adapt=b_adapt, tau_adapt=tau_adapt)

    return network
    
def calculate_k(N_alpha_real, N_alpha_sim, K_real):
    k = 1/(1/K_real - 1/N_alpha_real + 1/N_alpha_sim)
    return k

def calculate_k_values_sim(N_sim, K_values_real):
    K_values_sim = {}
    for alpha, alpha_params in K_values_real.items():
        K_values_sim[alpha] = {}
        N_alpha_real = alpha_params['n_real']
        for beta,k_real in alpha_params['k_values'].items():
            K_values_sim[alpha][beta] = calculate_k(N_alpha_real, N_sim, k_real)
    return K_values_sim

def calculate_all_to_all(N_sim, K_values_real):
    K_values_sim = {}
    for alpha,params in K_values_real.items():
        K_values_sim[alpha] = {}
        for beta in params['k_values'].keys():
            K_values_sim[alpha][beta] = N_sim
    return K_values_sim

def calculate_connectivity(network, data, N_sim, all_to_all=False, Proto_to_itself=True):
    connectivity_params = {}
    #K_values_real = {}
    K_values_sim = {}
    if all_to_all:
        print("connecting all-to-all")
    for alpha in network:
        connectivity_params[alpha] = {}
        K_values_sim[alpha] = {}
        #K_values_real[alpha] = {"n_real": int(np.prod(data[alpha]["properties"]["n_real"])), "k_values": {}}
        for beta,params in data[alpha]["outgoing_connections"].items():
            if beta not in network:
                continue
            if beta==alpha and not Proto_to_itself:
                continue
            connectivity_params[alpha][beta] = params
            if all_to_all:
                K_values_sim[alpha][beta] = N_sim
            elif "k_real" in params.keys():
                K_values_sim[alpha][beta] = calculate_k(int(np.prod(data[alpha]["properties"]["n_real"])), N_sim, params["k_real"])
                #K_values_real[alpha]["k_values"][beta] = params["k_real"]
            elif "k_sim" in params.keys():
                K_values_sim[alpha][beta] = params["k_sim"]
    return connectivity_params, K_values_sim

def connect_all_pops(all_pops: list[Network], connectivity_params, K_values_sim, dt, I_ext=None):
    for alpha in all_pops:
        alpha_name = alpha.id
        if alpha_name in connectivity_params.keys():
            for beta in all_pops:
                beta_name = beta.id
                if beta_name in connectivity_params[alpha_name].keys():
                    same_pop = True if alpha_name == beta_name else False
                    K_sim = K_values_sim[alpha_name][beta_name]
                    connect_two_pops(alpha, beta, K_sim, connectivity_params[alpha_name][beta_name], same_pop, dt)

    if I_ext==None:
        for pop in all_pops:
            pop.I_ext = pop.calculate_I_ext_mean(dt)
    else:
        for pop in all_pops:
            pop.I_ext = I_ext[pop.id]


def connect_two_pops(alpha, beta, K_sim, params, same_pop, dt):
    alpha_name = alpha.id

    beta.add_receiving_from(alpha)
    beta.set_K_connections(alpha_name, K_sim)
    beta.set_tau_rise_params(alpha_name, params['mean_tau_rise'], params['sd_tau_rise'], params['range_tau_rise'])
    beta.set_tau_decay_params(alpha_name, params['mean_tau_decay'], params['sd_tau_decay'], params['range_tau_decay'])
    beta.set_delay_params(alpha_name, params['mean_delay'], params['sd_delay'], params['range_delay'], dt)
    beta.set_I_syn(alpha_name)
    beta.set_I_rise(alpha_name)
    beta.set_connectivity_matrix(dt, alpha_name, alpha.size, alpha.mean_FR, params['mean_W'], same_pop)


def simulate_network(all_pops, t_sim, dt=0.001):

    for t in range(int(t_sim/dt)):
        # first calculate next v for everyone
        for pop in all_pops:
            pop.sum_synaptic_inputs(t, dt)
            pop.calculate_next_v(t,dt)

        # record spikes, calculate v_reset for spiking cells, update v for all
        for pop in all_pops:
            pop.record_spikes_and_reset(t, dt)

    all_spikes_recordings = {}
    for pop in all_pops:
        # if pop.id=="Proto":
        #     plt.show()
        #     plt.plot(pop.I_syn_hist)
        #     plt.show()
        all_spikes_recordings[pop.id] = [np.nonzero(i)[0].tolist() for i in pop.spikes.toarray().tolist()]  # transform sparse list of lists of 0 and 1 (1 = spike) into list of lists of indices of spikes
        if pop.warnings != 0:
            print(pop.id, ":", pop.warnings, "warnings recorded.")

    
    return all_spikes_recordings

def reset(all_pops, n_steps):
    for pop in all_pops:
        pop.v = uniform_sample(pop.v_rest, pop.v_th, pop.size)
        pop.next_v = pop.v.copy()
        pop.spikes = sparse.lil_array((pop.size, n_steps), dtype=int)
        pop.t_last_spike = np.full(pop.size, np.inf)
        pop.noise = np.zeros_like(pop.noise)
        pop.I_syn_tot = np.zeros_like(pop.I_syn_tot)
        for k in pop.I_syn.keys():
            pop.I_syn[k] = np.zeros(pop.size)
            pop.I_rise[k] = np.zeros(pop.size)


def spike_list_to_binary_signal(spike_list, n_steps):
    signal = np.zeros(n_steps)
    for i in spike_list:
        signal[i] = 1
    return signal

def spikes_to_std_rates(spikes: dict, n_steps, dt):
    std_rates = {k:np.std([spike_list_to_binary_signal(signal, n_steps) for signal in v], axis=0) for k,v in spikes.items()}
    return std_rates

def spikes_to_mean_rate(spikes: dict, n_steps, dt):
    rates = {k:np.divide(np.mean([spike_list_to_binary_signal(signal, n_steps) for signal in v], axis=0), dt) for k,v in spikes.items()}
    return rates

def square_moving_average(X, n):
    '''Return the moving average over X with window n without changing dimensions of X'''
    z2= np.cumsum(np.pad(X, (n,0), 'constant', constant_values=0))
    z1 = np.cumsum(np.pad(X, (0,n), 'constant', constant_values=X[-1]))
    return (z1-z2)[(n-1):-1]/n


def exp_moving_average(a, winSize, alpha=1):
    ws = alpha ** np.arange(winSize)
    w_sum = ws.sum()
    # ema_mean = np.convolve(a, ws)[winSize-1:] / w_sum
    ema_mean = np.convolve(a, ws/w_sum, mode="same")
    return ema_mean

def gaussian_smoothing(a, sigma=2e-3):
    return gaussian_filter1d(a, sigma)

def spikes_to_smoothed_rates(spikes: dict, n_steps, dt, rolling_window = 0.005, method = "exponential"):
    method_funcs = {"square": square_moving_average, "gaussian": gaussian_smoothing, "exponential": exp_moving_average}

    rates = {k:[np.divide(spike_list_to_binary_signal(signal, n_steps), dt) for signal in v] for k,v in spikes.items()}
    smoothed = {k:[method_funcs[method] (signal, int(rolling_window/dt)) for signal in v] for k,v in rates.items()}

    return smoothed

def spikes_to_smoothed_mean_rate(spikes: dict, n_steps, dt, rolling_window = 0.005, method = "exponential"):
    
    smoothed = spikes_to_smoothed_rates(spikes, n_steps, dt, rolling_window, method)
        
    # for v in smoothed["Proto"][:3]:
    #     plt.plot(v)
    # plt.show()
    # exit()

    # averaged = np.mean(smoothed["Proto"], axis=1)
    # plt.plot(smoothed["Proto"][np.argmax(averaged)], color="r")
    # plt.show()
    # exit()

    mean_rates = {k:np.mean(v, axis=0) for k,v in smoothed.items()}
    #std_rates = {k:np.std(v, axis=0) for k,v in smoothed.items()}
    std = {k:np.std(v) for k,v in mean_rates.items()}
    
    return mean_rates, std


def run_LIF(params_file, species, populations_list, t_sim, n_model=1000, dt=1e-4, state="Ctrl", all_to_all=False, noise=None, W_dict=None, extra_stim_dict=dict(), I_ext=None, Proto_to_itself=True, noise_variance = {"Proto": 12, "STN": 7, "D1": 20, "D2": 19, "Arky": 15, "FSI": 24, "Ctx": 2, "GPi": 13, "Th": 17}):
    input_params = load_params(params_file)
    data = preprocess(input_params, species, rate=False)

    n_steps = int(round(t_sim/dt, 0))

    I_ext_noise_method = noise

    network = populations_list
    all_pops = []


    ###############

    if W_dict != None:
        data = edit_W_values(W_dict, data)


    # create populations
    for pop in network:
        properties = data[pop]["properties"]
        all_pops.append(create_pop(id=properties["id"], nb_neurons=n_model, n_steps=n_steps, state=state,
                                   mean_v_rest=properties["mean_v_rest"], sd_v_rest=properties["sd_v_rest"], range_v_rest=properties["range_v_rest"],
                                   mean_v_th=properties["mean_v_th"], sd_v_th=properties["sd_v_th"],
                                   mean_tau=properties["mean_tau"], sd_tau=properties["sd_tau"], range_tau=properties["range_tau"],
                                   mean_FR=properties[f"mean_FR_{state}"], sd_FR=properties[f"sd_FR_{state}"], range_FR=properties["range_FR"], nonlinearity_thresh=properties["nonlinearity_thresh"],
                                   I_ext_noise_method=I_ext_noise_method, noise_variance=noise_variance[pop], extra_stim_dict = extra_stim_dict,
                                   a_adapt = properties.get("a", 0), b_adapt = properties.get("b", 0), tau_adapt = properties.get("tau_adapt", 1)))


    # connect populations
    connectivity_params, K_values_sim = calculate_connectivity(network, data, n_model, all_to_all, Proto_to_itself)
    connect_all_pops(all_pops, connectivity_params, K_values_sim, dt, I_ext)

    # run simulation
    spikes = simulate_network(all_pops, t_sim, dt)

    return spikes