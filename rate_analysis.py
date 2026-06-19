import numpy as np, pandas as pd, scipy.stats as stat
import scipy.signal as sig, scipy.fft as sfft
import random, json
#from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, FixedLocator, FixedFormatter, AutoMinorLocator
from itertools import combinations, combinations_with_replacement
from astropy.stats import rayleightest

from LIF_model import spikes_to_smoothed_mean_rate
from utils import bandpass_filter


def plot_table(W_dict):
    W_df = pd.DataFrame({"projection":[k for k in W_dict.keys()], "G": [v for v in W_dict.values()]})
    #fig, ax = plt.subplots()
    #fig.set_figwidth(3)

    # hide axes
    #fig.patch.set_visible(False)
    # ax.axis('off')
    # ax.axis('tight')
    plt.axis('off')
    plt.axis('tight')
    plt.table(cellText=W_df.values, colLabels=W_df.columns, loc='center').auto_set_column_width(col=list(range(len(W_df.columns))))


def plot_full_table(G_dict: dict, pops: list):
    from itertools import combinations
    loops = json.load(open("params/graphical_params.json"))["loops"]
    loops_in_sim = []
    G_loops = []
    for loop, params in loops.items():
        if(all(x in pops for x in params["nuclei"])):
            G_loop = 1
            for A, B in combinations(params["nuclei"], 2):             
                G_loop *= G_dict.get(f"G_{A}_to_{B}", 1)
                if A!=B:
                    G_loop *= G_dict.get(f"G_{B}_to_{A}", 1)
            loops_in_sim.append(loop)
            G_loops.append(round(G_loop, 6))
    df = pd.DataFrame({"loop": loops_in_sim, "G*": G_loops})
    plt.axis('off')
    plt.axis('tight')
    plt.table(cellText=df.values, colLabels=df.columns, loc='center').auto_set_column_width(col=list(range(len(df.columns))))


def rate_to_psth(rate, dt, bin_size):
    duration = len(rate)*dt
    psth, _, __ = stat.binned_statistic(np.arange(len(rate)), rate, statistic='mean', bins = int(duration/bin_size))
    return psth


def get_events_onsets(bins_list, bin_size, n_bins=3): #n_bins = minimal number of consecutive bins for an event
    onsets = []
    i = 0
    ongoing_event=False
    while i<len(bins_list)-n_bins:
        bin = bins_list[i]
        if not ongoing_event and bin+n_bins-1 == bins_list[i+n_bins-1]:
            onsets.append(bin*bin_size)
            ongoing_event=True
        if ongoing_event and bin+n_bins not in bins_list:
            end = (bin+n_bins)*bin_size
            ongoing_event=False
        i+=1
    while len(onsets)<3:
        onsets.append(np.nan)
    return onsets


def plot_rate(rates, t_sim, dt, skip=0, xaxis_len=2, std=None, bin_size_ms = 1.0, zero_time=0, ax=None):
    if ax is None:
        ax = plt.gca()
    skip_idx = int(round(skip/dt, 0))
    colors = json.load(open("params/graphical_params.json"))["nuclei"]
    x = np.arange(0, t_sim, bin_size_ms/1000) - zero_time
    ymax=0
    for k in rates.keys():
        rate = rates[k] if np.ndim(rates[k])==1 else np.mean(rates[k], axis=0)
        alpha = 0.6 if k=="FSI" else 1
        duration = len(rate)*dt
        psth, _, __ = stat.binned_statistic(np.arange(len(rate)), rate, statistic='mean', bins = int(duration/(bin_size_ms/1000)))
        ax.plot(x, psth, label=k, color=colors[k], alpha=alpha)
        if np.max(rate[skip_idx:])>ymax: ymax=np.max(rate[skip_idx:])
        if std!=None:
            std_k = std[k]
            psth_std, _, __ = stat.binned_statistic(np.arange(len(std_k)), std_k, statistic='mean', bins = int(duration/(bin_size_ms/1000)))
            ax.fill_between(x, psth-psth_std, psth+psth_std, alpha=0.4, color=colors[k])
    ax.set_ylabel('Firing rate (spks/s)')
    ax.set_xlabel("Time (s)")
    ax.set_xlim(skip-zero_time, min(skip+xaxis_len-zero_time, t_sim-zero_time))
    ax.set_ylim(0, ymax+15)
    ax.grid(False)
    ax.set_title("Average firing rate by population")
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(loc="upper right")
    

def plot_raster(spikes, t_sim, dt, skip=0, n_raster=20, n_model=1000, xaxis_len=2):
    colors = json.load(open("params/graphical_params.json"))["nuclei"]
    fig, ax = plt.subplots(len(spikes.keys()), 1, sharex=True, squeeze=False)
    #fig.set_figwidth(max(6.4, 6*(t_sim-skip)))
    fig.set_figheight(1.5*len(spikes.keys()))
    for i,k in enumerate(spikes.keys()):
        sample = random.sample(range(n_model), min(n_raster, n_model))
        if sample==1:
            ax[i][0].eventplot([np.multiply(spikes[k],dt)], color=colors[k])    
        else:
            ax[i][0].eventplot([np.multiply(spikes[k][s],dt) for s in sample], color=colors[k])
        ax[i][0].set_title(k)
        ax[i][0].set_xlim(skip, min(skip+xaxis_len, t_sim))
        #ax[i].set_xticklabels(['{:3.1f}'.format(x*dt) for x in ax[i].get_xticks()])
    ax[-1][0].set_xlabel("Time (s)")
    return fig


def is_stable_oscillation(signal, n_ref_peak=1, last_first_peak_ratio_thresh = [0.99, 1.1]):
    peaks, properties = sig.find_peaks(signal, prominence=0) #setting prominences to access it in properties but not excluding any peaks
    mid = int(len(peaks)//2)
    if len(peaks) > 10 and np.max(properties["prominences"][mid:-1]) > 0.5:  #ignoring prominence of last peak which can by biased by the truncated signal
        # print(peaks[1 : mid])
        # print(signal[peaks[1 : mid]])
        # print(max(signal[peaks[1 : mid]]))
        last_first_peak_ratio = max(signal[peaks[mid:-1]]) / max(signal[peaks[ : mid]])
    
    # if len(peaks) > n_ref_peak+3 and np.max(properties["prominences"][-4:-1]) > 0.1:  #ignoring prominence of last peak which can by biased by the truncated signal
    #     last_first_peak_ratio = max(signal[peaks[-4:-1]]) / max(signal[peaks[n_ref_peak : n_ref_peak+3]]) # in case signal has oscillations with 3 peaks: compares highest
    #     # print(last_first_peak_ratio)
    
    else:
        return False, 0
    if last_first_peak_ratio_thresh[0] < last_first_peak_ratio < last_first_peak_ratio_thresh[1]:
        return True, last_first_peak_ratio
    return False, last_first_peak_ratio


def compute_fft_of_signal(signal, dt, window_size=1, freq_max = 100):
    N = int(round(window_size/dt, 0))
    Y = sfft.fft(signal, N)
    fv = sfft.fftfreq(N, dt)[:N//2]
    max_idx = np.where(fv > freq_max)[0][0]
    return fv[1:max_idx], 2.0/N * np.abs(Y[1:max_idx])


def get_peak_f_from_fft(f, fft):
    return f[np.argmax(fft)]


def compute_welch_of_signal(signal, dt, window_size=1):
    return sig.welch(signal, fs=int(1/dt), window=sig.get_window("hann", int(window_size/dt)), detrend='linear')


def get_filter_bounds(window_size, min, max):
    return int(np.ceil(min*window_size)), int(np.floor(max*window_size))


def get_peak_f_from_welch(f, Pxx, window_size=1):
    min, max = 3, 100       #ignore frequencies below 3 Hz and above 100 Hz
    if window_size!=1:
        min, max = get_filter_bounds(window_size, min, max)
    return f[np.argmax(Pxx[min:max])+min]


def get_peak_f_of_signal(signal, dt, window_size=1):
    f, Pxx = compute_welch_of_signal(signal, dt, window_size)
    return get_peak_f_from_welch(f, Pxx, window_size)


def plot_fft(rates: dict, dt, window_size=1, skip=0, fmax=70, ax=None):
    if ax is None:
        ax = plt.gca()
    colors = json.load(open("params/graphical_params.json"))["nuclei"]

    f_list={}
    fft_list={}
    for pop, rate in rates.items():
        if np.ndim(rate)==2:
            f, fft = np.mean([compute_fft_of_signal(signal[int(skip/dt):], dt, window_size) for signal in rate], axis=0)            
        elif np.ndim(rate)==1:
            signal = rate[int(skip/dt):]
            f, fft = compute_fft_of_signal(signal, dt, window_size)
        else:
            print(np.ndim(rates))
            raise ValueError("rates has wrong number of dimensions (check return_all)")
        f_list[pop] = f
        fft_list[pop] = fft
    
    ref_nucleus = "Proto" if "Proto" in rates.keys() else [k for k in rates.keys()][0]
    mean_max_f = get_peak_f_from_fft(f_list[ref_nucleus], fft_list[ref_nucleus])
    plt.axvline(mean_max_f, ls='dashed', color="slategray", label=f"F(peak) = {mean_max_f} Hz")

    for k,v in f_list.items():
        ax.plot(v, fft_list[k], label=k, color=colors[k])
    ax.set_ylabel('FFT')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xlim(0,fmax)
    #plt.title("Power spectral density (Welch)")
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend(loc="upper right")


def plot_welch(rates: dict, dt, noise_method, window_size=1, skip=0, fmax=100, ax=None):
    if ax is None:
        ax = plt.gca()
    colors = json.load(open("params/graphical_params.json"))["nuclei"]
    #sos = sig.butter(3, 200, btype='low', fs=1000, output='sos')
    #filtered = sig.sosfil[:100]t(sos, wave)

    f_list={}
    Pxx_den_list={}
    for pop, rate in rates.items():
        if np.ndim(rate)==2:
            f, Pxx_den = np.mean([compute_welch_of_signal(signal[int(skip/dt):], dt, window_size) for signal in rate], axis=0)            
        elif np.ndim(rate)==1:
            signal = rate[int(skip/dt):]
            f, Pxx_den = compute_welch_of_signal(signal, dt, window_size)
        else:
            print(np.ndim(rates))
            raise ValueError("rates has wrong number of dimensions (check return_all)")
        f_list[pop] = f
        Pxx_den_list[pop] = Pxx_den
    ref_nucleus = "Proto" if "Proto" in rates.keys() else [k for k in rates.keys()][0]
    # stable_oscill = True if noise_method == None and is_stable_oscillation(rates[ref_nucleus][int(skip/dt):])[0] else False

    # if stable_oscill or noise_method!= None:
    mean_max_f = get_peak_f_from_welch(f_list[ref_nucleus], Pxx_den_list[ref_nucleus])
    # ax.axvline(mean_max_f, ls='dashed', color="slategray", label=f"_Mean frequency at peak: {mean_max_f} Hz")

    for k,v in f_list.items():
        ax.plot(v, Pxx_den_list[k], label=k, color=colors[k])
    ax.set_ylabel('PSD')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xlim(0,fmax)
    ax.grid(True)
    ax.spines[['right', 'top']].set_visible(False)
    #plt.title("Power spectral density (Welch)")
    ax.legend(loc="upper right")


def plot_phase_diagram(oscillating, stable, xlabel, ylabel, G_theory=None, maxlim=3, title="", linear=False, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.scatter(oscillating["x"], oscillating["y"], marker="P", color="forestgreen", label="Oscillations", alpha=0.8)
    ax.scatter(stable["x"], stable["y"], marker="X", color="firebrick", label="Stable", alpha=0.8)

    if G_theory != None:
        if linear:
            x = np.arange(-0.1,maxlim,0.1)
            y = abs(G_theory)-x
        else:
            x = np.arange(0.1,maxlim,0.1)
            y = abs(G_theory)/x
        ax.plot(x,y, color="black")
    ax.set_xlim(0,maxlim)
    ax.set_ylim(0,maxlim)
    ax.set_xlabel(r'$G_{'+xlabel+'}$', fontsize=12)
    ax.set_ylabel(r'$G_{'+ylabel+'}$', fontsize=12)
    ax.set_title(title)


def plot_mean_coherences(rates, pops, ref_pop, dt, window_size=0.5, n_pairs=1000, ax=None):
    if ax is None:
        ax = plt.gca()
    colors = json.load(open("params/graphical_params.json"))["nuclei"]
    ref_rates = rates[ref_pop]
    for pop in pops:
        same_pop = (ref_pop==pop)
        other_rates = rates[pop]

        f, Cxy = sig.coherence(ref_rates[0], other_rates[-1], fs=1/dt, nperseg=window_size/dt)
        sum_n_Cxy = np.zeros(len(f))

    # n_peak_freqs = np.zeros(n_pairs)
        random_pairs = generate_n_random_pairs(n_pairs, len(ref_rates), len(other_rates), same_pop)
        for i,(a,b) in enumerate(random_pairs):
            f, Cxy = sig.coherence(ref_rates[a], other_rates[b], fs=1/dt, nperseg=window_size/dt)
            sum_n_Cxy+=Cxy
        mean_Cxy = np.divide(sum_n_Cxy, n_pairs)
        ax.plot(f[1:], mean_Cxy[1:], color=colors[pop], label=pop)
    ax.set_xlim(0,100)
    ax.set_title(f"Mean coherence with {ref_pop} of n={n_pairs} neuron pairs")
    

def get_phase_angle(signal, ref_signal, dt, window_size=0.5):
    f, Cxy = sig.coherence(signal, ref_signal, fs=1/dt, nperseg=window_size/dt)
    f_, Pxy = sig.csd(signal, ref_signal, fs=1/dt, nperseg=window_size/dt)
    print(np.shape(Pxy))
    idxs = np.argmax(Cxy, axis=1)
    return np.angle(np.array([Pxy[i,idx] for i,idx in enumerate(idxs)]))


def generate_n_random_pairs(n_pairs, sample_size_a, sample_size_b, same_pop=False):
    all_pairs = [(x,y) for y in range(sample_size_b) for x in range(sample_size_a)]
    if same_pop: ##avoid auto-pairs
        for x in range(sample_size_a):
            all_pairs.remove((x,x))
    random_indexes = np.random.choice(len(all_pairs), size=n_pairs, replace=False)
    return [all_pairs[idx] for idx in random_indexes]


def get_pop_phase_angles(data, ref_data, peak_freq, dt, window_size, n_pairs, same_pop=False):
    distrib = np.zeros(n_pairs)
    random_pairs = generate_n_random_pairs(n_pairs, len(data), len(ref_data), same_pop)
    for i, (a,b) in enumerate(random_pairs):
        f, Pxy = sig.csd(data[a], ref_data[b], fs=1/dt, nperseg=window_size/dt)
        freq_idx = np.argmin(np.abs(f-peak_freq))
        distrib[i] = np.angle(Pxy[freq_idx], deg=True)
    return distrib


def get_peak_frequency_from_coherence(rates1, rates2, same_pop, dt, window_size=1, n_pairs=1000, freq_max=100):
    f, _Cxy = sig.coherence(rates1[0], rates2[-1], fs=1/dt, nperseg=window_size/dt)
    sum_n_Cxy = np.zeros(len(f))

    # n_peak_freqs = np.zeros(n_pairs)
    random_pairs = generate_n_random_pairs(n_pairs, len(rates1), len(rates2), same_pop)
    for i,(a,b) in enumerate(random_pairs):
        f, Cxy = sig.coherence(rates1[a], rates2[b], fs=1/dt, nperseg=window_size/dt)
        sum_n_Cxy+=Cxy
    mean_Cxy = np.divide(sum_n_Cxy, n_pairs)
    print(mean_Cxy)
    mean_Cxy_cropped = np.where((f>0) & (f<=freq_max), mean_Cxy, 0)
    peak_freq = f[np.argmax(mean_Cxy_cropped)]
    return peak_freq, np.max(mean_Cxy_cropped)


def plot_relative_phase_angles(rates, pops_sublist, ref_pop, dt, window_size=1, skip=0, loop_name=None, n_pairs=1000, freq_method="coherence", bin_size=1,  ax=None, normalize=True):
    if ax is None:
        ax = plt.gca()

    if loop_name!= None:
        pops = json.load(open("params/graphical_params.json"))["loops"][loop_name]["nuclei"]
    else:
        pops = pops_sublist
    if np.ndim(rates[pops[0]])!=2:
        raise ValueError("pass all individual neurons rates, not average population rates")    
    
    if freq_method=="coherence":
        if "Proto" in pops:
            rates_Proto = rates["Proto"][: , int(skip/dt):]
            peak_freq, _ = get_peak_frequency_from_coherence(rates_Proto, rates_Proto, same_pop=True, dt=dt, window_size=window_size, n_pairs=n_pairs)
        else:
            peak_freqs = []
            values = []
            for A,B in combinations_with_replacement(pops, 2):
                if A in rates.keys() and B in rates.keys():
                    rates_A = rates[A][: , int(skip/dt):]
                    rates_B = rates[B][: , int(skip/dt):]
                    f, v = get_peak_frequency_from_coherence(rates_A, rates_B, same_pop=(A==B), dt=dt, window_size=window_size, n_pairs=n_pairs)
                    peak_freqs.append(f)
                    values.append(v)
            print(max(values))
            peak_freq = peak_freqs[np.argmax(values)]
    elif freq_method=="fft":
        f, fft = compute_fft_of_signal(np.mean(rates["Proto"], 0)[int(skip/dt):], dt)
        peak_freq = f[np.argmax(fft)]
    else:
        peak_freq = float(freq_method)
    print("freq used for phase:",peak_freq)

    colors = json.load(open("params/graphical_params.json"))["nuclei"]
    n_bins = int(round(360/bin_size, 0))
    width = (2*np.pi) / n_bins
    theta = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
    #  width = (2*np.pi) / 360
    # theta = np.linspace(np.pi, -np.pi, int(round(360/bin_size,0)), endpoint=False)[::-1] ##create array backwards and reversing to get like startpoint=False and endpoint=True (?)

    rates_sublist = {k:rates[k] for k in pops_sublist}
    ref_data = rates_sublist[ref_pop][:,int(skip/dt):]
    for i,(pop, data) in enumerate(rates_sublist.items()):
        print(pop)
        test_data = data[:,int(skip/dt):]
        angle_distrib = get_pop_phase_angles(test_data, ref_data, peak_freq=peak_freq, dt=dt, window_size=window_size, n_pairs=n_pairs, same_pop=(pop==ref_pop))
        
        bar_data = np.histogram(np.where(angle_distrib==180, -180, angle_distrib), bins=n_bins, range=(-180,180))[0]
        # print(np.sum(bar_data))
        print("mean angle:",np.mean(angle_distrib))
        print("n nonzero:", len(np.where(bar_data!=0)[0]))
        if normalize:
            ax.bar(theta, bar_data/np.max(bar_data), width=width, color=colors[pop], label=pop)
        else:
            ax.bar(theta, bar_data, width=width, color=colors[pop], label=pop)
        mean_angle = np.round(np.rad2deg(stat.circmean(np.deg2rad(angle_distrib), high = np.pi, low = -np.pi)), 1)
        std_angle = np.round(np.rad2deg(stat.circstd(np.deg2rad(angle_distrib), high = np.pi, low = -np.pi)), 1)
        print(pop, mean_angle, std_angle)
        # ax.text(-0.3-0.1*i, 1.4, f"{mean_angle}° ± {std_angle}°", color=colors[pop])
        ax.text(1.08, 0.6 - 0.07*i, f"{mean_angle}° ± {std_angle}°", color=colors[pop],
                transform=ax.transAxes, ha='left', va='top')
    ax.set_yticks([])
    ax.set_thetagrids(np.arange(0, 360, 45), labels=['0°', '45°', '90°', '135°', '±180°', '-135°', '-90°', '-45°'])

    ax.legend(loc="upper left", bbox_to_anchor=(.8 + np.cos(1.15)/2, .6 + np.sin(1.15)/2))
    ax.set_title(f"f={int(peak_freq)} Hz, n={n_pairs} random pairs")

def get_ref_signal_peaks(spikes, ref_pop, dt, t_sim, skip):
    ref_signal = spikes_to_smoothed_mean_rate({ref_pop: spikes[ref_pop]}, int(round(t_sim/dt, 0)), dt)[0][ref_pop]
    ref_signal_filtered = bandpass_filter(ref_signal, 1/dt, 8, 70)
    peak_f = get_peak_f_of_signal(ref_signal_filtered, dt)

    amplitude_envelope = np.abs(sig.hilbert(ref_signal_filtered))
    peak_height_min = np.percentile(amplitude_envelope, 0.96)
    print(peak_height_min)
    ref_peaks_t = sig.find_peaks(ref_signal_filtered, height=peak_height_min)[0]
    if skip!=0:
        ref_peaks_t = ref_peaks_t[ref_peaks_t >= int(round(skip/dt,0))]
    # plt.plot(ref_signal_filtered)
    # plt.plot(ref_peaks_t, ref_signal_filtered[ref_peaks_t], color="orange", ls=None, marker="x")
    # plt.show()
    return ref_peaks_t, peak_f

def get_spike_phase_distributions(spikes, left_peak_series, right_peak_series):
    pop_phases_distribs = []
    rayleigh_pvalues = np.empty(len(spikes))
    for i, neuron_spike_times in enumerate(spikes): #spikes are in time steps (not seconds), as are the peak series.
        phases = np.empty(0)
        for (left_peak, right_peak) in zip (left_peak_series, right_peak_series):
            mask = (neuron_spike_times>=left_peak) & (neuron_spike_times<right_peak)
            spikes_between_peaks = np.array(neuron_spike_times)[mask]
            phases_for_those_spikes = 360 * (spikes_between_peaks - left_peak) / (right_peak - left_peak)
            phases = np.append(phases, phases_for_those_spikes.copy())
        pop_phases_distribs.append(phases.copy())
        rayleigh_pvalues[i] = float(rayleightest(np.deg2rad(phases))) if len(phases)>0 else 1
    return pop_phases_distribs, rayleigh_pvalues


def plot_relative_phase_angles_from_spikes(spikes_dict, pops_sublist, ref_pop, dt, t_sim, skip=0, normalize=True, ax=None):
    if ax is None:
        ax = plt.gca()

    ref_peaks_t, peak_f = get_ref_signal_peaks(spikes_dict, ref_pop, dt, t_sim, skip)
    left_peak_series = ref_peaks_t[:-1]
    right_peak_series = ref_peaks_t[1:]
    
    phases_dict = {}
    rayleigh_pvalues_dict = {}
    mean_firing_rates = {}
    for pop in pops_sublist:
        phases_dict[pop], rayleigh_pvalues_dict[pop] = get_spike_phase_distributions(spikes_dict[pop], left_peak_series, right_peak_series)
        mean_firing_rates[pop] = np.mean([len(neuron_spikes) for neuron_spikes in spikes_dict[pop]]) / t_sim
        if pop=="STN":
            print("number zeroes:", len(np.where(np.array([j for i in phases_dict[pop] for j in i])<1)[0]))

    colors = json.load(open("params/graphical_params.json"))["nuclei"]
    bin_size = 2    
    n_bins = int(round(360/bin_size,0))
    width = (2*np.pi) / n_bins
    theta = np.linspace(0, 2*np.pi, n_bins, endpoint=False)

    # width = (2*np.pi) / int(round(360/bin_size,0))
    # theta = np.linspace(0, 2*np.pi, int(round(360/bin_size,0)), endpoint=False)

    for i, (pop, phase_values) in enumerate(phases_dict.items()):
        
        idx_rayleigh_pvalues_pass = np.where(rayleigh_pvalues_dict[pop] < 0.05)[0]
        print(f"{pop}: {len(idx_rayleigh_pvalues_pass)} of {len(phase_values)} passed Rayleigh test")
        phase_values_rayleigh_pass  = [phase_values[i] for i in idx_rayleigh_pvalues_pass]

        # binned_phases = [np.histogram(x, bins=int(round(360/bin_size,0)), range=(0,360))[0] for x in phase_values_rayleigh_pass]
        # print(binned_phases)
        # max_phase_values = np.array([np.argmax(np.histogram(x, bins=int(round(360/bin_size,0)), range=(0,360))[0]) for x in phase_values_rayleigh_pass])

        # print(max_phase_values)
        bar_data = np.sum([np.histogram(x, bins=n_bins, range=(0,360))[0] for x in phase_values_rayleigh_pass], axis=0)
        # bar_data = np.histogram(max_phase_values*bin_size, bins=int(round(360/bin_size,0)), range=(0,360))[0]

        if normalize:
            # ax.bar(theta, bar_data/np.max(bar_data), width=width, color=colors[pop], label=pop)
            print(mean_firing_rates[pop])
            ax.bar(theta, bar_data/mean_firing_rates[pop], width=width, color=colors[pop], label=pop)
        else:
            ax.bar(theta, bar_data, width=width, color=colors[pop], label=pop)

        
        mean_phase_values = np.array([np.rad2deg(stat.circmean(np.deg2rad([x]), high = np.pi, low = -np.pi)) for x in phase_values_rayleigh_pass])
        mean_angle = np.round(np.rad2deg(stat.circmean(np.deg2rad(mean_phase_values), high = np.pi, low = -np.pi)), 1)
        std_angle = np.round(np.rad2deg(stat.circstd(np.deg2rad(mean_phase_values), high = np.pi, low = -np.pi)), 1)
        print(f"{pop}: {mean_angle}° ± {std_angle}°")
        ax.text(1.08, 0.6 - 0.07*i, f"{mean_angle}° ± {std_angle}°", color=colors[pop],
                transform=ax.transAxes, ha='left', va='top')
    ax.set_yticks([])
    ax.set_thetagrids(np.arange(0, 360, 45), labels=['0°', '45°', '90°', '135°', '±180°', '-135°', '-90°', '-45°'])
    ax.legend(loc="upper left", bbox_to_anchor=(.8 + np.cos(1.15)/2, .6 + np.sin(1.15)/2))
    ax.set_title(f"{ref_pop} peak f={int(peak_f)} Hz")

def plot_relative_phase_angle_simplified(rates, pops_sublist, ref_pop, dt, window_size=1, skip=0, ax=None):
    """Plot phase angle diagram based on mean population rates: single angle value (arrow) instead of distribution"""
    if ax is None:
        ax = plt.gca()

    if np.ndim(rates[pops_sublist[0]])!=1:
        rates_sublist = {k: np.mean(rates[k], axis=0) for k in pops_sublist}
    else:
        rates_sublist = {k: rates[k] for k in pops_sublist}

    colors = json.load(open("params/graphical_params.json"))["nuclei"]
    # width = (2*np.pi) / 360
    # theta = np.linspace(np.pi, -np.pi, int(round(360/bin_size,0)), endpoint=False)[::-1]

    ref_data = rates_sublist[ref_pop][int(skip/dt):]

    if is_stable_oscillation(ref_data)[0]:
        ref_freq = get_peak_f_of_signal(ref_data, dt, window_size)
        ax.set_title(f"f={ref_freq} Hz", x=-0.1, style='italic', weight=10)
        ax.arrow(0, 0, 0, 1, facecolor = colors[ref_pop], edgecolor=colors[ref_pop], width = 0.025, zorder=10)
    
        for pop, data in rates_sublist.items():
            if pop==ref_pop:
                continue
            print(pop)
            test_data = data[int(skip/dt):]
            if is_stable_oscillation(test_data)[0]:
                angle = get_mean_phase_angle(test_data, ref_data, ref_freq, dt, window_size)
                ax.arrow(np.radians(angle), 0, 0, 1, facecolor = colors[pop], edgecolor=colors[pop], width = 0.03, zorder=10)                 
                print(pop, angle, "°")
    ax.grid(axis="y")
    ax.set_yticks([])
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax.tick_params(axis='x', labelsize=9, pad=1)


def get_mean_phase_angle(signal, ref_signal, freq, dt, window_size=1):
    f, Pxy = sig.csd(signal, ref_signal, fs=1/dt, nperseg=window_size/dt)
    freq_idx = np.argmin(np.abs(f-freq))
    angle = np.angle(Pxy[freq_idx], deg=True)

    return angle


def plot_spectrogram(rate, dt, skip=0, duration=10, t_stim=None, window_size=0.5, nperseg=4096):
    if t_stim == None:
        signal = np.mean(rate, axis=0)[int(skip/dt):int((skip+duration)/dt)]
        t_offset = 0
        xlab = 'Time (s)'
    else:
        t_offset = duration/2.2
        signal = np.mean(rate, axis=0)[int((t_stim-t_offset)/dt):int((t_stim+duration-t_offset)/dt)]
        xlab = 'Time from cortical stim (s)'
    f, t, Sxx = sig.spectrogram(signal, fs=int(1/dt), nperseg=nperseg, noverlap = nperseg // 2)
    print(t-t_offset)
    plt.pcolormesh(t-t_offset, f, Sxx, shading='gouraud')
    plt.xlabel(xlab)
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0,70)
    plt.colorbar()


def plot_frequency_heatmap(f_list, xvalues, xlabel, yvalues, ylabel, x_eq=None, y_eq=None, title="", ax=None):

    if ax is None:
        ax = plt.gca()

    binsize_x = (xvalues[-1]- xvalues[0]) / (len(xvalues)-1)
    binsize_y = (yvalues[-1]- yvalues[0]) / (len(yvalues)-1)
    print(binsize_x, binsize_y)

    cmap = plt.get_cmap("jet", 60)

    img = ax.imshow(f_list, cmap = cmap, vmin=0, vmax=60, aspect="auto",
                    extent=[xvalues[0] - binsize_x/2, xvalues[-1] + binsize_x/2, yvalues[-1] + binsize_y/2, yvalues[0] - binsize_y/2])
    if x_eq != None:
        ax.axvline(abs(x_eq), color="black", ls="--")
    if y_eq != None:
        ax.axhline(abs(y_eq), color="black", ls="--")
    ax.invert_yaxis()

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return img

    # clb = fig.colorbar(img, shrink=0.8, ax = ax)
    # clb.set_ticks([0, 10, 20, 30, 40, 50, 60, 70])
    # clb.set_ticklabels([0, 10, 20, 30, 40, 50, 60, 70])
    # clb.set_label("frequency (Hz)", labelpad=40, y=0.5, rotation=-90)
    # #clb.ax.tick_params(labelsize=clb_tick_size, length = tick_length)
    # #set_max_dec_tick(ax, n_decimal = n_decimal)
    # clb.ax.yaxis.tick_right()


def plot_4D_frequency_heatmap(f_values, G_Proto_to_Proto_values, G_STN_to_Proto_values, G_D2_to_Proto_values, G_GPi_to_Th_values, pad=10):
    fig, ax = plt.subplots(nrows=len(G_GPi_to_Th_values), ncols=len(G_Proto_to_Proto_values), figsize=(12,12), sharex=True, sharey=True)
    # print("subplots:", np.shape(ax))
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
    fig.text(0.01, 0.5, "|G GPi-Th|", va='center', rotation='vertical', fontsize=14)

    clb = fig.colorbar(img, shrink=0.4, ax = ax[:,:])
    clb.set_ticks([0, 10, 20, 30, 40, 50, 60])
    clb.set_ticklabels([0, 10, 20, 30, 40, 50, 60])
    clb.set_label("frequency (Hz)", labelpad=20, y=0.5, rotation=-90, fontsize=14)
    #clb.ax.tick_params(labelsize=clb_tick_size, length = tick_length)
    #set_max_dec_tick(ax, n_decimal = n_decimal)
    clb.ax.yaxis.tick_right()

    return fig, ax