import numpy as np
from scipy.signal import sosfiltfilt, butter
from scipy.stats import truncnorm, norm
from scipy.optimize import fsolve, curve_fit
# from sympy import sin, cos, tan, cot
from itertools import combinations


def uniform_sample(lower_bound, upper_bound, size=1):
    return np.random.uniform(low = lower_bound, high=upper_bound, size=size)

def gaussian_trunc_sample(m, sd, lower_bound, upper_bound, size=1):
    a, b = (lower_bound - m) / sd, (upper_bound - m) / sd
    return truncnorm.rvs(a, b, loc=m, scale=sd, size=size)
    
def gaussian_sample(m, sd, size=1):
    if sd==0:
        return np.full(size, m)
    return np.random.normal(loc=m, scale=sd, size=size)
    
def lognormal_sample(m, sd=None, size=1):
    if sd==None:
        order_mag = 1
        sd = (10 ** order_mag - 1) / (10 ** order_mag + 1) * abs(m)

    sd_norm = np.sqrt(np.log(1 + (sd / m) ** 2))
    m_norm = np.log( m ** 2 / np.sqrt(m ** 2 + sd ** 2))
    norm_sample = gaussian_sample(m_norm, sd_norm, size)
    return np.sign(m) * np.exp(norm_sample)

def f_LIF(v, v_rest, tau, I_syn, I_ext, w=0): #return dv/dt
    return (-(v -v_rest) + I_ext + I_syn - w) / tau

def Runge_Kutta_2nd_order(dt, v, v_rest, tau, I_syn, I_ext, w=0):
    f_t = f_LIF(v, v_rest, tau, I_syn, I_ext, w)
    next_v = v + dt/2 * f_t + dt/2 * (-(v + dt * f_t - v_rest) + I_ext + I_syn - w) / tau
    return next_v

def linear_interpolation(x, x1, x2, y1, y2):
    y = y1 + (x-x1)*(y2-y1)/(x2-x1)
    return y

def sigmoid_FI(I, L, k, I_0):
    return L / (1 + np.exp(-k * (I - I_0)))

def sigmoid_interpolation(target_FR, I_ext_data, FR_data):
    p0 = [max(FR_data), 1.0, np.mean(I_ext_data)]
    popt, _ = curve_fit(sigmoid_FI, I_ext_data, FR_data, p0=p0, maxfev=5000)
    L, k, I_0 = popt
    return I_0 + (1/k) * np.log(target_FR / (L - target_FR))


def Hansel_linear_interpolation(dt, v, next_v, v_rest, v_th, tau):
    v_reset = (next_v - v_th) * (1 + dt/tau * (v - v_rest)/(next_v - v)) + v_rest
    return v_reset

def no_noise(amplitude, std, n, dt, sqrt_dt, tau = 0, noise_dt_before = 0):
    return np.zeros_like(noise_dt_before)

def fwd_Euler(dt, y, f):
    return y + dt * f

def Gaussian_noise(amplitude, std, n, dt, sqrt_dt, tau = 0, noise_dt_before = 0):
    noise = 1/sqrt_dt * amplitude * gaussian_sample(0, std, n)
    return noise

def Ornstein_Uhlenbeck_noise(amplitude, std, n, dt, sqrt_dt, tau=0.01, noise_dt_before = 0):
    noise = noise_dt_before*np.exp(-dt/tau) + std*np.sqrt(1 - np.exp(-2*dt/tau)) * gaussian_sample(0, 1, n)
    return noise

def _Ornstein_Uhlenbeck_noise(amplitude, std, n, dt, sqrt_dt, tau= 0.01,  noise_dt_before = 0):
    ''' Ornstein-Uhlenbeck process as time correlated noise generator'''
    noise_prime = -noise_dt_before /tau + std * np.sqrt(2/tau) * Gaussian_noise(amplitude, 1, n, dt, sqrt_dt)
    noise = fwd_Euler(dt, noise_dt_before, noise_prime)
    #return np.zeros_like(noise)
    return noise

def bandpass_filter(signal, fs, low, high, order=6):
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def solve_beta_1_nuclei(x, tau, delay):
    f = x[0]
    return f*tau + np.tan(f*delay)

def solve_beta_2_nuclei(x, taus_deg1, taus_deg2, sum_delays):
    f = x[0]
    return -f*(taus_deg1) * 1/np.tan(f*(sum_delays)) + (taus_deg2)*f**2 - 1

def solve_beta_3_nuclei(x, taus_deg1, taus_deg2, taus_deg3, sum_delays):
    f = x[0]
    return (taus_deg3 * f**3) + (taus_deg2 * f**2 -1) * np.tan(f*sum_delays) - (f*taus_deg1)

def solve_beta_4_nuclei(x, taus_deg1, taus_deg2, taus_deg3, taus_deg4, sum_delays):
    f = x[0]
    return f*taus_deg1 - f**3 * taus_deg3 + (1 - f**2 * taus_deg2 + f**4 * taus_deg4) * np.tan(f*sum_delays)

def solve_beta_4_nuclei_bifurcated(x, taus_deg1, taus_deg2, taus_deg3, taus_deg4, taus_deg5, sum_side_taus, prod_side_taus, sum_delays):
    f = x[0]
    return (f*taus_deg1 - f**3*taus_deg3 + f**5*taus_deg5 + 
            ( (-1 + f**2*taus_deg2 - f**4*taus_deg4)*np.sin(f*sum_delays)*(1 - f**2*prod_side_taus) - np.cos(f*sum_delays)*f*sum_side_taus ) / 
            (np.cos(f*sum_delays)*(f**2*prod_side_taus - 1) - np.sin(f*sum_delays)*f*sum_side_taus))
    

def theory_1_nuclei(x, list_taus, sum_delays):
    tau = list_taus[0]
    root = fsolve(solve_beta_1_nuclei, x, args=(tau, sum_delays))
    G = 1/(np.cos(root[0]*sum_delays))
    return root[0], float(G)

def theory_2_nuclei(x, list_taus, sum_delays):
    taus_deg1 = np.sum(list_taus)
    taus_deg2 = np.prod(list_taus)
    root = fsolve(solve_beta_2_nuclei, x, args=(taus_deg1, taus_deg2, sum_delays))
    G = (-root[0]*taus_deg1)/np.sin(root[0]*sum_delays)
    return root[0], float(G)

def theory_3_nuclei(x, list_taus, sum_delays):
    taus_deg1 = np.sum(list_taus)
    taus_deg2 = np.sum([a*b for a,b in combinations(list_taus,2)])
    taus_deg3 = np.prod(list_taus)
    root = fsolve(solve_beta_3_nuclei, x, args=(taus_deg1, taus_deg2, taus_deg3, sum_delays))
    G = (1 - root[0]**2 * taus_deg2)/np.cos(root[0] * sum_delays)
    return root[0], float(G)


def theory_4_nuclei(x, list_taus, sum_delays):
    taus_deg1 = np.sum(list_taus)
    taus_deg2 = np.sum([a*b for a,b in combinations(list_taus,2)])
    taus_deg3 = np.sum([a*b*c for a,b,c in combinations(list_taus,3)])
    taus_deg4 = np.prod(list_taus)
    root = fsolve(solve_beta_4_nuclei, x, args=(taus_deg1, taus_deg2, taus_deg3, taus_deg4, sum_delays))
    G = (1 - root[0]**2*(taus_deg2) + root[0]**4 * taus_deg4)/(np.cos(root[0]*sum_delays))
    return root[0], float(G)


def theory_4_nuclei_bifurcated(x, list_taus, sum_delays, list_side_taus):
    taus_deg1 = np.sum(list_taus)
    taus_deg2 = np.sum([a*b for a,b in combinations(list_taus,2)])
    taus_deg3 = np.sum([a*b*c for a,b,c in combinations(list_taus,3)])
    taus_deg4 = np.sum([a*b*c*d for a,b,c,d in combinations(list_taus,4)])
    taus_deg5 = np.prod(list_taus)
    sum_side_taus = np.sum(list_side_taus)
    prod_side_taus = np.prod(list_side_taus)
    root = fsolve(solve_beta_4_nuclei_bifurcated, x, args=(taus_deg1, taus_deg2, taus_deg3, taus_deg4, taus_deg5, sum_side_taus, prod_side_taus, sum_delays))
    G = (-1 + root[0]**2*taus_deg2 - root[0]**4*taus_deg4) / (np.cos(root[0]*sum_delays)*(root[0]*prod_side_taus - 1) - np.sin(root[0]*sum_delays)*root[0]*sum_side_taus)
    return root[0], float(G)