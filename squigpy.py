import math
from scipy.integrate import solve_ivp
import numpy as np

# --- Constant pi_p ---
def pi_p(p):
    return 2 * math.gamma(1/p)**2 / (p*math.gamma(2/p))
# Vectorize the equation
pi_p = np.vectorize(pi_p)

# --- Vectorfield required to solve Diff. Eqs.
def vectorfield(t, f, p):
    return [-f[1]**(p-1), f[0]**(p-1)]


# --- Vectorized Squine and Cosquine function ---
# All p > 1 are valid
def squine(t, p):
    # Check for array
    if type(t) != np.ndarray:
        t = np.array(t)
    # Set up
    ppi = pi_p(p)
    s = np.zeros(t.size)
    # -- 0 < t < pi_p/2
    mask = (t < ppi/2)
    if np.count_nonzero(mask) != 0:
        sinp = solve_ivp(vectorfield, (0,ppi/2+0.06), [1, 0], 
                        t_eval=t[mask], method='DOP853', args={p})
        s[mask] = sinp.y[1]
    # -- pi_p/2 < t < pi_p
    mask = (t >= ppi/2) & (t < ppi)
    if np.count_nonzero(mask) != 0:
        sinp = sinp = solve_ivp(vectorfield, (0,ppi/2+0.06), [1, 0], 
                        t_eval=np.flip(ppi/2-(t[mask]-ppi/2)), 
                        method='DOP853', args={p})
        s[mask] = np.flip(sinp.y[1])
    # -- pi_p < t < 3/2*pi_p
    mask = (t >= ppi) & (t < ppi*3/2)
    if np.count_nonzero(mask) != 0:
        sinp = sinp = solve_ivp(vectorfield, (0,ppi/2+0.06), [1, 0], 
                        t_eval=((t[mask]-ppi)), method='DOP853', args={p})
        s[mask] = -sinp.y[1]
    # -- pi_p*3/2 < t < 2*pi_p
    mask = (t >= ppi*3/2) & (t < 2*ppi)
    if np.count_nonzero(mask) != 0:
        sinp = sinp = solve_ivp(vectorfield, (0,ppi/2+0.06), [1, 0], 
                        t_eval=np.flip(ppi/2-(t[mask]-ppi*3/2)), 
                        method='DOP853', args={p})
        s[mask] = -np.flip(sinp.y[1])
    return s


def cosquine(t, p):
    # Check for array
    if type(t) != np.ndarray:
        t = np.array(t)
    # Set up
    ppi = pi_p(p)
    c = np.ones(t.size)
    # -- 0 < t < pi_p/2
    mask = (t < ppi/2)
    if np.count_nonzero(mask) != 0:
        sinp = solve_ivp(vectorfield, (0,ppi/2+0.06), [1, 0], 
                        t_eval=np.flip(ppi/2-t[mask]), method='DOP853', args={p})
        c[mask] = np.flip(sinp.y[1])
    # -- pi_p/2 < t < pi_p
    mask = (t >= ppi/2) & (t < ppi)
    if np.count_nonzero(mask) != 0:
        sinp = sinp = solve_ivp(vectorfield, (0,ppi/2+0.06), [1, 0], 
                        t_eval=(t[mask]-ppi/2), method='DOP853', args={p})
        c[mask] = -(sinp.y[1])
    # -- pi_p < t < 3/2*pi_p
    mask = (t >= ppi) & (t < ppi*3/2)
    if np.count_nonzero(mask) != 0:
        sinp = sinp = solve_ivp(vectorfield, (0,ppi/2+0.06), [1, 0], 
                        t_eval=np.flip(ppi/2-(t[mask]-ppi)), 
                        method='DOP853', args={p})
        c[mask] = -np.flip(sinp.y[1])
    # -- pi_p*3/2 < t < 2*pi_p
    mask = (t >= ppi*3/2) & (t < 2*ppi)
    if np.count_nonzero(mask) != 0:
        sinp = sinp = solve_ivp(vectorfield, (0,ppi/2+0.06), [1, 0], 
                        t_eval=(t[mask]-ppi*3/2), method='DOP853', args={p})
        c[mask] = (sinp.y[1])
    return c


def tanquent(t, p):
    return squine(t, p) / cosquine(t,p)