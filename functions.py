#!/usr/bin/env python
# coding: utf-8

# internal workings of the code!

# this code runs internally with eV so these might be useful
meter = 5.06773093741e6        # [eV^-1/m]
km    = 1.0e3*meter            # [eV^-1/km]
MeV   = 1.0e6                  # [eV/MeV]

# === Standard library imports ===@===@===@===@===@===@===

import colorsys
import math
import os
import time
import warnings


# === Third-party imports ===@===@===@===@===@===@===

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
from IPython.display import clear_output, display, HTML
from odeintw import odeintw
from scipy.integrate import quad, quad_vec
from scipy.linalg import expm

#############################################################################

# === Array/matrix utilities ===@===@===@===@===@===@===

# matrix/array operations
sh = lambda x: print(np.shape(x))
tr = lambda x: print(np.trace(x))

# take diagonal of each block
def dibloc(x): return np.real(np.array([np.diag(block) for block in x]))

# we are gonna need these
def sq(x): return x * x
def cube(x): return x * x * x
def ht(x): return np.heaviside(x, 0)

# conjugate transpose (works for higher dimensional arrays too)
def dagger(x):
    if np.ndim(x) == 2:
        return np.conj(x).T
    if np.ndim(x) == 3:
        return np.transpose(np.conj(x), axes=(0, 2, 1))
    if np.ndim(x) == 4:
        return np.transpose(np.conj(x), axes=(0, 1, 3, 2))
    else:
        print("TOO MANY DIMENSIONS!")

# calc bin centres
def calc_bin_centres(bin_edges):
    return 0.5 * (bin_edges[1:] + bin_edges[:-1])


# === Mixing ===@===@===@===@===@===@===

# generate PMNS with custom mixing parameters + CP phase
def Uall(theta12, theta23, theta13, deltaCP):

    d_ = np.exp(-1j * deltaCP)
    d  = np.exp( 1j * deltaCP)

    s12, c12 = np.sin(theta12), np.cos(theta12)
    s23, c23 = np.sin(theta23), np.cos(theta23)
    s13, c13 = np.sin(theta13), np.cos(theta13)

    U = np.linalg.multi_dot(([[1, 0,    0   ],
                               [0, c23,  s23 ],
                               [0, -s23, c23 ]],
                                                [[c13,      0, s13 * d_],
                                                 [0,        1, 0       ],
                                                 [-s13 * d, 0, c13    ]],
                              [[c12,  s12, 0],
                               [-s12, c12, 0],
                               [0,    0,   1]]))
    return U

# rho_m =  U_dagger * rho_m * U (for neutrinos i_nu == 0)
def flav_to_mass(rho, U, i_nu):

    if i_nu == 0:
        return np.linalg.multi_dot((dagger(U), rho, U))
    else:
        return np.linalg.multi_dot((U, rho, dagger(U)))

# rho_f = U * rho_m * U_dagger (for neutrinos i_nu == 0)
def mass_to_flav(rho, U, i_nu):

    if i_nu == 0:
        return np.linalg.multi_dot((U, rho, dagger(U)))
    else:
        return np.linalg.multi_dot((dagger(U), rho, U))
    

# === Neutrino decay formulas ===@===@===@===@===@===@===

# useful decay functions
def f(x): return 0.5 * x + 2 + 2 * np.log(x) / x - 2 / sq(x) - 0.5 / cube(x)
def h(x): return 0.5 * x - 2 + 2 * np.log(x) / x + 2 / sq(x) - 0.5 / cube(x)
def k(x): return 0.5 * x - 2 * np.log(x) / x - 0.5 / cube(x)

# decay width for helicity Conserving decay
def gam_c(mi, mj, Ei, g_s, g_ps):
    x = mi / mj
    g = (mi * mj / (16 * np.pi * Ei)) * (sq(g_s) * f(x) + sq(g_ps) * h(x))
    return g * ht(x - 1)

# decay width for helicity Violating decay
def gam_v(mi, mj, Ei, g_s, g_ps):
    x = mi / mj
    g = (mi * mj / (16 * np.pi * Ei)) * (sq(g_s) + sq(g_ps)) * k(x)
    return g * ht(x - 1)

# differential partial width for helicity Conserving decay
def wgam_c(mi, mj, Ei, Ej, g_s, g_ps):
    x = mi / mj
    A = (1 / x) * (Ei / Ej) + x * (Ej / Ei)
    Wg = (mi * mj / (16 * np.pi * sq(Ei))) * (sq(g_s) * (A + 2) + sq(g_ps) * (A - 2))  
    return Wg * ht(x - 1) * ht(sq(x) * Ej - Ei) * ht(Ei - Ej)

# differential partial width for helicity Violating decay
def wgam_v(mi, mj, Ei, Ej, g_s, g_ps):
    x = mi / mj
    A = (1 / x) * (Ei / Ej) + x * (Ej / Ei)
    Wg = (mi * mj / (16 * np.pi * sq(Ei))) * (sq(g_s) + sq(g_ps)) * (1 / x + x - A) 
    return Wg * ht(x - 1) * ht(sq(x) * Ej - Ei) * ht(Ei - Ej)


# === Open Quantum System Machinery ===@===@===@===@===@===@===

# Dynamical map -> Choi
def s2c(matrix):
    M = matrix.shape[0]
    L = int(np.sqrt(M))
    return np.einsum('abcd->dbca', matrix.reshape(L,L,L,L)).reshape(M,M)

# Choi -> Kraus
def c2k(matrix):
    M = matrix.shape[0]
    L = int(np.sqrt(M))
    
    # real eigenvalues, sorted ascending, orthonormal evecs
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # ignore tiny eigenvalues
    tol = eigenvalues[-1] * 1e-10
    mask = eigenvalues > tol
    eigenvalues = eigenvalues[mask]
    eigenvectors = eigenvectors[:, mask] 
    
    # build all Kraus operators at once
    kraus = (np.sqrt(eigenvalues) * eigenvectors).T.reshape(-1, L, L)
    kraus = np.swapaxes(kraus, 1, 2)
    
    return list(kraus)

# create Lindblad operators
def lindblad_operators(e_edges, masses, g_s, g_ps, channel):
    
    for name in channel:
        if not (name.count('->') == 1):
            raise ValueError(f"Channel key '{name}' must contain exactly one '->'.")
        parent, daughter = name.split('->')
        if not (parent[0] in ('n', 'a') and daughter[0] in ('n', 'a')):
            raise ValueError(f"Channel key '{name}': parent and daughter must start with 'n' or 'a'.")

    e_centr = calc_bin_centres(e_edges)
    n_bins  = len(e_centr)
    Ndim    = len(masses)           
    
    par_indx = sorted({v["index_p"] for v in channel.values()})
    par_dict = {p: slot for slot, p in enumerate(par_indx)}

    A = np.zeros((len(par_indx), n_bins, n_bins, Ndim, Ndim), dtype=np.complex128)
    
    for c, (chan, data) in enumerate(channel.items()):
        for ini in range(n_bins):
            for fin in range(n_bins):
                
                i, j   = data["index_p"], data["index_d"]
                mi, mj = masses[i], masses[j]
                x_     = mi / mj
                E_lo   = max(e_edges[fin],     e_centr[ini] / sq(x_))
                E_hi   = min(e_edges[fin + 1], e_centr[ini])
                
                parent, daughter = chan.split('->')
                hv   = parent[0] != daughter[0]     # check if decay is helicity conserving/violating
                wgam = wgam_v if hv else wgam_c
                ans    = quad_vec(lambda Ej: wgam(mi, mj, e_centr[ini], Ej, g_s, g_ps), E_lo, E_hi)[0]
                A[par_dict[i], ini, fin, j, i] = np.sqrt(ans)
                
                                                     # L        (Npar, NE, NE, Nnu, Nnu)
    Adag     = [dagger(A_) for A_ in A]              # L^†      (NE, NE, Nnu, Nnu)                    
    sumAdagA = np.matmul(Adag, A).sum(axis=(0, 2))   # Sum L^†L (NE, Nnu, Nnu)

    return A, Adag, sumAdagA

# the RHS of the master equation ODE - solved at every step, L
def master_eqn(p_flat, L, H, A, Adag, sumAdagA):

    n_bins = len(H)
    Ndim   = H.shape[-1]

    p  = p_flat.reshape(n_bins, Ndim, Ndim)
    dp = np.zeros((n_bins, Ndim, Ndim), dtype=np.complex128)
     
    dp += -1j * (np.matmul(H, p) - np.matmul(p, H))       
    dp -= 0.5 * (sumAdagA @ p + p @ sumAdagA)                
    dp += (A @ (p[None, :, None] @ Adag)).sum(axis=(0, 1))

    return dp.ravel()

# the unravelled master equation - returns the dynamical map
def unravelled_master_eqn(L, H, A, Adag, sumAdagA):

    n_bins = len(H)
    Ndim   = H.shape[-1]
    Npar   = A.shape[0]

    L_super = np.zeros((sq(Ndim) * n_bins, sq(Ndim) * n_bins), dtype=np.complex128)

    I = np.eye(Ndim)

    for n in range(n_bins):
        block  = -1j * (np.kron(np.eye(Ndim), H[n]) - np.kron(H[n].T, I))
        block -= 0.5 * (np.kron(I, sumAdagA[n]) + np.kron(sumAdagA[n].T, I))
        L_super[
            sq(Ndim) * n : sq(Ndim) * (n + 1),
            sq(Ndim) * n : sq(Ndim) * (n + 1),
        ] += block

    for ini in range(n_bins):
        for fin in range(n_bins):
            L_super[
                    sq(Ndim) * fin : sq(Ndim) * (fin + 1),
                    sq(Ndim) * ini : sq(Ndim) * (ini + 1)
                   ]+= sum(np.kron(A[p, ini, fin].conj(), A[p, ini, fin]) for p in range(Npar))

    return expm(L_super * L)



# === User Functions ===@===@===@===@===@===@===

def make_majorana(masses, channel):
    """
    Extend a Dirac decay channel dictionary to a Majorana one by doubling the state space.

    The Dirac channel dictionary has keys of the form "nX->nY" (e.g. "n2->n1"), where
    "n"("a") denotes a neutrino(antineutrino) mass eigenstate. This function produces 
    all four Majorana channels for each Dirac channel:

        nX->nY   (nu -> nu,   helicity-conserving)
        aX->aY   (anu -> anu, helicity-conserving)
        nX->aY   (nu -> anu,  helicity-violating)
        aX->nY   (anu -> nu,  helicity-violating)

    The doubled mass array is [m1, m2, ..., m1, m2, ...], where the first half
    corresponds to neutrinos and the second half to antineutrinos.

    Parameters
    ----------
    masses (Ndim,)                     : Neutrino masses (lightest first).
    channel                            : Decay-channel dictionary.  Each entry is a channel with
                                         indices for both parent and daughter.

    Returns
    -------
    masses_maj (2*Ndim,)               : Doubled Majorana neutrino masses.
    channel_maj                        : Doubled Majorana decay-channel dictionary.  Each entry is 
                                         a channel with indices for both parent and daughter.
    """

    for name in channel:
        if not (name.count('->') == 1):
            raise ValueError(f"Channel key '{name}' must contain exactly one '->'.")
        parent, daughter = name.split('->')
        if not (parent.startswith('n') and daughter.startswith('n')):
            raise ValueError(f"Channel key '{name}' must be of the form 'nX->nY'.")

    Ndim = len(masses)

    masses_maj  = list(masses) + list(masses)
    channel_maj = {}
    for name, data in channel.items():
        i, j = data["index_p"], data["index_d"]
        parent, daughter = name.split('->')
        a_parent   = parent.replace('n', 'a')
        a_daughter = daughter.replace('n', 'a')
        channel_maj[name]                        = {**data}                                
        channel_maj[f"{a_parent}->{a_daughter}"] = {"index_p": Ndim+i, "index_d": Ndim+j}  
        channel_maj[f"{parent}->{a_daughter}"]   = {"index_p": i,      "index_d": Ndim+j}  
        channel_maj[f"{a_parent}->{daughter}"]   = {"index_p": Ndim+i, "index_d": j}     
        
    return masses_maj, channel_maj


def lind(initial_value, L, e_edges, masses, g_s, g_ps, channel):
    """
    Solve the Lindblad master equation by directly solving ODE.

    Parameters
    ----------
    initial_value (n_bins, Ndim, Ndim) : Initial density matrix in the mass basis, for each energy bin.   
    L                                  : Propagation distance.   
    e_edges (n_bins + 1,)              : Energy bin edges.
    masses (Ndim,)                     : Neutrino masses (lightest first, doubled for Majorana).
    g_s(g_ps)                          : Scalar(pseudoscalar) coupling.
    channel                            : Decay-channel dictionary.  Each entry is a channel with
                                         indices for both parent and daughter.
    Returns
    -------
    ndarray (n_bins, Ndim, Ndim)       : Evolved density matrix at L, one block per energy bin.
    """
    
    e_centr = calc_bin_centres(e_edges)
    n_bins  = len(e_centr)
    Ndim    = len(masses)                

    # Hamiltonian
    H = np.zeros((n_bins, Ndim, Ndim), dtype=np.complex128) 
    for k, m in enumerate(masses):
        H[:, k, k] = (sq(m) - sq(masses[0])) / (2 * e_centr)

    # Lindblad operators
    A, Adag, sumAdagA = lindblad_operators(e_edges, masses, g_s, g_ps, channel)

    # solve the ODE - we have chosen 100 steps, and these tolerances
    solution = odeintw(master_eqn, initial_value, np.linspace(0, L, 100),
                       args=(H, A, Adag, sumAdagA),
                       rtol=1e-8, atol=1e-8, mxstep=50_000)

    return solution[-1].reshape(n_bins, Ndim, Ndim)


def dynam(initial_value, L, e_edges, masses,  g_s, g_ps, channel):
    """
    Solve the Lindblad equation via the dynamical map (this is the default method).

    Applies the dynamical map (obtained from the matrix exponential of the
    Liouvillian) directly to the vectorized initial state. 

    Parameters
    ----------
    initial_value (n_bins, Ndim, Ndim) : Initial density matrix in the mass basis, for each energy bin.   
    L                                  : Propagation distance.   
    e_edges (n_bins + 1,)              : Energy bin edges.
    masses (Ndim,)                     : Neutrino masses (lightest first, doubled for Majorana).
    g_s(g_ps)                          : Scalar(pseudoscalar) coupling.
    channel                            : Decay-channel dictionary.  Each entry is a channel with
                                         indices for both parent and daughter.
    Returns
    -------
    ndarray (n_bins, Ndim, Ndim)       : Evolved density matrix at L, one block per energy bin.
    """
    
    e_centr = calc_bin_centres(e_edges)
    n_bins  = len(e_centr)
    Ndim    = len(masses)

    # Hamiltonian
    H = np.zeros((n_bins, Ndim, Ndim), dtype=np.complex128)
    for k, m in enumerate(masses):
        H[:, k, k] = (sq(m) - sq(masses[0])) / (2 * e_centr)

    # Lindblad operators
    A, Adag, sumAdagA = lindblad_operators(e_edges, masses, g_s, g_ps, channel)

    # unravel the initial density matrix
    vec_rho = initial_value.reshape(-1)

    # get the dynamical map
    dy_map = unravelled_master_eqn(L, H, A, Adag, sumAdagA)

    # solve the system
    solution = dy_map @ vec_rho

    return solution.reshape(n_bins, Ndim, Ndim)


def kraus(initial_value, L, e_edges, masses, g_s, g_ps, channel):
    """
    Solve the Lindblad equation via the Kraus operator decomposition.

    Converts the dynamical map (obtained from the matrix exponential of the
    Liouvillian) into Kraus operators via the Choi-Jamiolkowski isomorphism
    and applies them block-by-block.  Equivalent to 'dynam' but uses the 
    Kraus representation explicitly, which is useful for verification or
    downstream quantum-information analysis.
    
    Parameters
    ----------
    initial_value (n_bins, Ndim, Ndim) : Initial density matrix in the mass basis, for each energy bin.   
    L                                  : Propagation distance.   
    e_edges (n_bins + 1,)              : Energy bin edges.
    masses (Ndim,)                     : Neutrino masses (lightest first, doubled for Majorana).
    g_s(g_ps)                          : Scalar(pseudoscalar) coupling.
    channel                            : Decay-channel dictionary.  Each entry is a channel with
                                         indices for both parent and daughter.
    Returns
    -------
    ndarray (n_bins, Ndim, Ndim)       : Evolved density matrix at L, one block per energy bin.
    """
    e_centr = calc_bin_centres(e_edges)
    n_bins  = len(e_centr)

    Ndim = len(masses)

    # Hamiltonian
    H = np.zeros((n_bins, Ndim, Ndim), dtype=np.complex128)
    for k, m in enumerate(masses):
        H[:, k, k] = (sq(m) - sq(masses[0])) / (2 * e_centr)

    # Lindblad operators
    A, Adag, sumAdagA = lindblad_operators(e_edges, masses, g_s, g_ps, channel)

    # get the dynamical map
    dy_map = unravelled_master_eqn(L, H, A, Adag, sumAdagA)

    # solve the system via Kraus operators
    solution = np.zeros((n_bins, Ndim, Ndim), dtype=np.complex128)
    for ini in range(n_bins):
        for fin in range(n_bins):
            block = dy_map[
                sq(Ndim) * fin : sq(Ndim) * (fin + 1),
                sq(Ndim) * ini : sq(Ndim) * (ini + 1),
            ]
            for M in c2k(s2c(block)):
                solution[fin] += M @ initial_value[ini] @ dagger(M)

    return solution