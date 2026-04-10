#!/usr/bin/env python
# coding: utf-8

# these are the plots from the paper!

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

decay_purple = (0.3, 0, 0.8)

plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['mathtext.fontset'] = 'cm'

plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

plt.rcParams.update({
    "text.usetex": True,                            
    "font.family": "serif",                         
    "font.serif": ["Computer Modern Roman"],        
    "font.size": 15,                                
    "axes.labelsize": 17,                           
    "axes.titlesize": 17,                           
    "legend.fontsize": 8,                           
    "xtick.labelsize": 11,                          
    "ytick.labelsize": 11,                          
    "legend.title_fontsize": 8,                     
    "text.latex.preamble": [r"\usepackage{amsmath}", r"\usepackage{upgreek}", r'\usepackage{amssymb}'],
    "figure.dpi": 300,
})

plt.show()

class Roll:
    def __init__(self, label="", end=0):
        self._label = label
        self._end   = end
        self._t0    = None
        self._out   = None

    def __enter__(self):
        self._t0  = time.time()
        self._out = widgets.Output()
        display(self._out)
        with self._out:
            display(HTML(f'<p style="font-size:2.5em">⏱ '
                         f'<span style="font-size:0.5em; font-family:monospace">'
                         f'{self._label}, Loading...</span></p>'))
        return self

    def __exit__(self, *_):
        elapsed = time.time() - self._t0
        face    = "⚀⚁⚂⚃⚄⚅"[self._end]
        with self._out:
            clear_output(wait=True)
            display(HTML(f'<p style="font-size:2.5em">{face} '
                         f'<span style="font-size:0.5em; font-family:monospace">'
                         f'{self._label}, Done in {elapsed:.1f}s</span></p>'))
def add_matrix_inlay(ax, e_edges, matrix, N, colors, nu_labels, lim=1.5):
    
    inset = ax.inset_axes([0.53, 0.23, 0.38, 0.3])
    stacked_stairs(inset, e_edges, matrix[:, :3], 3, colors, nu_labels)

    height = [0.4, 0.8, 1.2]
    cumulative = np.zeros(len(e_edges) - 1)
    for flav in range(3):
        y = matrix[:, flav]
        inset.text(
            e_edges[-1] / MeV * 0.875, height[flav],
            nu_labels[flav],
            color=colors[flav], fontsize=12,
            va='center', ha='left', fontweight='bold'
        )
        cumulative += y

    inset.set_xlim(e_edges[0]/MeV, e_edges[-1]/MeV)
    inset.set_ylim(0, lim)

    inset.set_xticks([])
    inset.yaxis.tick_right()
    inset.set_yticks([0, lim])
    inset.set_yticklabels(["0", str(lim)])
    inset.tick_params(length=2)

def stacked_stairs(ax, e_edges, matrix, N, colors, nu_labels):
    cumulative = np.zeros(len(e_edges) - 1)
    for flav in range(N):
        y    = matrix[:, flav]
        bot  = cumulative.copy()
        top  = cumulative + y
        x    = e_edges / MeV
        b    = np.append(bot, bot[-1])
        t    = np.append(top, top[-1])
        ax.fill_between(x, b, t, step='post',
                        color=colors[flav], alpha=0.80,
                        label=nu_labels[flav], linewidth=0)
        ax.step(x, t, where='post',
                color=colors[flav], linewidth=0.)
        cumulative = top
        
def empty_stairs(ax, e_edges, matrix, N, nu_labels):
    cumulative = np.zeros(len(e_edges) - 1)
    for flav in range(N):
        y = matrix[:, flav]
        if np.allclose(y, 0):
            continue
        bot = cumulative.copy()
        top = cumulative + y
        x   = e_edges / MeV
        t   = np.append(top, top[-1])
        ax.step(x, t, where='post', color="k", linewidth=1, ls="dotted")
        cumulative = top


def time_method(func, *args, **kwargs):
    t0  = time.perf_counter()
    out = func(*args, **kwargs)
    return time.perf_counter() - t0, out


def darken(rgb, factor=0.7):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, l * factor, s)

def stealth_arrow(ax, x0, y0, x1, y1, rad=0.0, color='black', lw=0.5, size=0.18, t_head=0.5):
    
    if abs(rad) < 1e-9:
        dx, dy = x1-x0, y1-y0
        length = np.hypot(dx, dy)
        tang   = np.array([dx/length, dy/length])
        tip    = np.array([x0 + t_head*(x1-x0), y0 + t_head*(y1-y0)])
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=2)
    else:
        mx, my = 0.5*(x0+x1), 0.5*(y0+y1)
        dx, dy = x1-x0, y1-y0
        cx = mx - rad*dy
        cy = my + rad*dx
        def qbez(t):
            return ((1-t)**2 * np.array([x0, y0])
                    + 2*(1-t)*t * np.array([cx, cy])
                    + t**2      * np.array([x1, y1]))
        tip   = qbez(t_head)
        eps   = 1e-3
        tang  = qbez(t_head + eps) - qbez(t_head - eps)
        tang /= np.linalg.norm(tang)
        ts  = np.linspace(0.0, 1.0, 300)
        pts = np.array([qbez(t) for t in ts])
        ax.plot(pts[:,0], pts[:,1], color=color, lw=lw, zorder=2)
        tip = tip + 0.5*size * tang
    tip = tip + 0.5*size * tang
    perp   = np.array([-tang[1], tang[0]])
    head   = tip
    left   = tip - size*tang + (size*0.38)*perp
    indent = tip - size*0.78*tang
    right  = tip - size*tang - (size*0.38)*perp
    ax.add_patch(mpatches.Polygon(
        [head, left, indent, right],
        closed=True, facecolor=color, edgecolor=color,
        lw=0.0, zorder=5,
    ))
    
def pip_positions(n, s=0.55):
    o = s * 0.48    
    layouts = {
        1: [(0.00,  0.00)],
        2: [(-o,  o), ( o, -o)],
        3: [(0.00, 0.00), (-o,  o), ( o, -o)],
        4: [(-o,  o), ( o,  o), (-o, -o), ( o, -o)],
        5: [(0.00, 0.00), (-o,  o), ( o,  o), (-o, -o), ( o, -o)],
        6: [(-o,  o), ( o,  o), (-o, 0.00), ( o, 0.00), (-o, -o), ( o, -o)],
    }
    return layouts.get(min(n, 6), [(0, 0)])

def draw_decay_inlay(ax, colors, N, mass_array, N_max=6, sub_w_in=4.5, sub_h_in=4.0):
    
    spacing = 2.0
    s, pad  = 0.55, 0.30
    hs, e   = s/2, s/2 + pad

    total_w  = (N_max - 1) * spacing
    x_offset = (N_max - N) * spacing
    x_span   = total_w + 1.4
    pos      = [(x_offset + k * spacing, 0.0) for k in range(N)]

    y_bot = -e
    for i in range(N):
        for j in range(i):
            sep = (N-1-j) - (N-1-i)
            if sep > 1:
                level = sep - 1
                rad   = -(0.16 + 0.10 * level)
                dx    = sep * spacing
                cy    = -e + rad * dx
                y_mid = -e * 0.5 + 0.5 * cy
                y_bot = min(y_bot, y_mid - 0.25)

    y_top  = 1.3
    y_span = y_top - y_bot

    inset_w = 0.95
    inset_h = inset_w * (sub_w_in / sub_h_in) * (y_span / x_span)
    inset_x = (1.0 - inset_w) / 2              
    inset_y = 1.0 - inset_h + 0.0

    inset = ax.inset_axes([inset_x, inset_y, inset_w, inset_h])
    inset.set_xlim(-0.7, total_w + 0.7)
    inset.set_ylim(y_bot, y_top)
    inset.set_aspect('equal')
    inset.axis('off')

    for k in range(N):
        flavor = N - 1 - k
        x, y   = pos[k]
        inset.add_patch(mpatches.FancyBboxPatch(
            (x - hs, y - hs), s, s,
            boxstyle  = "round,pad=0.30",
            linewidth = 1.0,
            edgecolor = "black",
            facecolor = colors[flavor],
            alpha     = 1.0,
            zorder    = 2,
        ))
        for dx, dy in pip_positions(N - k, s=s):
            inset.plot(x + dx, y + dy, "ko", ms=3.5, zorder=3)

    for i in range(N):
        for j in range(i):
            src_k = N - 1 - i
            dst_k = N - 1 - j
            sep   = dst_k - src_k

            if sep == 1:
                stealth_arrow(inset,
                    pos[src_k][0] + e,  0.0,
                    pos[dst_k][0] - e,  0.0,
                    rad=0.0)
            else:
                level = sep - 1
                rad   = -(0.1 + 0.05 * level)
                stealth_arrow(inset,
                    pos[src_k][0], -e,
                    pos[dst_k][0], -e,
                    rad=rad)

    colors_ = [darken(c, 0.7) for c in colors]
    for k in range(N):
        flavor   = N - 1 - k
        x, _     = pos[k]
        orig_idx = N - 1 - k          
        m        = mass_array[orig_idx]

        if orig_idx == 0 or orig_idx > 2:
            exp = int(np.floor(np.log10(abs(m)))) if m != 0 else 0
            man = m / 10**exp if m != 0 else 0

            if abs(man - 1) < 1e-9:
                exp_str = f"\u2212{abs(exp)}" if exp < 0 else str(exp)
                lbl = rf"$\mathtt{{10^{{{exp_str}}}}}$ eV"
            elif 0.1 <= abs(m) < 10:
                lbl = rf"${m:g}$ eV"
            else:
                exp_str = f"\u2212{abs(exp)}" if exp < 0 else str(exp)
                lbl = rf"$\mathtt{{{man:g}\cdot10^{{{exp_str}}}}}$ eV"

            inset.text(x, hs + 0.42, lbl, color=colors_[flavor],
                       fontsize=8.5, ha='center', va='bottom')

        else:
            lbl = rf"$m_1 + \sqrt{{\Delta m^2_{{{orig_idx+1}1}}}}$"
            inset.text(x, hs + 0.42, lbl, color=colors_[flavor],
                       fontsize=5.5, ha='center', va='bottom')

def draw_dice_inlay(ax, colors, N, N_max=6, sub_w_in=4.5, sub_h_in=4.0,
                     inset_x_offset=0.0, skip_pairs=None):
    spacing = 2.0
    s, pad  = 0.55, 0.30
    hs, e   = s/2, s/2 + pad

    total_w  = (N_max - 1) * spacing
    x_offset = (N_max - N) * spacing
    x_span   = total_w + 1.4
    pos      = [(x_offset + k * spacing, 0.0) for k in range(N)]

    y_bot = -e
    for i in range(N):
        for j in range(i):
            if skip_pairs and (i, j) in skip_pairs:
                continue
            sep = (N-1-j) - (N-1-i)
            if sep > 1:
                level = sep - 1
                rad   = -(0.16 + 0.10 * level)
                dx    = sep * spacing
                cy    = -e + rad * dx
                y_mid = -e * 0.5 + 0.5 * cy
                y_bot = min(y_bot, y_mid - 0.25)

    y_top  = 1.0
    y_span = y_top - y_bot
    
    dice_bottom = -(hs + pad)
    y_bot = min(y_bot, dice_bottom - 0.05)

    inset_w = 0.95
    inset_h = inset_w * (sub_w_in / sub_h_in) * (y_span / x_span)
    inset_x = (1.0 - inset_w) / 2 + inset_x_offset
    inset_y = 1.0 - inset_h + 0.0

    inset = ax.inset_axes([inset_x, inset_y, inset_w, inset_h])
    inset.set_xlim(-0.7, total_w + 0.7)
    inset.set_ylim(y_bot, y_top)
    inset.set_aspect('equal')
    inset.axis('off')

    for k in range(N):
        flavor = N - 1 - k
        x, y   = pos[k]
        inset.add_patch(mpatches.FancyBboxPatch(
            (x - hs, y - hs), s, s,
            boxstyle  = "round,pad=0.30",
            linewidth = 1.0,
            edgecolor = "black",
            facecolor = colors[flavor],
            alpha     = 1.0,
            zorder    = 2,
        ))
        for dx, dy in pip_positions(N - k, s=s):
            inset.plot(x + dx, y + dy, "ko", ms=3.5, zorder=3)

    for i in range(N):
        for j in range(i):
            if skip_pairs and (i, j) in skip_pairs:
                continue
            src_k = N - 1 - i
            dst_k = N - 1 - j
            sep   = dst_k - src_k

            if sep == 1:
                stealth_arrow(inset,
                    pos[src_k][0] + e, 0.0,
                    pos[dst_k][0] - e, 0.0,
                    rad=0.0)
            else:
                level = sep - 1
                rad   = -(0.16 + 0.10 * level)
                stealth_arrow(inset,
                    pos[src_k][0], -e,
                    pos[dst_k][0], -e,
                    rad=rad)
    
def add_matrix_inlay_log(ax, e_edges, matrix, N, colors, nu_labels):
    
#     lim   = ax.get_ylim()[1]
    inset = ax.inset_axes([0.53, 0.13, 0.38, 0.3])
    
    stacked_stairs(inset, e_edges, matrix, N, colors, nu_labels)
    
    inset.set_yscale("log")
    inset.set_xlim(e_edges[0]/MeV, e_edges[-1]/MeV)
    inset.set_ylim(1, 5e8)          
    inset.set_xticks([])
    inset.yaxis.tick_right()
    inset.set_yticks([1, 1e6])
#     inset.set_yticklabels(["1", f"{lim:.3g}"])
    inset.tick_params(length=2)
#     inset.set_ylim(lim * 1e-6, lim)

def draw_double_dice_inlay(ax, colors, N, N_max=6, sub_w_in=4.5, sub_h_in=4.0,
                     inset_x_offset=0.0, skip_pairs=None, row_gap=2):
    
    N1 = min(3, N)
    N2 = N - N1
    spacing = 2.0
    s, pad  = 0.55, 0.30
    hs, e   = s/2, s/2 + pad
    total_w  = (N_max - 1) * spacing
    x_span   = total_w + 1.4

    x_off1 = (N_max - N1) * spacing
    x_off2 = (N_max - N2) * spacing
    pos1   = [(x_off1 + k * spacing,         0.0) for k in range(N1)]
    pos2   = [(x_off2 + k * spacing, -row_gap   ) for k in range(N2)]

    y_top = 1.0
    y_bot = -(e + row_gap + 0.05)
    for pos_row, N_row, above in [(pos1, N1, False), (pos2, N2, True)]:
        y_row = pos_row[0][1]
        sign  = +1 if above else -1
        for i in range(N_row):
            for j in range(i):
                if skip_pairs and (i, j) in skip_pairs:
                    continue
                sep = (N_row - 1 - j) - (N_row - 1 - i)
                if sep > 1: 
                    level = sep - 1
                    rad = -(0.05 + 0.05 * level) # hello
                    cy    = y_row + sign * e + sign * abs(rad) * (sep * spacing)
                    y_mid = (y_row + sign * e + cy) * 0.5
                    y_bot = min(y_bot, y_mid - 0.25)

    y_span  = y_top - y_bot
    inset_w = 0.95
    inset_h = inset_w * (sub_w_in / sub_h_in) * (y_span / x_span)
    inset_x = (1.0 - inset_w) / 2 + inset_x_offset
    inset_y = 1.0 - inset_h
    inset   = ax.inset_axes([inset_x, inset_y, inset_w, inset_h])
    inset.set_xlim(-0.7, total_w + 0.7)
    inset.set_ylim(y_bot, y_top)
    inset.set_aspect('equal')
    inset.axis('off')

    # draw dice
    for k in range(N1):
        flavor = N1 - 1 - k
        x, y   = pos1[k]
        inset.add_patch(mpatches.FancyBboxPatch(
            (x - hs, y - hs), s, s,
            boxstyle="round,pad=0.30", linewidth=1.0,
            edgecolor="black", facecolor=colors[flavor], alpha=1.0, zorder=2))
        for dx, dy in pip_positions(N1 - k, s=s):
            inset.plot(x + dx, y + dy, "ko", ms=3.5, zorder=3)

    for k in range(N2):
        flavor = N1 + (N2 - 1 - k)
        x, y   = pos2[k]
        inset.add_patch(mpatches.FancyBboxPatch(
            (x - hs, y - hs), s, s,
            boxstyle="round,pad=0.30", linewidth=1.0,
            edgecolor="black", facecolor=colors[flavor], alpha=1.0, zorder=2))
        for dx, dy in pip_positions(N2 - k, s=s):
            inset.plot(x + dx, y + dy, "ks", ms=3.5, zorder=3)

    # within-row arrows
    for pos_row, N_row, above in [(pos1, N1, False), (pos2, N2, True)]:
        y_row = pos_row[0][1]
        sign  = +1 if above else -1
        for i in range(N_row):
            for j in range(i):
                if skip_pairs and (i, j) in skip_pairs:
                    continue
                src_k = N_row - 1 - i
                dst_k = N_row - 1 - j
                sep   = dst_k - src_k
                if sep == 1:
                    stealth_arrow(inset,
                        pos_row[src_k][0] + e, y_row,
                        pos_row[dst_k][0] - e, y_row,
                        rad=0.0)
                else:
                    level = sep - 1
                    rad   = -(0.05 + 0.05 * level) 
                    stealth_arrow(inset,
                        pos_row[src_k][0], y_row + sign * e,
                        pos_row[dst_k][0], y_row + sign * e,
                        rad=sign * abs(rad))

    # corner arrows
    top_pip3_x, top_pip3_y = pos1[N1 - 3]
    bot_pip3_x, bot_pip3_y = pos2[N2 - 3]

    f = -0.05

    for k in [N1 - 1, N1 - 2]:
        tx, ty = pos1[k]
        stealth_arrow(inset,
            bot_pip3_x + e + f, bot_pip3_y + e + f,
            tx - e - f,         ty - e - f,
            rad=0.0, t_head=0.35, color="royalblue")

    for k in [N2 - 1, N2 - 2]:
        bx, by = pos2[k]
        stealth_arrow(inset,
            top_pip3_x + e + f, top_pip3_y - e - f,
            bx - e - f,         by + e + f,
            rad=0.0, t_head=0.35, color="royalblue")
        
def mantissa(m1_):
    exp = int(np.floor(np.log10(abs(m1_))))
    man = m1_ / 10**exp
    
    if abs(man - 1) < 1e-9:
        return rf"$\mathtt{{10^{{{exp}}}}}$"
    return rf"$\mathtt{{{man:g}\times10^{{{exp}}}}}$"

#############################################################################
 
def dice_plot(e_edges, mass_array, flav_array, init_state, g_s, dist, masses):
    
    N_values=[3,4,5,6]
    nu_labels=[r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$", r"$\nu_4$", r"$\nu_5$", r"$\nu_6$"]
    
    colors = sns.color_palette("RdBu", 6)

    sub_w, sub_h = 4.4, 2.7
    fig, axes = plt.subplots(4, 1, figsize=(sub_w, sub_h*4), sharex=True, sharey=True)

    info = "\n".join([
        rf"$\mathtt{{g}} = \mathtt{{{g_s}}}$",
        rf"$\mathtt{{L}} = \mathtt{{{dist/km:g}}}$ km",
        r"$\mathtt{Dirac}$"+" "+r"$\nu$"
    ])

    axes[0].text(0.0375, 0.94, info,transform=axes[0].transAxes, fontsize=8.5, 
    va="top", ha="left", color=decay_purple,linespacing=1.6,bbox=dict(boxstyle="round,pad=0.3", 
    facecolor="white", edgecolor=decay_purple, linewidth=0.8, alpha=0.9))
        
    for ax, N, mass_, flav_, init_ in zip(axes, N_values, mass_array, flav_array, init_state):

        stacked_stairs(ax, e_edges, mass_, N, colors, nu_labels)
        empty_stairs(ax, e_edges, init_, N, nu_labels=["None"]*N)

        add_matrix_inlay(ax, e_edges, flav_, N, np.flip(sns.color_palette("RdGy", 7)), nu_labels)

        ax.set_xlim(0,5)
        ax.set_ylim(0,5)

        ax.text(6.5/2, 0.85, rf"$\textrm{{Initial Spectrum, pure }}\nu_{N}$", fontsize=8, va='center')

    axes[-1].set_xlabel("Neutrino energy [MeV]", fontsize=12)

    for ax in axes:
        ax.set_ylabel("Flux [a.u.]", fontsize=12)

    plt.tight_layout(pad=0.05)

    for ax, N in zip(axes, N_values):

        pos = ax.get_position()
        sub_w_in = pos.width * fig.get_size_inches()[0]
        sub_h_in = pos.height * fig.get_size_inches()[1]

        draw_decay_inlay(ax, colors, N, masses,
                         sub_w_in=sub_w_in,
                         sub_h_in=sub_h_in)

    return fig, axes

def comp_plot(test_bins, result, g_s, dist):

    decay_purple_inv = (0.7, 1.0, 0.2)
    ingoing_inv = "#9C6A11"
    outgoing_inv = "#00A3A3"

    display_names = {
        "LIND":  "Lindbladian ODE",
        "LIOUV": "Dynamical Map/Kraus",
    }

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    x = np.array(test_bins, dtype=float)
    x_fine = np.linspace(x.min(), x.max(), 300)

    for (name, times), color in zip(result.items(), [ingoing_inv, outgoing_inv]):
        label = display_names.get(name, name)
        y = np.array(times, dtype=float) 
    #     / np.array(times, dtype=float)[0]
        fitted_exp = np.polyfit(np.log(x), np.log(y), 1)[0]
        if name=="LIND":
            exp_theory = 3
        else:
            exp_theory = 2
        y_theory = y[0] * (x_fine / x[0]) ** exp_theory

        ax.loglog(x, y, 'o-', color=color, lw=1, ms=4,
                  label=rf'{label}  $\mathcal{{O}}(N_E^{{{fitted_exp:.2f}}})$')
        line, = ax.loglog(x_fine, y_theory, '--', color=color, alpha=0.5)
        ax.text(x_fine[-1] * 1.05, y_theory[-1],rf'$\mathcal{{O}}(N_E^{{{exp_theory}}})$',
            color=color, fontsize=7, va='center', ha='left', alpha=0.8
        )

    inlay_text = "\n".join([
        rf"$\mathtt{{g}} = \mathtt{{{g_s}}}$",
        rf"$\mathtt{{L}} = \mathtt{{{dist/km:g}}}$ km",
        rf"$\mathtt{{N_t}} = \mathtt{{{100}}}$",
        rf"$\mathtt{{N_\nu}} = \mathtt{{{3}}}$",
    ])
    ax.text(
        0.045, 0.935, inlay_text,
        transform=ax.transAxes,
        fontsize=8.5, va="top", ha="left",
        color=decay_purple, linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=decay_purple, linewidth=0.8, alpha=0.9)
    )

    pos      = ax.get_position()
    sub_w_in = pos.width  * fig.get_size_inches()[0]
    sub_h_in = pos.height * fig.get_size_inches()[1]
    draw_dice_inlay(ax, ["white"]*6, 3, sub_w_in=3.84, sub_h_in=2.71,
                     inset_x_offset=-0.27,
                     skip_pairs={(2, 0)})

    ax.set_xlabel(r'Number of energy bins, $N_E$', fontsize=11)
    ax.set_ylabel(r'Wall time [$s$]', fontsize=11)

    ax.set_ylim(0.5,2e5)

    ax.set_xticks([50, 100, 200, 300, 500, 700])
    ax.set_xticklabels(['50', '100', '200', '300', '500', '700'], fontsize=11)
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    ax.grid(False)
    
    return fig, ax


def owl_plot(e_edges, owl, lind, g_s, dist, m1_):

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green

    colors_ = np.flip(sns.color_palette("RdGy", 7))

    nu_labels = [r"$\nu_e$", r"${\nu}_\mu$", r"$\nu_\tau$"]

    sub_w, sub_h = 4.4, 3.0
    
    num = len(owl)

    fig, ax_top = plt.subplots(figsize=(sub_w, sub_h)) 
    add = [0.6, 0.40, 0.0]
    for k, a in enumerate(add):
        ax_top.stairs([a]*num + owl.T[k], e_edges/MeV, color=colors[k], linestyle="solid", linewidth=1)
        ax_top.stairs([a]*num + lind.T[k], e_edges/MeV, color=colors[k], linestyle="dashed", linewidth=1)
        ax_top.fill_between(
            e_edges[:-1]/MeV, [a]*num + lind.T[k], [a]*num + owl.T[k], step="post", color=colors[k], alpha=0.3)

    ax_top.text(4.26, 0.84, nu_labels[0] + ' $+$ ' + str(add[0]), color=colors[0], 
                fontsize=10, va='center', ha='left')
    ax_top.text(4.26, 0.68, nu_labels[1] + ' $+$ ' + str(add[1]), color=colors[1], 
                fontsize=10, va='center', ha='left')
    ax_top.text(4.26, 0.3, nu_labels[2] + ' $+$ ' + str(add[2]), color=colors[2], 
                fontsize=10, va='center', ha='left')

    ax_top.text(3.06, 1.07, r'$\textrm{Initial Spectrum, pure }$' + nu_labels[1], color="k", 
                fontsize=8, va='center', ha='left')

    model_lines = [
        plt.Line2D([0], [0], color="k", linestyle="solid", lw=1, label="OWL"),
        plt.Line2D([0], [0], color="k", linestyle="dashed", lw=1, label="OQS")
    ]
    diff_patch = mpatches.Patch(facecolor="k", alpha=0.3, label="Difference")
    handles = model_lines + [diff_patch]
    ax_top.legend(handles=handles,loc=(0.27,0.72), fontsize=10, frameon=False,
    ncol=1,columnspacing=1.2)
    plt.xlim(0,5)
    plt.ylim(0,2.3)

    ax_top.set_ylabel("Flux [a.u.]", fontsize=11)
    ax_top.set_xlabel("Neutrino energy [MeV]", fontsize=11)

    info = "\n".join([
        rf"$\mathtt{{g}} = \mathtt{{{g_s}}}$",
        rf"$\mathtt{{L}} = \mathtt{{{dist/km:g}}}$ km",
        rf"$\mathtt{{m_1}} =$ " + mantissa(m1_)+ " eV",
        r"$\mathtt{Dirac}$"+" "+r"$\nu$"
    ])

    ax_top.text(0.0375, 0.94, info,transform=ax_top.transAxes,
        fontsize=8.5, va="top", ha="left", color=decay_purple,
        linespacing=1.6,bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
        edgecolor=decay_purple, linewidth=0.8, alpha=0.9))

    pos      = ax_top.get_position()
    sub_w_in = pos.width  * fig.get_size_inches()[0]
    sub_h_in = pos.height * fig.get_size_inches()[1]
    draw_dice_inlay(ax_top, ["white"]*6, 3, sub_w_in=3.84, sub_h_in=2.71,
                     inset_x_offset=0.039,
                     skip_pairs={(1, 0)})

    plt.axhline(1, color="k", linewidth=1, ls="dotted")

    plt.tight_layout(pad=0.25)
    
    
def maj_plot(e_edges, maj_flav, maj_mass, flux, g_s, dist, m1_):

    nu_labels = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$",
                 r"$\bar\nu_e$", r"$\bar\nu_\mu$", r"$\bar\nu_\tau$"]

    palette   = sns.color_palette("RdBu", 6)
    colors    = [palette[0], palette[1], palette[2], palette[5], palette[4], palette[3]]

    colors_flav = list(np.flip(sns.color_palette("RdGy", 6))[:3]) + sns.color_palette("PRGn", 6)[:3]
    colors_flav_ = [darken(c, 0.7) for c in colors_flav]
    
    decay_purple_inv = (0.7, 1.0, 0.2)

    info = "\n".join([
        rf"$\mathtt{{g}} =$ " + mantissa(g_s),
        rf"$\mathtt{{L}} = \mathtt{{{dist/km:g}}}$ km",
        rf"$\mathtt{{m_1}} =$ " + mantissa(m1_) + " eV",
        r"$\mathtt{Maj.}$"+" "+r"$\nu\bullet/\bar{\nu}\hspace{0.85ex}\rule{0.85ex}{0.85ex}$"
    ])

    fig, ax = plt.subplots(figsize=(4.4, 3.1))

    ax.text(0.04, 0.94, info,transform=ax.transAxes, fontsize=8.5, va="top", 
    ha="left", color=decay_purple,linespacing=1.6,bbox=dict(boxstyle="round,pad=0.3", 
    facecolor="white",edgecolor=decay_purple, linewidth=0.8, alpha=0.9))

    flux_ = np.zeros((len(flux), 6))
    flux_[:, 3] = flux      

    empty_stairs(ax, e_edges, flux_, 6, nu_labels=["None"]*6)
    stacked_stairs(ax, e_edges, maj_flav, 6, colors_flav, nu_labels)

    legend_elements = [mpatches.Patch(facecolor=colors_flav[i], label=nu_labels[i]) for i in range(6)]
    ax.legend(handles=legend_elements, fontsize=10, 
                            labelcolor=colors_flav_, loc=(0.3,0.52), frameon=False, labelspacing=0.375)

    ax.set_xlim(0, 3)
    ax.set_ylim(1, 1e13)

    add_matrix_inlay_log(ax, e_edges, maj_mass, 6, colors, [None]*6)

    ax.set_xlabel("Neutrino energy [MeV]", fontsize=12)
    ax.set_ylabel("Flux [cm${}^{-2}$ s${}^{-1}$ MeV${}^{-1}$]", fontsize=12)
    ax.text(2.42, 6e6, r'$\textrm{Initial Spectrum, pure }$' + nu_labels[3],
            color="k", fontsize=8, va='center', ha='center')

    pos      = ax.get_position()
    sub_w_in = pos.width  * fig.get_size_inches()[0]
    sub_h_in = pos.height * fig.get_size_inches()[1]
    draw_double_dice_inlay(ax, colors, 6, sub_w_in=3.84, sub_h_in=2.71,
                     inset_x_offset=0.0, skip_pairs={(1, 0)})

    plt.vlines(x=1.806, ymin=0.1, ymax=1e1, lw=0.75, color="k")
    ax.text(1.77, 5,r'$\textrm{IBD}$' + '\n' + r'$\textrm{threshold}$',
            color="k", fontsize=8, va='center', ha='right')

    plt.yscale("log")
    plt.tight_layout(pad=0.25)