# nuDICE
>neutrino Decay & Interference Computational Engine

A package for computing neutrino oscillation+decay probabilities using methods from quantum information theory. In particular, nuDICE offers routines to time-evolve an ensemble of density matrices describing a neutrino state with a broad energy spectrum, and to compute the resulting probabilities in both the mass and flavor bases.  Specifically, the package offers the following high-level functions:

- ⚀ `lind`: Solves the Lindblad master equation numerically for for a given set of decay channels and couplings.
- ⚁ `dynam`: Computes the dynamical map from the matrix exponential of the Liouvillian and applies it to the vectorized initial state. 

Moreover, the package includes the following helper functions:

- ⚂ `Uall`: computes the full mixing matrix for a given set of mixing angles and CP-violating phase.
- ⚃ `make_majorana`: constructs the Majorana version of a given set of masses and decay channels, which includes both neutrinos and antineutrinos.
- ⚄ `mass_to_flav`: transforms a density matrix in the mass basis to the flavour basis.
- ⚅ `flav_to_mass`: transforms a density matrix in the flavour basis to the mass basis.

The package is based on the following publication:

Joachim Kopp and George A. Parker, "Visible Neutrino Decay As An Open Quantum System", [arXiv:2604.09776](https://arxiv.org/abs/2604.09776) [hep-ph].
