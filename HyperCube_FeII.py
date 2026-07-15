#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyperCube_FeII.py
=================
Broad Fe II pseudo-continuum templates for HyperCube (Type 1 AGN fitting).

This module is pure (no Qt) and dependency-light (numpy only — it does NOT require
pPXF), so it can be imported cleanly inside multiprocessing workers. It loads an
empirical Fe II template from disk (feii_templates/*.txt), resamples it to a uniform
ln(lambda) grid once, and re-renders it per fit-evaluation.

Depending on the Fe II template chosen, the number of free parameters changes.
Currently, only the Mrk 493 STIS template from Park et al. 2022 is implemented,
and this has three free parameters: amplitude, velocity offset, and  velocity 
dispersion.

Fe II is treated as a continuum component. 

Bundled templates
----------------
park2022 : Park et al. (2022) empirical Fe II template built from the HST/STIS
           spectrum of Mrk 493. Covers rest framewavelengths from 4000-5600 A.
           Sampled at 2 A. Measured FWHM from several emission lines is 15 A.

TO DO: Add the 2025 version of the Kovacevic template.

Public API (PROVISIONAL — will grow as the composite continuum is wired up)
----------
list_templates()                          -> dict of template metadata for the UI
FeIITemplate(name).load_and_prepare(velscale, z)
                                          -> read + normalize + log-rebin, bake in z
   .eval(x_vals, amp, v_shift, sigma)     -> Fe II model on observed-frame x_vals
"""

import os
import glob
import numpy as np

C_KMS = 299792.458  # speed of light, km/s

# Folder holding the bundled Fe II template text files.
_HERE = os.path.dirname(os.path.abspath(__file__))
_FOLDER = os.path.join(_HERE, 'feii_templates')

TEMPLATES = {
    'park2022': {
        'folder': 'park_2022',
        'kind': 'spectrum',
        'fwhm': 15.0,
        'label': 'Park et al. 2022 Fe II Template',
        'note': 'Empirical, from HST/STIS of Mrk 493 (incl. Ti/Ni/Cr II).',
        'lam_min': 4000,
        'lam_max': 5600,
    },
    # TODO: 'kovacevic2010' / '...2025' -> kind='linegroups'. Semi-empirical F/G/S/P
    # multiplet groups; intra-group ratios set by an excitation temperature, so the
    # free-parameter set is larger (per-group amplitudes + T + width + shift).
}

def list_templates():
    """
    Return {name: {...meta..., n_files}} for every bundled template that has files
    on disk (for the composite-continuum UI). Skips registry entries whose folder is
    missing/empty.
    """
    out = {}
    for name, meta in TEMPLATES.items():
        folder = os.path.join(_FOLDER, meta['folder'])
        files = sorted( glob.glob (os.path.join(folder, '*.csv') ) )
        if not files:
            continue
        out[name] = dict(meta, n_files=len(files))
    return out


# ── template readers ─────────────────────────────────────────────────────────
# 
# As mentioned previously, Fe II templates can come in different forms. 
# The simplest of these is a single spectrum, such as the templates from
# Park et al. (2022), or Boroson and Green (1992). Currently, only templates
# of this type are  supported.
#
# In the future, I plan to add the template from Kovacevic et al. (2025), which
# is a more improved version of the 2010 template. The template comes as multiple
# CSVs containing different line lists for different Fe II multiplets. 
#
# To address the fact that Fe II templates come in different forms, each form
# (which I will denote as 'spectrum' and 'linegroups') will require different
# readers/loaders as well as evaluators and free parameter sets. The 'kind'
# key within TEMPLATES serves as the switch for which route the code goes through.
# The current type(s) implemented are:
#
# 'spectrum'- simple wavelength and flux vectors. Evaluated by convolving, shifting,
#             and scaling the spectrum. 
#             Three free parameters: amplitude, velocity offset, velocity dispersion.

def _loadtable(path):
    """
    Read a CSV as a numpy array. This assumes that all Fe II templates
    are in the form of CSVs.
    """
    return np.genfromtxt(path, delimiter=',', comments='#')


def _read_spectrum(folder):
    """
    Read in a 'spectrum' template an return a dictionary.
    """
    files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No Fe II template file in {folder}")
    data = _loadtable(files[0])
    lam = np.asarray(data[:, 0], dtype=float)
    flux = np.asarray(data[:, 1], dtype=float)
    order = np.argsort(lam)
    return {'kind': 'spectrum', 'lam': lam[order],
            'flux': np.nan_to_num(flux[order])}


#def _read_linegroups(folder):
#    """'linegroups' kind (Kovacevic 2010/2025): read per-group line lists and return
#    the atomic data needed to generate the spectrum at a fitted excitation
#    temperature. NOT YET IMPLEMENTED."""
#    raise NotImplementedError(
#        "'linegroups' Fe II templates (Kovacevic 2010/2025) are not implemented yet: "
#        "intra-group line ratios are set by a fitted excitation temperature, so the "
#        "reader must return per-group line lists (lam, gf, E_upper, ...) and the "
#        "free-parameter set differs (per-group amplitudes + temperature).")


_READERS = {
    'spectrum': _read_spectrum,
    #'linegroups': _read_linegroups,
}


def _read_template(name):
    """
    General reader function for the Fe II templates.
    """
    if name not in TEMPLATES:
        raise ValueError(f"Unknown Fe II template: {name}")
    meta = TEMPLATES[name]
    folder = os.path.join(_FOLDER, meta['folder'])
    temp_dict = _READERS[meta['kind']](folder)
    temp_dict['name'] = name
    temp_dict['meta'] = meta
    return temp_dict


# ── data helper functions ───────────────────────────────────────────────────────────
def _log_rebin(lam, flux, velscale):
    """
    Resample a spectrum onto a uniform ln(lambda) grid at `velscale`
    (km/s/pixel) by interpolation. Returns (flux_log, ln_lam, velscale).

    Working in ln(lambda) makes a velocity shift/broadening a constant number of
    pixels, so the per-eval convolution is a single fixed-width kernel.

    This is a numpy-based alternative to the ppxf util log_rebin.

    Instead of conserving flux, we just interpolate onto a wavelength grid. 
    """

    lam = np.asarray(lam, dtype=float)
    ln_lo, ln_hi = np.log(lam[0]), np.log(lam[-1])

    dln = float(velscale) / C_KMS

    n = int(np.floor((ln_hi - ln_lo) / dln))
    
    ln_lam = ln_lo + dln * np.arange(n)

    # interpolate onto the wavelength grid
    flux_log = np.interp(np.exp(ln_lam), lam, flux)

    return flux_log, ln_lam, float(velscale)


def _vel_convolve(spec, velscale, v, sigma):
    """
    Alternate version of HyperCube_pPXF._losvd_convolve without the 
    GH terms.
    """
    sigma = max(float(sigma), velscale / 5.0)
    sig_pix = sigma / velscale
    v_pix = float(v) / velscale
    half = int(np.ceil(5 * sig_pix + abs(v_pix))) + 1
    # Keep the kernel no longer than the signal: np.convolve(mode='same') returns
    # length max(len(spec), len(k)), so a kernel longer than spec (large sigma/|v|,
    # or a narrow template) yields an oversized output that mismatches the caller's
    # wavelength grid in np.interp ("fp and xp are not of the same length").
    half = max(1, min(half, (len(spec) - 1) // 2))
    x = np.arange(-half, half + 1)
    k = np.exp(-0.5 * ((x - v_pix) / sig_pix) ** 2)
    s = k.sum()
    if s != 0:
        k = k / s
    return np.convolve(spec, k, mode='same')


# ── high-level API ───────────────────────────────────────────────────────────
class FeIITemplate:
    """
    Load and prepare an Fe II template for repeated evaluation inside the fit.

    For 'spectrum' templates, the prepared state is a normalized template on
    a wavelength grid sampled uniformly in ln(lambda). The eval method applies
    broadening, shifting, scaling, and then resamples it onto the data grid. 
    This is called for each fit iteration.

    Attributes for the other kinds of templates will be added later.
    """

    def __init__(self, name):
        if name not in TEMPLATES:
            raise ValueError(f"Unknown Fe II template: {name}")
        self.name = name
        self.meta = TEMPLATES[name]
        self.kind = self.meta['kind']

        # Intrinsic template width (km/s) from the measured FWHM (A) at mid-lambda;
        # the requested broadening adds in quadrature on top of this (see eval()).
        fwhm_A = float(self.meta.get('fwhm', 0.0))
        lam_mid = 0.5 * (float(self.meta['lam_min']) + float(self.meta['lam_max']))
        self.sigma_temp = (C_KMS * fwhm_A / (2.3548 * lam_mid)) if fwhm_A > 0 else 0.0

        # Prepared state (populated by load_and_prepare).
        self.flux_log = None      # normalized template on the ln_lam grid
        self.ln_lam = None        # uniform ln(lambda) grid (rest frame)
        self.velscale = None      # km/s/pixel of the ln_lam grid
        self.z = None             # galaxy redshift, baked in at prepare (rest -> observed)

    def load_and_prepare(self, velscale, z):
        """
        Read + normalize + log-rebin the template once, and bake in the galaxy
        redshift `z` (used to map the rest-frame template onto the observed-frame
        data grid in eval, mirroring how HyperCube_pPXF stores z in its cache).
        Returns self.
        """
        if self.kind != 'spectrum':
            raise NotImplementedError(
                f"load_and_prepare for kind='{self.kind}' is not implemented yet.")
        temp_dict = _read_template(self.name)
        lam, flux = temp_dict['lam'], temp_dict['flux']
        med = np.median(flux[flux > 0]) if np.any(flux > 0) else 1.0
        flux = flux / med                       # normalize -> amp ~ O(spectrum flux)
        self.flux_log, self.ln_lam, self.velscale = _log_rebin(lam, flux, velscale)
        self.z = float(z)
        return self

    def eval(self, x_vals, amp, v_shift, sigma):
        """
        Fe II model on observed-frame `x_vals`. The redshift was baked in at
        load_and_prepare, so eval takes only the three free parameters.
        """
        if self.flux_log is None:
            raise RuntimeError("Call load_and_prepare(velscale, z) before eval().")

        x_vals = np.asarray(x_vals, dtype=float)

        # Only add the width BEYOND the template's own intrinsic width (quadrature).
        extra = float(sigma) ** 2 - self.sigma_temp ** 2
        sig_add = np.sqrt(extra) if extra > 0 else self.velscale / 5.0

        broad = _vel_convolve(self.flux_log, self.velscale, v_shift, sig_add)
        lam_obs = np.exp(self.ln_lam) * (1.0 + self.z)

        y = np.interp(x_vals, lam_obs, broad, left=0.0, right=0.0)

        return float(amp) * y