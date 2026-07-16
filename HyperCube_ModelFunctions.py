#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:41:08 2025

@author: justin
"""

import re
import ast
import numpy as np

# Define the Gaussians and linear functions for each region
def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))

def linear(x, slope, intercept):
    return slope * x + intercept

def eval_spline(knots_x, knots_y, x):
    """Evaluate a continuum spline defined by (knots_x, knots_y) at x.

    Interpolating B-spline of order k = min(3, n_knots-1): 2 knots → linear,
    3 → quadratic, >=4 → cubic. Values outside the knot span are clamped to the
    nearest end knot. Mirrors ViewerWindow._eval_spline in HyperCube.py.
    """
    kx = np.asarray(knots_x, dtype=float)
    ky = np.asarray(knots_y, dtype=float)
    order = np.argsort(kx)
    kx, ky = kx[order], ky[order]
    keep = np.concatenate(([True], np.diff(kx) > 0))
    kx, ky = kx[keep], ky[keep]
    x = np.asarray(x, dtype=float)
    if kx.size < 2:
        return np.full_like(x, ky[0] if ky.size else 0.0)
    k = min(3, kx.size - 1)
    from scipy.interpolate import make_interp_spline
    spl = make_interp_spline(kx, ky, k=k)
    xc = np.clip(x, kx[0], kx[-1])
    return spl(xc)


def eval_poly(coefs, x1, x2, x):
    """Evaluate a Chebyshev-basis continuum polynomial (coefs over the domain
    [x1, x2]) at x. Mirrors ViewerWindow._eval_poly in HyperCube.py."""
    c = np.asarray(coefs, dtype=float)
    x = np.asarray(x, dtype=float)
    if c.size == 0:
        return np.zeros_like(x)
    if not (np.isfinite(x1) and np.isfinite(x2) and x2 > x1):
        return np.full_like(x, c[0])
    return np.polynomial.Chebyshev(c, domain=[float(x1), float(x2)])(x)

def eval_power_law(slope, amp, x, x_pivot=None):
    """Evaluate a power law amp * (x / x_pivot) ** slope.

    x_pivot anchors the amplitude (the flux at lambda = x_pivot). If None, it
    defaults to the midpoint of the supplied x range (half of the fitted region),
    which is the convention used when a caller does not pass an explicit pivot.
    Passing x_pivot explicitly keeps the pivot identical between the fit and the
    overlay, which sample x on different grids.
    """

    x = np.asarray(x, dtype=float)

    if x_pivot is None:
        # set the pivot wavelength to half of the wavelength range
        x_pivot = 0.5 * (x.max() + x.min())

    return amp * (x / x_pivot) ** slope

# Vectorized Gaussian computation
def sum_gaussians(x, params, num_gaussians):
    if num_gaussians == 0:
        return np.zeros_like(x)  # No Gaussians to compute
    amps, cens, sigmas = np.array(params[:num_gaussians]).T
    return np.sum(amps[:, None] * np.exp(-0.5 * ((x - cens[:, None]) / sigmas[:, None]) ** 2), axis=0)

# ── Composite-continuum component registry ───────────────────────────────────
# A region's continuum is an ordered, additive list of *components*, each a dict
# {'type': <t>, '<sub>_0': init, '<sub>_fit': fit, ...(+ structural fields)}.
# Supported types: linear, poly, spline, powerlaw, feii, stellar. This registry
# is the single source of truth for their parameter schema so the model (below),
# the param builder (HyperCube.py) and the fit kernel (HyperCube_fit.py) agree.
#
# lmfit param naming (flat, 2-level): region r (1-based), in-model component index
# k (0-based) -> c{r}_{k}_<sub>  (vectors: c{r}_{k}_polyc_{j} / c{r}_{k}_knoty_{i},
# fixed knot abscissae c{r}_{k}_knotx_{i}).

COMPONENT_TYPES = ('linear', 'poly', 'spline', 'powerlaw', 'feii', 'stellar',
                   'type1_agn')

# Types evaluated INSIDE the joint lmfit model. 'stellar' is currently
# pre-subtracted (in_model=False); flipping it True (+ a scale param + a stellar
# branch in eval_component) folds it into the joint fit later.
# 'type1_agn' = a FROZEN unresolved AGN bundle (power law + Fe II + BLR lines)
# whose SHAPE was locked at a nucleus spaxel; the only free param is one overall
# amplitude agn_amp that traces the PSF across the cube. Its payload is the frozen
# bundle spectrum (wl, flux) in physical units; see eval_component.
_IN_MODEL = {'linear': True, 'poly': True, 'spline': True,
             'powerlaw': True, 'feii': True, 'stellar': False, 'type1_agn': True}

# Scalar sub-parameters per type: (subname, default, min, max). poly/spline carry
# variable-length vectors instead (handled explicitly).
_SCALAR_SUBPARAMS = {
    'linear':   [('slope', 0.0, None, None), ('intercept', 0.0, None, None)],
    'powerlaw': [('pl_amp', 1.0, None, None), ('pl_slope', -1.5, None, None)],
    'feii':     [('feii_amp', 1.0, 0.0, None), ('feii_v', 0.0, -3000.0, 3000.0),
                 ('feii_sigma', 2000.0, 100.0, 10000.0)],
    'type1_agn': [('agn_amp', 1.0, 0.0, None)],
    'poly': [], 'spline': [], 'stellar': [],
}

# Sub-parameter bases whose values scale with the spectrum flux. For a linear
# component y = slope*x + intercept, y is flux and x is wavelength (fixed), so
# BOTH slope and intercept scale. pl_slope / feii_v / feii_sigma do NOT. agn_amp
# multiplies a physical-unit frozen bundle, so it scales with flux too (init 1.0
# at the nucleus -> recovers to ~1.0 there under the scaled fit).
FLUX_SCALED = {'slope', 'intercept', 'pl_amp', 'feii_amp', 'polyc', 'knoty', 'agn_amp'}

_CPARAM_RE = re.compile(r'^c\d+_\d+_(.+)$')


def is_in_model(ctype):
    return _IN_MODEL.get(str(ctype), False)


def _cf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def as_float_list(v):
    """Coerce a value (list, repr-string from CSV, or None) to a list of floats."""
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except Exception:
            return []
    try:
        return [float(z) for z in v]
    except Exception:
        return []


def coerce_components(value):
    """Return a clean list-of-dicts from a df_cont 'components' cell (a real list,
    a repr string from CSV, or None/NaN)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            # repr() of a cell holding np.nan / np.inf writes bare `nan`/`inf`,
            # which ast.literal_eval rejects — map them to None and retry.
            try:
                _clean = re.sub(r'[-+]?\b(?:nan|inf|Infinity)\b', 'None', value)
                value = ast.literal_eval(_clean)
            except Exception:
                return []
    if isinstance(value, dict):
        value = [value]
    try:
        return [dict(c) for c in value if isinstance(c, dict)]
    except Exception:
        return []


def legacy_to_component(cont_type, row):
    """Build one component dict from a legacy df_cont row (exclusive cont_type +
    flat columns). `row` is a mapping (pandas Series / dict). Used to migrate
    pre-composite sessions/CSVs and every existing single-type region."""
    ct = str(cont_type) if cont_type is not None else 'linear'
    g = (lambda k, d=np.nan: row.get(k, d)) if hasattr(row, 'get') else (lambda k, d=np.nan: d)
    if ct == 'poly':
        coefs = as_float_list(g('poly_coef_0'))
        deg = _cf(g('poly_degree'))
        return {'type': 'poly',
                'poly_degree': int(deg) if np.isfinite(deg) else max(len(coefs) - 1, 0),
                'poly_coef_0': coefs, 'poly_coef_fit': as_float_list(g('poly_coef_fit')) or coefs}
    if ct == 'spline':
        return {'type': 'spline', 'knots_x': as_float_list(g('knots_x')),
                'knots_y_0': as_float_list(g('knots_y_0')),
                'knots_y_fit': as_float_list(g('knots_y_fit'))}
    if ct == 'powerlaw':
        return {'type': 'powerlaw', 'pl_amp_0': _cf(g('pl_amp_0')),
                'pl_slope_0': _cf(g('pl_slope_0')), 'pl_amp_fit': _cf(g('pl_amp_fit')),
                'pl_slope_fit': _cf(g('pl_slope_fit'))}
    if ct == 'feii':
        return {'type': 'feii', 'feii_template': str(g('feii_template', '') or ''),
                'feii_amp_0': _cf(g('feii_amp_0')), 'feii_v_0': _cf(g('feii_v_0')),
                'feii_sigma_0': _cf(g('feii_sigma_0')), 'feii_amp_fit': _cf(g('feii_amp_fit')),
                'feii_v_fit': _cf(g('feii_v_fit')), 'feii_sigma_fit': _cf(g('feii_sigma_fit'))}
    if ct == 'stellar':
        mom = _cf(g('stellar_moments'))
        _md = _cf(g('stellar_mdegree'))
        _dg = _cf(g('stellar_degree'))
        comp = {'type': 'stellar', 'stellar_library': str(g('stellar_library', '') or ''),
                'stellar_moments': int(mom) if np.isfinite(mom) else 2,
                'stellar_mdegree': int(_md) if np.isfinite(_md) else 10,
                'stellar_degree': int(_dg) if np.isfinite(_dg) else -1}
        for f in ('stellar_V', 'stellar_sigma', 'stellar_h3', 'stellar_h4', 'stellar_scale'):
            comp[f + '_0'] = _cf(g(f + '_0'))
            comp[f + '_fit'] = _cf(g(f + '_fit'))
        return comp
    return {'type': 'linear', 'slope_0': _cf(g('Slope_0', 0.0)),
            'intercept_0': _cf(g('Intercept_0', 0.0)), 'slope_fit': _cf(g('Slope_fit')),
            'intercept_fit': _cf(g('Intercept_fit'))}


def add_component_params(params, r, k, comp):
    """Add the lmfit params for ONE in-model component (region r 1-based, index k
    0-based) under prefix c{r}_{k}_. Returns the structural descriptor
    {'type', 'ncoef'/'nknot'} the model reads the params back with."""
    t = str(comp['type'])
    pfx = f'c{r}_{k}_'
    desc = {'type': t}
    if t == 'poly':
        coefs = as_float_list(comp.get('poly_coef_0'))
        desc['ncoef'] = len(coefs)
        for j, c in enumerate(coefs):
            params.add(f'{pfx}polyc_{j}', value=float(c), vary=True)
    elif t == 'spline':
        kx = as_float_list(comp.get('knots_x'))
        ky = as_float_list(comp.get('knots_y_0'))
        desc['nknot'] = len(kx)
        ptp = (max(ky) - min(ky)) if len(ky) > 1 else 0.0
        delta = 0.5 * ptp if ptp > 0 else max(abs(np.mean(ky)) if ky else 1.0, 1.0)
        for i in range(len(kx)):
            params.add(f'{pfx}knotx_{i}', value=float(kx[i]), vary=False)
            yi = float(ky[i]) if i < len(ky) else 0.0
            params.add(f'{pfx}knoty_{i}', value=yi, vary=True, min=yi - delta, max=yi + delta)
    else:
        for sub, default, lo, hi in _SCALAR_SUBPARAMS.get(t, []):
            val = _cf(comp.get(sub + '_0'))
            if not np.isfinite(val) or (sub == 'feii_sigma' and val <= 0):
                val = default
            params.add(pfx + sub, value=val, vary=True, min=lo, max=hi)
    return desc


def is_component_param(name):
    """True if `name` is a continuum-component lmfit param (c{r}_{k}_<sub>)."""
    return bool(_CPARAM_RE.match(str(name)))


def is_flux_scaled_param(name):
    """True if a c{r}_{k}_<sub> lmfit param name is amplitude-like (flux-scaled)."""
    m = _CPARAM_RE.match(str(name))
    if not m:
        return False
    sub = m.group(1)
    head, _, tail = sub.rpartition('_')
    base = head if (head and tail.isdigit()) else sub
    return base in FLUX_SCALED


def eval_component(desc, x, x1, x2, kwargs, r, k):
    """Evaluate ONE in-model component over x (already masked to the region).
    `desc` is the structural descriptor from add_component_params /
    build_region_components; numeric params come from kwargs by c{r}_{k}_ name."""
    t = desc['type']
    pfx = f'c{r}_{k}_'
    if t == 'linear':
        return kwargs[pfx + 'slope'] * x + kwargs[pfx + 'intercept']
    if t == 'poly':
        n = int(desc.get('ncoef', 0))
        if n <= 0:
            return np.zeros_like(x)
        return eval_poly([kwargs[f'{pfx}polyc_{j}'] for j in range(n)], x1, x2, x)
    if t == 'spline':
        n = int(desc.get('nknot', 0))
        if n < 2:
            return np.zeros_like(x)
        kx = [kwargs[f'{pfx}knotx_{i}'] for i in range(n)]
        ky = [kwargs[f'{pfx}knoty_{i}'] for i in range(n)]
        return eval_spline(kx, ky, x)
    if t == 'powerlaw':
        pivot = 0.5 * (x1 + x2)
        return eval_power_law(kwargs[pfx + 'pl_slope'], kwargs[pfx + 'pl_amp'], x, pivot)
    if t == 'feii':
        tmpl = desc.get('payload')
        if tmpl is None:
            return np.zeros_like(x)
        return tmpl.eval(x, kwargs[pfx + 'feii_amp'], kwargs[pfx + 'feii_v'],
                         kwargs[pfx + 'feii_sigma'])
    if t == 'type1_agn':
        # Frozen unresolved bundle: payload = (wl_bundle, flux_bundle) in physical
        # units (agn_amp=1 == the nucleus). Only the overall amplitude varies.
        payload = desc.get('payload')
        if payload is None:
            return np.zeros_like(x)
        wl_b, fl_b = payload
        return kwargs[pfx + 'agn_amp'] * np.interp(x, wl_b, fl_b)
    return np.zeros_like(x)


def component_fit_from_params(comp, r, k, result_params, flux_scale):
    """Return a copy of `comp` with _fit fields filled from result_params
    (un-rescaling amplitude-like params by flux_scale)."""
    t = str(comp['type'])
    pfx = f'c{r}_{k}_'
    out = dict(comp)
    if t == 'linear':
        out['slope_fit'] = result_params[pfx + 'slope'].value * flux_scale
        out['intercept_fit'] = result_params[pfx + 'intercept'].value * flux_scale
    elif t == 'poly':
        n = len(as_float_list(comp.get('poly_coef_0')))
        out['poly_coef_fit'] = [result_params[f'{pfx}polyc_{j}'].value * flux_scale
                                for j in range(n)]
    elif t == 'spline':
        n = len(as_float_list(comp.get('knots_x')))
        out['knots_y_fit'] = [result_params[f'{pfx}knoty_{i}'].value * flux_scale
                              for i in range(n)]
    elif t == 'powerlaw':
        out['pl_amp_fit'] = result_params[pfx + 'pl_amp'].value * flux_scale
        out['pl_slope_fit'] = result_params[pfx + 'pl_slope'].value
    elif t == 'feii':
        out['feii_amp_fit'] = result_params[pfx + 'feii_amp'].value * flux_scale
        out['feii_v_fit'] = result_params[pfx + 'feii_v'].value
        out['feii_sigma_fit'] = result_params[pfx + 'feii_sigma'].value
    elif t == 'type1_agn':
        # agn_amp is dimensionless (=1 at the nucleus) but multiplies a physical
        # bundle, so it is flux-scaled like an amplitude.
        out['agn_amp_fit'] = result_params[pfx + 'agn_amp'].value * flux_scale
    return out


class PiecewiseModel:
    def __init__(self, n_regions, n_gaussians, region_components=None):
        self.n_regions = n_regions
        self.n_gaussians = n_gaussians
        # {region_number (1-based) -> ordered list of in-model component structural
        # descriptors {'type', ('ncoef'/'nknot'), ('payload': Fe II template)}}.
        # A region's continuum baseline is the SUM of its components. Empty/None ->
        # that region falls back to a single linear component (slope/intercept).
        self.region_components = region_components or {}

    def model_function(self, x, **kwargs):
        """
        Generalized piecewise model with N regions and M Gaussians.
        - Region boundaries: x{region_num}_start, x{region_num}_end
        - Continuum: an additive list of components per region (see region_components);
          component k of region r reads params named c{r}_{k}_<sub>. If a region has
          no descriptor, it falls back to slope{r}/intercept{r} (back-compat).
        - Gaussian parameters: amp{gauss_num}, cen{gauss_num}, sigma{gauss_num}
        - Gaussian assignments: NR{region_num} (number of Gaussians in each region)
        """
        y_fit = np.zeros_like(x)
        gaussian_index = 0  # Tracks which Gaussians we've used

        for region in range(1, self.n_regions + 1):
            # Get region boundaries
            x_start = kwargs[f'x{region}_start']
            x_end = kwargs[f'x{region}_end']
            region_mask = (x >= x_start) & (x < x_end)

            if not np.any(region_mask):
                continue

            xr = x[region_mask]
            # Continuum baseline = SUM over this region's additive components
            # (composite path, used by the interactive single-spaxel fit).
            comps = self.region_components.get(region)
            if comps:
                baseline = np.zeros_like(xr)
                for k, desc in enumerate(comps):
                    baseline = baseline + eval_component(
                        desc, xr, x_start, x_end, kwargs, region, k)
            else:
                # Legacy fallback (no component descriptors): the flat-param model
                # still used by the parallel cube path — polynomial if NP{r}>=1,
                # else spline if NK{r}>=2, else linear slope/intercept.
                n_poly = int(kwargs.get(f'NP{region}', 0))
                n_knots = int(kwargs.get(f'NK{region}', 0))
                if n_poly >= 1:
                    coefs = [kwargs[f'polyc{region}_{j}'] for j in range(n_poly)]
                    baseline = eval_poly(coefs, x_start, x_end, xr)
                elif n_knots >= 2:
                    kx = [kwargs[f'knotx{region}_{k}'] for k in range(n_knots)]
                    ky = [kwargs[f'knoty{region}_{k}'] for k in range(n_knots)]
                    baseline = eval_spline(kx, ky, xr)
                else:
                    baseline = kwargs.get(f'slope{region}', 0.0) * xr + \
                        kwargs.get(f'intercept{region}', 0.0)

            # Get number of Gaussians for this region
            n_region_gauss = kwargs[f'NR{region}']

            # Sum Gaussians for this region
            gaussian_sum = np.zeros_like(x[region_mask])
            for g in range(n_region_gauss):
                if gaussian_index >= self.n_gaussians:
                    raise ValueError("Not enough Gaussians defined for all regions")

                amp = kwargs[f'amp{gaussian_index + 1}']
                cen = kwargs[f'cen{gaussian_index + 1}']
                sigma = kwargs[f'sigma{gaussian_index + 1}']

                gaussian_sum += amp * np.exp(-(x[region_mask] - cen)**2 / (2 * sigma**2))
                gaussian_index += 1

            # Compute region model
            y_fit[region_mask] = baseline + gaussian_sum

        return y_fit
    
    
