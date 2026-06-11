#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:41:08 2025

@author: justin
"""

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

# Vectorized Gaussian computation
def sum_gaussians(x, params, num_gaussians):
    if num_gaussians == 0:
        return np.zeros_like(x)  # No Gaussians to compute
    amps, cens, sigmas = np.array(params[:num_gaussians]).T
    return np.sum(amps[:, None] * np.exp(-0.5 * ((x - cens[:, None]) / sigmas[:, None]) ** 2), axis=0)

class PiecewiseModel:
    def __init__(self, n_regions, n_gaussians):
        self.n_regions = n_regions
        self.n_gaussians = n_gaussians
        
    def model_function(self, x, **kwargs):
        """
        Generalized piecewise model with N regions and M Gaussians
        Parameters are passed as keyword arguments with specific naming conventions:
        - Region boundaries: x{region_num}_start, x{region_num}_end
        - Linear components: slope{region_num}, intercept{region_num}
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

            # Continuum baseline: polynomial if NP{region}>=1 (Chebyshev over
            # [x_start,x_end]); else spline if NK{region}>=2; else linear.
            n_poly = int(kwargs.get(f'NP{region}', 0))
            n_knots = int(kwargs.get(f'NK{region}', 0))
            if n_poly >= 1:
                coefs = [kwargs[f'polyc{region}_{j}'] for j in range(n_poly)]
                baseline = eval_poly(coefs, x_start, x_end, x[region_mask])
            elif n_knots >= 2:
                kx = [kwargs[f'knotx{region}_{k}'] for k in range(n_knots)]
                ky = [kwargs[f'knoty{region}_{k}'] for k in range(n_knots)]
                baseline = eval_spline(kx, ky, x[region_mask])
            else:
                slope = kwargs[f'slope{region}']
                intercept = kwargs[f'intercept{region}']
                baseline = slope * x[region_mask] + intercept

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
    
    
