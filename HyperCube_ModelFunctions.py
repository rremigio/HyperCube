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
                
            # Get linear components
            slope = kwargs[f'slope{region}']
            intercept = kwargs[f'intercept{region}']
            
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
            y_fit[region_mask] = slope * x[region_mask] + intercept + gaussian_sum
            
        return y_fit
    
    
