#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:01:22 2025

@author: justin
   __version__ = "0.2.0"
"""



import sys
import numpy as np
import pandas as pd
import ast  # For safely evaluating strings as Python literals
import re
from functools import partial
import csv
import gc
import psutil
import os 
from pathlib import Path
import warnings

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.patches as patches

from astropy.io import fits
from astropy.wcs import WCS

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtWidgets import (
    QVBoxLayout, QPushButton, QLineEdit, QScrollArea, QGridLayout,
    QWidget, QLabel, QFrame, QMenu, QTextEdit, QMainWindow,
    QHBoxLayout, QMenuBar, QProgressBar, QDialog, QSplashScreen,
    QAction, QSplitter, QFileDialog, QApplication, QGroupBox, QMessageBox,
    QDockWidget, QComboBox, QScrollBar)
from PyQt5.QtGui import QFontMetrics, QPixmap
from PyQt5.QtGui import QKeySequence

from lmfit import Model, Parameters

import HyperCube_ModelFunctions







# Global variables for application-wide FITS access
global FITS_HEADER, FITS_DATA
FITS_HEADER = None
FITS_DATA = None

global viewing_fit
viewing_fit = False

global spectral_count
spectral_count = None

global df_obs, df, df_cont, df_fit

global fit_results
fit_results = []
constraints = []

global spectrum

global snr_map
global snr_value
snr_value = 0

data_observation_init = {'sourcename': [''],
                    'redshift': [''],
                    'resolvingpower': ['']}

df_obs = pd.DataFrame(data_observation_init)


# Sample data for continuum and lines
data_cont_init = {'Continuum Name': [],
             'x1': [],
             'x2': [],
             'Slope_0': [],
             'Intercept_0': [],
             'Slope_fit': [],
             'Intercept_fit': [],
             'region_ID': [],
             'lineactor:': []}

df_cont = pd.DataFrame(data_cont_init)

data_lines_init = {'Line_ID': [],
        'Line_Name': [],
        'SNR': [],
        'Rest Wavelength':[],
        'Amp_0': [],
        'Amp_0_lowlim': [],
        'Amp_0_highlim': [],
        'Centroid_0': [],
        'Centroid_0_lowlim': [],
        'Centroid_0_highlim': [],
        'Sigma_0': [],
        'Sigma_0_lowlim': [],
        'Sigma_0_highlim': [],
        'Constraints': np.array(5),
        'Amp_fit': [],
        'Centroid_fit': [],
        'Sigma_fit': [],
        'region_ID': [],
        'curveactor': []}

df = pd.DataFrame(data_lines_init)


df_fit = pd.DataFrame({})


def convert_velocity_to_centroid(velocity_constraint, line_name, df, galaxy_redshift):
    """
    Convert velocity-based constraints to centroid-based constraints.
    
    Parameters:
    - velocity_constraint (str): The velocity constraint (e.g., "vel == vel_[sii_1]")
    - line_name (str): The current line for which we're modifying the constraints
    - df (DataFrame): The DataFrame containing line information
    - galaxy_redshift (float): The galaxy's redshift (z)
    
    Returns:
    - str: A new centroid-based constraint.
    """
    # Extract the reference line from the constraint (e.g., "sii_1" from "vel == vel_[sii_1]")
    match = re.search(r"vel == vel_\[([\w\d_]+)\]", velocity_constraint)
    if not match:
        return velocity_constraint  # Return unchanged if no match
    
    ref_line_name = match.group(1)

    # Get rest wavelengths of the current and reference lines
    current_rest_wl = np.float64(df.loc[df['Line_Name'] == line_name, 'Rest Wavelength'].values[0])
    ref_rest_wl = np.float64(df.loc[df['Line_Name'] == ref_line_name, 'Rest Wavelength'].values[0])

    # Ensure the reference line exists
    if not ref_rest_wl:
        raise ValueError(f"Reference line {ref_line_name} not found in DataFrame.")

    # Convert velocity to centroid relation
    ref_cen_param = f"cen_[{ref_line_name}]"
    current_cen_param = "cen"

    # Centroid relation: λ_obs = λ_rest * (1 + z) * (1 + v/c)
    # Since v_ref == v_current, we derive:
    centroid_relation = f"{current_cen_param} == {ref_cen_param} * {current_rest_wl / ref_rest_wl:.6f}"

    return centroid_relation

def update_constraints_with_velocity(df, galaxy_redshift):
    """
    Update DataFrame constraints by converting velocity constraints to centroid constraints.
    
    Parameters:
    - df (DataFrame): Input DataFrame with Gaussian fitting information.
    - galaxy_redshift (float): The galaxy redshift.
    
    Returns:
    - DataFrame: Updated DataFrame with corrected centroid constraints.
    """
    updated_constraints = []

    for idx, row in df.iterrows():
        line_name = row['Line_Name']
        rest_wavelength = np.float64(row['Rest Wavelength'])
        constraints = row['constraints']
        
        wl_emitted_redshift = rest_wavelength * (1 + galaxy_redshift)
        
        
        # If 'vel' appears in constraints, process it
        if any('vel' in c for c in constraints):
            new_constraints = [
                convert_velocity_to_centroid(c, line_name, df, galaxy_redshift) if 'vel' in c else c
                for c in constraints
            ]
        else:
            new_constraints = constraints

        print(f'centroid constraint converted from velocity: {new_constraints}')
        updated_constraints.append(new_constraints)

    # Update the DataFrame
    df['constraints'] = updated_constraints
    return df


def add_dataframe_constraints_to_params(df, params):
    """
    Adds constraints to lmfit Parameters object, using:
    - Additive offsets for centroid inequalities (cen1 >= cen2 → cen1 = cen2 + offset)
    - Multiplicative ratios for all other inequalities (amp1 >= amp2 → amp1 = amp2 * ratio)
    - Direct expressions for equality constraints
    """
    if 'constraints' not in df.columns:
        print("Warning: 'constraints' column not found in DataFrame.")
        return

    for index, row in df.iterrows():
        line_id = int(row['Line_ID'])
        line_name = row['Line_Name']
        constraints_list_str = row['constraints']

        try:
            # Parse constraints string into list
            if isinstance(constraints_list_str, str):
                constraints_list = ast.literal_eval(constraints_list_str)
            elif isinstance(constraints_list_str, list):
                constraints_list = constraints_list_str
            else:
                constraints_list = ['', '', '', '', '']

            if not isinstance(constraints_list, list) or len(constraints_list) != 5:
                print(f"Warning: Invalid constraints format for Line_ID {line_id} ({line_name}): {constraints_list_str}")
                continue

            for constraint in constraints_list:
                constraint = constraint.strip()
                if not constraint:
                    continue

                # Split into left and right parts of the main operator
                operators = ['<=', '<', '>=', '>', '==']
                found_op = None
                for op in operators:
                    if op in constraint:
                        parts = constraint.split(op, 1)
                        found_op = op
                        break

                if not found_op:
                    print(f"Warning: Unsupported constraint format (no valid operator): {constraint}")
                    continue

                left_side = parts[0].strip()
                right_side = parts[1].strip()

                # Parse left side parameter
                param1_base = ''.join(filter(str.isalpha, left_side)).lower()
                param1_num = line_id + 1
                param1_name = f"{param1_base}{param1_num}"

                if param1_name not in params:
                    print(f"Warning: Parameter not found in params: {param1_name}")
                    continue

                # SPECIAL CASE: Centroid inequalities use additive offsets
                if param1_base == 'cen' and found_op in ['>=', '>', '<=', '<']:
                    if '_[' in right_side and ']' in right_side:
                        # Reference to another line's centroid
                        param2_base = ''.join(filter(str.isalpha, right_side.split('_[')[0])).lower()
                        other_line_name = right_side.split('_[')[1].split(']')[0]
                        
                        try:
                            other_row = df[df['Line_Name'] == other_line_name].iloc[0]
                            other_line_id = int(other_row['Line_ID'])
                            param2_num = other_line_id + 1
                            param2_name = f"{param2_base}{param2_num}"
                            
                            if param2_name not in params:
                                print(f"Warning: Parameter not found in params: {param2_name}")
                                continue

                            # Get current centroid values
                            cen1_init = params[param1_name].value
                            cen2_init = params[param2_name].value
                            
                            # Calculate initial offset value (ensure it's positive)
                            initial_offset = max(0.1, abs(cen1_init - cen2_init))
                            
                            # Determine max offset based on spectral range constraints
                            if found_op in ['>=', '>']:
                                # cen1 >= cen2 → cen1 = cen2 + offset
                                # Max offset should keep cen1 within x1_end
                                if 'x1_end' in params:
                                    max_offset = max(initial_offset, params['x1_end'].value - cen2_init)
                                else:
                                    max_offset = 100.0  # fallback
                                
                                offset_name = f"offset_{param2_name}_{param1_name}"
                                if offset_name not in params:
                                    params.add(
                                        offset_name,
                                        value=initial_offset,
                                        min=0.0,
                                        max=max_offset,
                                        vary=True
                                    )
                                params[param1_name].expr = f"{param2_name} + {offset_name}"
                                print(f"Centroid constraint: {param1_name} >= {param2_name} "
                                      f"(offset={initial_offset:.1f}, max={max_offset:.1f})")
                            
                            elif found_op in ['<=', '<']:
                                # cen1 <= cen2 → cen1 = cen2 - offset
                                # Max offset should keep cen1 within x1_start
                                if 'x1_start' in params:
                                    max_offset = max(initial_offset, cen2_init - params['x1_start'].value)
                                else:
                                    max_offset = 100.0  # fallback
                                
                                offset_name = f"offset_{param1_name}_{param2_name}"
                                if offset_name not in params:
                                    params.add(
                                        offset_name,
                                        value=initial_offset,
                                        min=0.0,
                                        max=max_offset,
                                        vary=True
                                    )
                                params[param1_name].expr = f"{param2_name} - {offset_name}"
                                print(f"Centroid constraint: {param1_name} <= {param2_name} "
                                      f"(offset={initial_offset:.1f}, max={max_offset:.1f})")
                        
                        except (IndexError, KeyError) as e:
                            print(f"Warning: Could not find line '{other_line_name}' for constraint: {constraint} - {e}")
                    else:
                        print(f"Warning: Centroid constraint must reference another line: {constraint}")
                    continue

                # DEFAULT CASE: Non-centroid parameters use ratio method
                # Parse right side (could be simple parameter or complex expression)
                if '_[' in right_side and ']' in right_side:
                    # Case 1: Reference to another line parameter (e.g., amp_[Halpha_c1])
                    param2_base = ''.join(filter(str.isalpha, right_side.split('_[')[0])).lower()
                    other_line_name = right_side.split('_[')[1].split(']')[0]
                    
                    try:
                        other_row = df[df['Line_Name'] == other_line_name].iloc[0]
                        other_line_id = int(other_row['Line_ID'])
                        param2_num = other_line_id + 1
                        param2_name = f"{param2_base}{param2_num}"
                        
                        if param2_name not in params:
                            print(f"Warning: Parameter not found in params: {param2_name}")
                            continue
                            
                        # Handle multiplicative factors
                        if '*' in right_side:
                            left_part, right_part = [x.strip() for x in right_side.split('*')]
                        
                            # Check which part is numeric and which is the parameter
                            if left_part.replace('.', '', 1).isdigit():
                                factor = float(left_part)
                                expr = f"{factor} * {param2_name}"
                            elif right_part.replace('.', '', 1).isdigit():
                                factor = float(right_part)
                                expr = f"{factor} * {param2_name}"
                            else:
                                print(f"Warning: Could not parse factor in constraint: {constraint}")
                                continue
                        else:
                            expr = param2_name
                            
                        # Apply the constraint
                        if found_op in ['<=', '<']:
                            # For A <= B, we express A = factor * B where factor <= 1
                            factor_name = f"ratio_{param1_name}_{param2_name}"
                            if factor_name not in params:
                                params.add(factor_name, value=0.9, min=0.0, max=1.0, vary=True)
                            params[param1_name].expr = f"{factor_name} * {expr}"
                            print(f"Constraint added: {param1_name} <= {expr} (using factor {factor_name})")
                        elif found_op in ['>=', '>']:
                            # For A >= B, we express A = B / factor where factor <= 1
                            factor_name = f"ratio_{param2_name}_{param1_name}"
                            if factor_name not in params:
                                params.add(factor_name, value=0.9, min=0.0, max=1.0, vary=True)
                            params[param1_name].expr = f"{expr} / {factor_name}"
                            print(f"Constraint added: {param1_name} >= {expr} (using factor {factor_name})")
                        elif found_op == '==':
                            params[param1_name].expr = expr
                            print(f"Constraint added: {param1_name} == {expr}")
                            
                    except (IndexError, KeyError):
                        print(f"Warning: Could not find line '{other_line_name}' for constraint: {constraint}")
                
                elif any(op in right_side for op in ['+', '-', '*', '/']):
                    # Case 2: Mathematical expression (e.g., "0.1 * amp_[Halpha_c1]")
                    try:
                        # Find all parameter references in the expression
                        expr_parts = right_side.split()
                        parsed_expr = []
                        
                        for part in expr_parts:
                            if '_[' in part and ']' in part:
                                # It's a parameter reference
                                param_base = ''.join(filter(str.isalpha, part.split('_[')[0])).lower()
                                other_line_name = part.split('_[')[1].split(']')[0]
                                
                                other_row = df[df['Line_Name'] == other_line_name].iloc[0]
                                other_line_id = int(other_row['Line_ID'])
                                param_num = other_line_id + 1
                                param_name = f"{param_base}{param_num}"
                                
                                if param_name not in params:
                                    print(f"Warning: Parameter not found in params: {param_name}")
                                    raise ValueError
                                    
                                parsed_expr.append(param_name)
                            else:
                                parsed_expr.append(part)
                                
                        expr = ' '.join(parsed_expr)
                        
                        # Apply the constraint
                        if found_op in ['<=', '<']:
                            factor_name = f"ratio_{param1_name}_expr_{hash(expr)}"[:30]
                            if factor_name not in params:
                                params.add(factor_name, value=0.9, min=0.0, max=1.0, vary=True)
                            params[param1_name].expr = f"{factor_name} * ({expr})"
                            print(f"Constraint added: {param1_name} <= {expr} (using factor {factor_name})")
                        elif found_op in ['>=', '>']:
                            factor_name = f"ratio_expr_{hash(expr)}_{param1_name}"[:30]
                            if factor_name not in params:
                                params.add(factor_name, value=1.1, min=1.0, max=10.0, vary=True)
                            params[param1_name].expr = f"({expr}) / {factor_name}"
                            print(f"Constraint added: {param1_name} >= {expr} (using factor {factor_name})")
                        elif found_op == '==':
                            params[param1_name].expr = f"({expr})"
                            print(f"Constraint added: {param1_name} == {expr}")
                            
                    except (ValueError, IndexError, KeyError):
                        print(f"Warning: Could not parse complex expression in constraint: {constraint}")
                
                else:
                    print(f"Warning: Unsupported right side format in constraint: {constraint}")

        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse constraints string for Line_ID {line_id} ({line_name}): {constraints_list_str} - {e}")



def generalized_model(x, slope, intercept, *gaussian_params):
    """
    Generalized Gaussian model for N Gaussians.
    :param x: Wavelength data.
    :param slope: Continuum slope.
    :param intercept: Continuum intercept.
    :param gaussian_params: Model parameters for N Gaussians [Amp_0, Centroid_0, Sigma_0, ...]
    """
    model = slope * x + intercept  # Linear model for continuum
    num_gaussians = len(gaussian_params) // 3
    for i in range(num_gaussians):
        amp = gaussian_params[3 * i]
        centroid = gaussian_params[3 * i + 1]
        sigma = gaussian_params[3 * i + 2]
        model += amp * np.exp(-(x - centroid)**2 / (2 * sigma**2))

    return model

# def load_stylesheet(app, filename):
#     with open(filename, 'r') as f:
#         style = f.read()
#         app.setStyleSheet(style)

# Apply PyQt5-like styling to Matplotlib
def _fmt(value, sig=4):
    """Format a numeric value for button display.
    Uses scientific notation when the absolute value is < 0.001 or >= 1e6,
    otherwise uses up to `sig` significant figures.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v == 0:
        return "0"
    if abs(v) < 0.001 or abs(v) >= 1e5:
        return f"{v:.{sig-1}e}"
    return f"{v:.{sig}g}"



# ── Emission line library ──────────────────────────────────────────────────────
_LINE_LIBRARY = None  # loaded lazily

def _load_line_library():
    """Load LineLibrary.csv from the HyperCube directory.

    Expected columns (exact names):
        wavelength_AA  : float, wavelength in Angstroms (range 770.409 – 10938.086)
        ion            : str,   emission line name
        IP_eV          : float, ionization potential in eV
    """
    global _LINE_LIBRARY
    if _LINE_LIBRARY is not None:
        return _LINE_LIBRARY
    try:
        lib_path = resource_path('LineLibrary.csv')
        _LINE_LIBRARY = pd.read_csv(lib_path,
                                    dtype={'wavelength_AA': float, 'ion': str, 'IP_eV': float})
        print(f"Loaded line library: {len(_LINE_LIBRARY)} lines from {lib_path}")
    except Exception as e:
        print(f"Could not load LineLibrary.csv: {e}")
        _LINE_LIBRARY = pd.DataFrame(columns=['wavelength_AA', 'ion', 'IP_eV'])
    return _LINE_LIBRARY


def _identify_line(obs_wavelength_AA, redshift):
    """Given an observed wavelength and source redshift, identify the nearest line.

    Computes rest_wav = obs_wav / (1 + z), then finds the nearest entry in
    LineLibrary.csv by rest wavelength.
    Returns (ion_name, rest_wav_AA) or ('', nan) if redshift is missing/invalid.
    """
    try:
        z = float(redshift)
    except (TypeError, ValueError):
        return '', np.nan
    if z <= 0 or not np.isfinite(z):
        return '', np.nan

    lib = _load_line_library()
    if len(lib) == 0:
        return '', np.nan

    rest_wav = obs_wavelength_AA / (1.0 + z)
    wavs = lib['wavelength_AA'].values.astype(float)
    idx = np.argmin(np.abs(wavs - rest_wav))
    ion_name   = str(lib.iloc[idx]['ion'])
    nearest_wav = float(wavs[idx])
    return ion_name, nearest_wav


def apply_mpl_qss_style(fig, ax, line):
    """Apply a PyQt5-style QSS theme to the given Matplotlib figure, axes, and line."""
    
    # Set the figure (entire canvas) background color
    fig.patch.set_facecolor('#323232')  # Matches QWidget background
    
    # Axis background color
    ax.set_facecolor('#2E3440')  
    
    # Axis labels and title (matches QWidget text color)
    ax.xaxis.label.set_color('#b1b1b1')
    ax.yaxis.label.set_color('#b1b1b1')
    ax.title.set_color('#b1b1b1')
    
    # Ticks (color and size)
    ax.tick_params(axis='x', colors='#b1b1b1', labelsize=12)
    ax.tick_params(axis='y', colors='#b1b1b1', labelsize=12)
    
    # Axis borders (spines)
    for spine in ax.spines.values():
        spine.set_edgecolor('#b1b1b1')
        spine.set_linewidth(0.75)
    
    # Grid (optional – enable for a clearer view)
    ax.grid(color='#4C566A', linestyle='--', linewidth=0.5)
    
    if line != None:
        # Adjust the spectrum line styling
        line.set_color('whitesmoke')  # Hover color from QPushButton
        line.set_linewidth(0.75)  # Slightly thicker for better visibility
    
    # Legend styling (if needed)
    # legend = ax.legend()
    # if legend:
    #     legend.get_frame().set_facecolor('#2E3440')
    #     legend.get_frame().set_edgecolor('#D8DEE9')
    #     plt.setp(legend.get_texts(), color='#D8DEE9')

    # Tighten layout to ensure good spacing
    fig.tight_layout()

class SpaxelButton(QPushButton):
    def __init__(self, text, col, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.col = col

class FrameButton(QPushButton):
    def __init__(self, text, row, col, frame_id, feature, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.row = row
        self.col = col
        self.frame_id = frame_id  # Store frame information in the button
        self.feature = feature
        # Expand horizontally so buttons fill their grid column as the dock is resized
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )

    def minimumSizeHint(self):
        # Allow buttons to compress below their text width when the dock is narrow
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())


class ViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.fit_params_window = None  # Placeholder, initially None
        self.fit_params_dock = None   # QDockWidget wrapper, initially None
        self.fits_header = None
        self.fits_data = None
        self.is_1d_spectrum = False
        self.current_spaxel = None  # Stores the currently highlighted spaxel (x, y)
        self.locked = False  # Initially unlocked
        self._chanmap_active = False   # True while C is held
        # Brightness/contrast drag state
        self._bc_drag_active = False
        # Background image display settings
        self._bkg_cmap  = 'gray'
        self._bkg_scale = 'Linear'
        self._bkg_brightness = 0.0   # shift: fraction of data range
        self._bkg_contrast  = 1.0   # scale multiplier on span
        self._init_guess_spaxel = None   # (x, y) that has interactive init guesses
        self._blue_rect = None            # blue rectangle marking that spaxel
        self._bc_drag_start  = None   # (x, y) in axes data coords at press
        self._bc_vmin0 = None         # vmin at drag start
        self._bc_vmax0 = None         # vmax at drag start
        self._chanmap_start  = None    # wavelength where C was pressed
        self._chanmap_span   = None    # axvspan patch on spectrum_ax
        self._chanmap_locked = False   # True after C released (selection fixed)
        # Subtraction windows (X and V keys)
        self._submap = {'x': {'active': False, 'start': None, 'span': None, 'locked': False},
                        'v': {'active': False, 'start': None, 'span': None, 'locked': False}}
        self.spectrum_cursor_pos = None  # Add tracking for spectrum cursor position
        self.spectrum_ax = None
        self.drawing_line = None
        self.fluxscalefactor = 1 # Default flux scale factor
        self.WLscalefactor = 1 # Default wavelength scale factor
        self.slope = 0  # Default slope
        self.intercept = 0  # Default intercept
        self.gaussian_active = False  # Toggle for Gaussian fitting
        self.current_gaussian = None  # Store the Gaussian plot
        self.initial_x = None  # Stores initial cursor position for Gaussian
        self.current_sigma = 10  # Default sigma
        self.current_amplitude = 1  # Default amplitude
        self.line_region = None
        
        self.zoom_rect = None  # Store the rectangle
        self.zoom_start_x = None  # Start position for zoom
        self.zoom_active = False
        self.zoom_limits = None  # Store zoom limits globally
        self.zoom_pad = 0.2
        self.modelfit = None
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle("HyperCube")
        self.setGeometry(100, 100, 1200, 700)

        # Main layout
        main_layout = QVBoxLayout()

        # Horizontal splitter: cube image left (~30%), spectrum viewer right (~70%)
        self.splitter = QSplitter(Qt.Horizontal)

        # Left panel: cube image (square-ish)
        self.left_panel = QWidget(self)
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(0)

        # ── Zoom toolbar ───────────────────────────────────────────────
        zoom_bar = QHBoxLayout()
        zoom_bar.setContentsMargins(4, 2, 4, 2)
        zoom_bar.setSpacing(2)

        self._cube_zoom_level = 1.0  # 1.0 = 100% (fit window)
        self._zoom_levels = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32]

        self.zoom_out_btn = QPushButton("−", self)
        self.zoom_out_btn.setFixedSize(24, 24)
        self.zoom_out_btn.setToolTip("Zoom out")
        self.zoom_out_btn.clicked.connect(self._cube_zoom_out)
        zoom_bar.addWidget(self.zoom_out_btn)

        self.zoom_combo = QComboBox(self)
        self.zoom_combo.setFixedHeight(24)
        self.zoom_combo.setMinimumWidth(130)
        self.zoom_combo.setMaximumWidth(160)
        self.zoom_combo.addItems([
            "Fit window", "Fit width", "Fit height",
            "3.125%", "6.25%", "12.5%", "25%", "50%",
            "100%", "200%", "400%", "800%", "1600%", "3200%"
        ])
        self.zoom_combo.setCurrentText("100%")
        self.zoom_combo.setStyleSheet(
            "QComboBox { color: #eff0f1; background-color: #3c3f41; border: 1px solid #555; padding: 0 4px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView {"
            "    color: #eff0f1;"
            "    background-color: #3c3f41;"
            "    selection-background-color: #d7801a;"
            "    selection-color: #000000;"
            "    border: 1px solid #555;"
            "}"
        )
        self.zoom_combo.currentTextChanged.connect(self._cube_zoom_combo_changed)
        zoom_bar.addWidget(self.zoom_combo)

        self.zoom_in_btn = QPushButton("+", self)
        self.zoom_in_btn.setFixedSize(24, 24)
        self.zoom_in_btn.setToolTip("Zoom in")
        self.zoom_in_btn.clicked.connect(self._cube_zoom_in)
        zoom_bar.addWidget(self.zoom_in_btn)

        # Thin separator between zoom and image scaling controls
        _sep1 = QFrame(self); _sep1.setFrameShape(QFrame.VLine); _sep1.setFrameShadow(QFrame.Sunken)
        zoom_bar.addWidget(_sep1)

        # Stretch combo (clip level)
        _combo_qss = (
            "QComboBox { color: #eff0f1; background-color: #3c3f41; border: 1px solid #555; padding: 0 4px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView {"
            "    color: #eff0f1; background-color: #3c3f41;"
            "    selection-background-color: #d7801a; selection-color: #000000;"
            "    border: 1px solid #555;}"
        )
        self.stretch_combo = QComboBox(self)
        self.stretch_combo.setFixedHeight(24)
        self.stretch_combo.setMinimumWidth(75)
        self.stretch_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.stretch_combo.addItems(["minmax", "99.9%", "99.5%", "99%", "98%", "95%", "manual"])
        self.stretch_combo.setCurrentText("99%")
        self.stretch_combo.setToolTip("Image stretch (clip level)")
        self.stretch_combo.setStyleSheet(_combo_qss)
        self.stretch_combo.currentTextChanged.connect(self._cube_redraw_with_current_settings)
        zoom_bar.addWidget(self.stretch_combo)

        # Scale combo (transfer function)
        self.scale_combo = QComboBox(self)
        self.scale_combo.setFixedHeight(24)
        self.scale_combo.setMinimumWidth(75)
        self.scale_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.scale_combo.addItems(["Linear", "Log", "Square root", "Squared", "Asinh"])
        self.scale_combo.setCurrentText("Linear")
        self.scale_combo.setToolTip("Image scaling (transfer function)")
        self.scale_combo.setStyleSheet(_combo_qss)
        self.scale_combo.currentTextChanged.connect(self._cube_redraw_with_current_settings)
        zoom_bar.addWidget(self.scale_combo)

        zoom_bar.addStretch()

        # Thin separator before reset button
        _sep_reset = QFrame(self)
        _sep_reset.setFrameShape(QFrame.VLine)
        _sep_reset.setFrameShadow(QFrame.Sunken)
        zoom_bar.addWidget(_sep_reset)

        # Reset view button
        self.reset_view_btn = QPushButton("↺  Reset", self)
        self.reset_view_btn.setFixedHeight(24)
        self.reset_view_btn.setToolTip(
            "Reset to original view: B/W colormap, default zoom and scaling"
        )
        self.reset_view_btn.clicked.connect(self._cube_reset_view)
        zoom_bar.addWidget(self.reset_view_btn)

        # (rotate/flip/coords buttons are in the right-side panel of the canvas grid)
        self._cube_rotation = 0
        self._cube_flip_h = False
        self._cube_flip_v = False
        self._cube_coords_mode = "xy"

        # Cube colormap selector
        _sep_cmap = QFrame(self); _sep_cmap.setFrameShape(QFrame.VLine); _sep_cmap.setFrameShadow(QFrame.Sunken)
        zoom_bar.addWidget(_sep_cmap)

        self.cube_cmap_combo = QComboBox(self)
        self.cube_cmap_combo.setFixedHeight(24)
        self.cube_cmap_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.cube_cmap_combo.addItems(['gray','gray_r','viridis','plasma','inferno','magma','hot','cool','Blues','Reds','Greens','bwr','seismic'])
        self.cube_cmap_combo.setCurrentText('gray')
        self.cube_cmap_combo.setToolTip("Cube image colormap")
        self.cube_cmap_combo.setStyleSheet(
            "QComboBox { color: #eff0f1; background-color: #3c3f41; border: 1px solid #555; padding: 0 4px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { color: #eff0f1; background-color: #3c3f41;"
            "  selection-background-color: #d7801a; selection-color: #000; border: 1px solid #555; }"
        )
        self.cube_cmap_combo.currentTextChanged.connect(
            lambda cmap: [setattr(self, '_last_cmap', cmap), self._cube_redraw_with_current_settings()]
        )
        zoom_bar.addWidget(self.cube_cmap_combo)

        # Thin separator before background image controls
        _sep_bkg = QFrame(self); _sep_bkg.setFrameShape(QFrame.VLine); _sep_bkg.setFrameShadow(QFrame.Sunken)
        zoom_bar.addWidget(_sep_bkg)

        self.bkg_image_btn = QPushButton("🌌 Bkg Image", self)
        self.bkg_image_btn.setFixedHeight(24)
        self.bkg_image_btn.setToolTip("Overlay HST background image (requires resolved source)")
        self.bkg_image_btn.setEnabled(False)
        self.bkg_image_btn.clicked.connect(self._show_bkg_image_dialog)
        zoom_bar.addWidget(self.bkg_image_btn)

        # Opacity slider for the cube overlay
        self.cube_opacity_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.cube_opacity_slider.setRange(0, 100)
        self.cube_opacity_slider.setValue(100)
        self.cube_opacity_slider.setFixedWidth(80)
        self.cube_opacity_slider.setFixedHeight(20)
        self.cube_opacity_slider.setToolTip("Cube overlay opacity")
        self.cube_opacity_slider.setVisible(False)  # shown after bkg image loaded
        self.cube_opacity_slider.valueChanged.connect(self._cube_opacity_changed)
        zoom_bar.addWidget(self.cube_opacity_slider)

        zoom_bar_widget = QWidget()
        zoom_bar_widget.setLayout(zoom_bar)
        zoom_bar_widget.setFixedHeight(30)
        self.left_layout.addWidget(zoom_bar_widget)
        # ───────────────────────────────────────────────────────────────

        self.canvas = FigureCanvas(plt.Figure())  # Matplotlib figure canvas
        self._init_placeholder_canvas()  # Dark themed placeholder before FITS loaded

        # Canvas + manual scrollbars in a grid layout
        # QScrollArea with setWidgetResizable=True never overflows, so we
        # use standalone QScrollBars wired directly to the axes pan logic.
        canvas_grid = QWidget()
        canvas_grid_layout = QGridLayout(canvas_grid)
        canvas_grid_layout.setContentsMargins(0, 0, 0, 0)
        canvas_grid_layout.setSpacing(0)

        self.cube_hbar = QScrollBar(Qt.Horizontal)
        self.cube_vbar = QScrollBar(Qt.Vertical)
        self.cube_hbar.setRange(0, 1000)
        self.cube_vbar.setRange(0, 1000)
        self.cube_hbar.setValue(500)
        self.cube_vbar.setValue(500)
        self.cube_hbar.setSingleStep(20)
        self.cube_vbar.setSingleStep(20)
        self.cube_hbar.setPageStep(100)
        self.cube_vbar.setPageStep(100)
        # Thicker, easier-to-grab scrollbars; hidden until zoomed in
        _bar_qss = (
            "QScrollBar:horizontal { height: 14px; }" 
            "QScrollBar:vertical   { width:  14px; }"
            "QScrollBar::handle:horizontal { min-width: 30px; }"
            "QScrollBar::handle:vertical   { min-height: 30px; }"
        )
        self.cube_hbar.setStyleSheet(_bar_qss)
        self.cube_vbar.setStyleSheet(_bar_qss)
        self.cube_hbar.hide()
        self.cube_vbar.hide()
        self._cube_scrollbar_updating = False  # guard against feedback loops
        self.cube_hbar.valueChanged.connect(self._cube_scrollbar_pan)
        self.cube_vbar.valueChanged.connect(self._cube_scrollbar_pan)

        # Side button column: rotate, flip, coords toggle
        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(2, 4, 2, 4)
        side_layout.setSpacing(4)
        side_layout.setAlignment(Qt.AlignTop)

        _side_btn_defs = [
            ("⟳", "Rotate 90° clockwise (disabled)",        lambda: self._cube_rotate(90)),
            ("⟲", "Rotate 90° counter-clockwise (disabled)", lambda: self._cube_rotate(-90)),
            ("⇔", "Flip horizontal (disabled)",              lambda: self._cube_flip('h')),
            ("⇕", "Flip vertical (disabled)",                lambda: self._cube_flip('v')),
        ]
        for label, tip, fn in _side_btn_defs:
            b = QPushButton(label, self)
            b.setFixedSize(28, 28)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            b.setEnabled(False)
            side_layout.addWidget(b)

        # Thin horizontal separator
        _hsep = QFrame(); _hsep.setFrameShape(QFrame.HLine); _hsep.setFrameShadow(QFrame.Sunken)
        side_layout.addWidget(_hsep)

        self.coords_toggle_btn = QPushButton("X/Y", self)
        self.coords_toggle_btn.setFixedSize(40, 28)
        self.coords_toggle_btn.setToolTip("Toggle between X/Y pixel and RA/Dec axis labels")
        self.coords_toggle_btn.clicked.connect(self._cube_toggle_coords)
        side_layout.addWidget(self.coords_toggle_btn)

        canvas_grid_layout.addWidget(self.canvas,    0, 0)
        canvas_grid_layout.addWidget(side_panel,     0, 1)
        canvas_grid_layout.addWidget(self.cube_vbar, 0, 2)
        canvas_grid_layout.addWidget(self.cube_hbar, 1, 0)
        canvas_grid_layout.setColumnStretch(0, 1)
        canvas_grid_layout.setRowStretch(0, 1)
        self.left_layout.addWidget(canvas_grid)
        self.left_panel.setLayout(self.left_layout)
        self.splitter.addWidget(self.left_panel)

        # Right panel: spectrum viewer
        self.right_panel = QWidget(self)
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(0)

        # ── Channel map / spectrum toolbar (hidden until spectrum is loaded) ──
        spec_bar = QHBoxLayout()
        spec_bar.setContentsMargins(4, 2, 4, 2)
        spec_bar.setSpacing(4)

        # Navigation: shift selection box left/right
        self.chanmap_prev_btn = QPushButton("−", self)
        self.chanmap_prev_btn.setFixedSize(24, 24)
        self.chanmap_prev_btn.setToolTip("Shift channel map selection left by 1 pixel")
        self.chanmap_prev_btn.clicked.connect(lambda: self._chanmap_shift(-1))
        spec_bar.addWidget(self.chanmap_prev_btn)

        # Centre pixel: button that opens a dialog on click (no keyboard focus stealing)
        self.chanmap_centre_btn = SpaxelButton("Pixel: —", "Channel centre")
        self.chanmap_centre_btn.setFixedHeight(24)
        self.chanmap_centre_btn.setMinimumWidth(90)
        self.chanmap_centre_btn.setToolTip("Centre pixel of channel map selection — click to edit")
        self.chanmap_centre_btn.clicked.connect(self._chanmap_centre_clicked)
        spec_bar.addWidget(self.chanmap_centre_btn)

        self.chanmap_next_btn = QPushButton("+", self)
        self.chanmap_next_btn.setFixedSize(24, 24)
        self.chanmap_next_btn.setToolTip("Shift channel map selection right by 1 pixel")
        self.chanmap_next_btn.clicked.connect(lambda: self._chanmap_shift(+1))
        spec_bar.addWidget(self.chanmap_next_btn)

        spec_bar.addStretch()
        self.spec_bar_widget = QWidget()
        self.spec_bar_widget.setLayout(spec_bar)
        self.spec_bar_widget.setFixedHeight(30)
        self.spec_bar_widget.hide()  # shown when spectrum canvas first appears
        self.right_layout.addWidget(self.spec_bar_widget)
        # ───────────────────────────────────────────────────────────────

        self.right_panel.setLayout(self.right_layout)
        self.splitter.addWidget(self.right_panel)

        # Equal 50/50 split by default; user can drag the splitter to adjust
        self.splitter.setSizes([600, 600])
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self.splitter)

        # ── Toolbar row: sits between the splitter panels and the window edge ──
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(6, 4, 6, 4)
        toolbar.setSpacing(8)

        # Left group: panel toggle + data-dependent tools
        self.open_fit_params_button = QPushButton("▼  Fit Parameters", self)
        self.open_fit_params_button.setFixedHeight(32)
        self.open_fit_params_button.setToolTip("Toggle Fit Parameters panel")
        self.open_fit_params_button.clicked.connect(self.toggle_fit_params_dock)
        toolbar.addWidget(self.open_fit_params_button)

        # Thin separator
        sep = QFrame(self)
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        toolbar.addWidget(sep)

        self.WLscalefactor_button = QPushButton("λ Scale", self)
        self.WLscalefactor_button.setFixedHeight(32)
        self.WLscalefactor_button.setToolTip("Wavelength scale factor")
        self.WLscalefactor_button.setEnabled(False)  # enabled after FITS load
        self.WLscalefactor_button.clicked.connect(self.press_scaleWL_button)
        toolbar.addWidget(self.WLscalefactor_button)

        self.fluxscalefactor_button = QPushButton("F Scale", self)
        self.fluxscalefactor_button.setFixedHeight(32)
        self.fluxscalefactor_button.setToolTip("Flux scale factor")
        self.fluxscalefactor_button.setEnabled(False)  # enabled after FITS load
        self.fluxscalefactor_button.clicked.connect(self.press_scaleflux_button)
        toolbar.addWidget(self.fluxscalefactor_button)

        toolbar.addStretch()

        # Right: prominent Open FITS button
        self.open_fits_button = QPushButton("  ⬆  Open FITS File", self)
        self.open_fits_button.setFixedHeight(36)
        self.open_fits_button.setStyleSheet(
            "QPushButton { background-color: #d7801a; color: white; font-weight: bold;"
            " border-radius: 6px; padding: 0 16px; }"
            "QPushButton:hover { background-color: #ffa02f; }"
        )
        self.open_fits_button.clicked.connect(self.open_fits_file)
        toolbar.addWidget(self.open_fits_button)

        main_layout.addLayout(toolbar)

        # Set up central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Set menu bar
        self.setMenuBar(self.create_menu_bar_VisualizerWindow())

        # Handle and separator styles are defined in QDarkOrange_style.qss

        # Ensure the window comes to the front
        # self.show()
        self.raise_()
        self.activateWindow()

        # Enable mouse tracking for the white-light image
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('scroll_event', self._cube_on_scroll)
        self.canvas.mpl_connect('button_press_event', self._cube_pan_start)
        self.canvas.mpl_connect('button_press_event', self._cube_bc_press)
        self.canvas.mpl_connect('motion_notify_event', self._cube_pan_move)
        self.canvas.mpl_connect('motion_notify_event', self._cube_bc_drag)
        self.canvas.mpl_connect('button_release_event', self._cube_pan_end)
        self.canvas.mpl_connect('button_release_event', self._cube_bc_release)
        self._cube_pan_active = False
        self._cube_pan_start_pos = None
        self._cube_pan_start_xlim = None
        self._cube_pan_start_ylim = None

        self.cursor_pos = None

    # ── Cube viewport zoom ────────────────────────────────────────────────────


    def _cube_scrollbar_pan(self):
        """Pan the cube axes from scrollbar position (0-1000 range)."""
        if self._cube_scrollbar_updating:
            return
        if not hasattr(self, 'ax') or self.ax is None:
            return
        imgs = [c for c in self.ax.get_children()
                if hasattr(c, 'get_extent') and callable(c.get_extent)]
        if not imgs:
            return

        x0, x1, y0, y1 = imgs[0].get_extent()
        img_w = abs(x1 - x0)
        img_h = abs(y1 - y0)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        view_w = abs(xlim[1] - xlim[0])
        view_h = abs(ylim[1] - ylim[0])

        pan_w = max(img_w - view_w, 0)
        pan_h = max(img_h - view_h, 0)

        frac_h = self.cube_hbar.value() / 1000.0
        frac_v = 1.0 - self.cube_vbar.value() / 1000.0  # invert: 0=top

        cx = x0 + view_w / 2 + frac_h * pan_w
        cy = y0 + view_h / 2 + frac_v * pan_h

        self.ax.set_xlim(cx - view_w / 2, cx + view_w / 2)
        self.ax.set_ylim(cy - view_h / 2, cy + view_h / 2)
        self.canvas.draw_idle()

    def _cube_update_scrollbars(self):
        """Sync scrollbar positions to current axes limits and show/hide them.
        
        Bars are only visible when the view is smaller than the image
        (i.e. the user is zoomed in enough that parts of the image are hidden).
        """
        if not hasattr(self, 'ax') or self.ax is None:
            return
        if not hasattr(self, 'cube_hbar'):
            return
        imgs = [c for c in self.ax.get_children()
                if hasattr(c, 'get_extent') and callable(c.get_extent)]
        if not imgs:
            return

        x0, x1, y0, y1 = imgs[0].get_extent()
        img_w = abs(x1 - x0)
        img_h = abs(y1 - y0)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        view_w = abs(xlim[1] - xlim[0])
        view_h = abs(ylim[1] - ylim[0])

        pan_w = img_w - view_w
        pan_h = img_h - view_h

        self._cube_scrollbar_updating = True

        # Horizontal bar
        if pan_w > 1e-6:
            cx = (xlim[0] + xlim[1]) / 2
            frac_h = np.clip((cx - (x0 + view_w / 2)) / pan_w, 0, 1)
            self.cube_hbar.setValue(int(frac_h * 1000))
            self.cube_hbar.show()
        else:
            self.cube_hbar.setValue(500)
            self.cube_hbar.hide()

        # Vertical bar
        if pan_h > 1e-6:
            cy = (ylim[0] + ylim[1]) / 2
            frac_v = np.clip((cy - (y0 + view_h / 2)) / pan_h, 0, 1)
            self.cube_vbar.setValue(int((1 - frac_v) * 1000))
            self.cube_vbar.show()
        else:
            self.cube_vbar.setValue(500)
            self.cube_vbar.hide()

        self._cube_scrollbar_updating = False

    def _cube_bc_press(self, event):
        """Start brightness/contrast drag on left-click (button 1) in the cube axes."""
        if event.button != 1 or event.inaxes != self.ax:
            return
        bkg = getattr(self, '_bkg_artist', None)
        # Get clim from the cube image only (skip the background artist)
        imgs = [c for c in self.ax.get_children()
                if hasattr(c, 'get_clim') and callable(c.get_clim) and c is not bkg]
        if not imgs:
            return
        self._bc_vmin0, self._bc_vmax0 = imgs[0].get_clim()
        self._bc_drag_start  = (event.x, event.y)  # pixels, not data coords
        self._bc_drag_active = True

    def _cube_bc_drag(self, event):
        """Adjust brightness (left/right) and contrast (up/down) during drag."""
        if not self._bc_drag_active or event.inaxes != self.ax:
            return
        if event.x is None or event.y is None:
            return

        dx = event.x - self._bc_drag_start[0]   # pixels rightward
        dy = event.y - self._bc_drag_start[1]   # pixels upward

        span = self._bc_vmax0 - self._bc_vmin0
        if span == 0:
            span = 1.0

        # Sensitivity: 300 px of drag = 1 full span shift/scale
        # Up/down = brightness (shift midpoint); left/right = contrast (scale span)
        brightness_shift = (dy / 300.0) * span   # up = brighter
        contrast_scale   = 10 ** (dx / 300.0)    # right = wider range (less contrast)

        mid = (self._bc_vmin0 + self._bc_vmax0) / 2 + brightness_shift
        half = (span / 2) * contrast_scale

        bkg = getattr(self, '_bkg_artist', None)
        # Only adjust clim on the cube image — leave the background untouched
        imgs = [c for c in self.ax.get_children()
                if hasattr(c, 'set_clim') and callable(c.set_clim) and c is not bkg]
        for img in imgs:
            img.set_clim(mid - half, mid + half)
        self.canvas.draw_idle()

    def _cube_bc_release(self, event):
        """End brightness/contrast drag."""
        if event.button == 1:
            self._bc_drag_active = False

    def _cube_clamp_limits(self, xlim, ylim):
        """Clamp xlim/ylim so the view never pans outside the image bounds."""
        if not hasattr(self, 'ax') or self.ax is None:
            return xlim, ylim
        imgs = [c for c in self.ax.get_children()
                if hasattr(c, 'get_extent') and callable(c.get_extent)]
        if not imgs:
            return xlim, ylim
        x0, x1, y0, y1 = imgs[0].get_extent()
        half_w = (xlim[1] - xlim[0]) / 2
        half_h = (ylim[1] - ylim[0]) / 2
        cx = np.clip((xlim[0] + xlim[1]) / 2, x0 + half_w, x1 - half_w)
        cy = np.clip((ylim[0] + ylim[1]) / 2, y0 + half_h, y1 - half_h)
        return (cx - half_w, cx + half_w), (cy - half_h, cy + half_h)

    def _cube_on_scroll(self, event):
        """Pan the cube image on trackpad/mouse scroll.
        
        Vertical scroll (step) pans up/down.
        Horizontal scroll (guiEvent with angleDeltaX) pans left/right.
        Hold Ctrl to pan horizontally on a one-axis scroll device.
        """
        if not hasattr(self, 'ax') or self.ax is None:
            return
        if event.inaxes != self.ax:
            return

        xlim = list(self.ax.get_xlim())
        ylim = list(self.ax.get_ylim())
        pan_frac = 0.1  # 10% of current view per scroll step

        dx = (xlim[1] - xlim[0]) * pan_frac
        dy = (ylim[1] - ylim[0]) * pan_frac

        # Dominant-axis locking: whichever axis has the larger delta wins,
        # preventing diagonal drift on trackpads.
        vert_raw = 0
        horiz_raw = 0
        if hasattr(event, 'guiEvent') and event.guiEvent is not None:
            ge = event.guiEvent
            if hasattr(ge, 'angleDelta'):
                horiz_raw = ge.angleDelta().x()
                vert_raw  = ge.angleDelta().y()

        if abs(horiz_raw) > abs(vert_raw) and horiz_raw != 0:
            # Pure horizontal gesture → pan left/right only
            direction = -1 if horiz_raw > 0 else 1
            xlim[0] += direction * dx
            xlim[1] += direction * dx
        else:
            # Vertical scroll (wheel or dominant vertical gesture) → pan up/down
            if vert_raw != 0:
                direction = 1 if vert_raw > 0 else -1
            else:
                direction = 1 if event.button == 'up' else -1
            ylim[0] += direction * dy
            ylim[1] += direction * dy

        xlim, ylim = self._cube_clamp_limits(xlim, ylim)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()
        self._cube_update_scrollbars()

    def _cube_pan_start(self, event):
        """Begin middle-mouse-button drag pan."""
        if event.button == 2 and event.inaxes == self.ax:
            self._cube_pan_active = True
            self._cube_pan_start_pos = (event.xdata, event.ydata)
            self._cube_pan_start_xlim = list(self.ax.get_xlim())
            self._cube_pan_start_ylim = list(self.ax.get_ylim())

    def _cube_pan_move(self, event):
        """Pan while middle-mouse button is held."""
        if not self._cube_pan_active or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = self._cube_pan_start_pos[0] - event.xdata
        dy = self._cube_pan_start_pos[1] - event.ydata
        xlim = [self._cube_pan_start_xlim[0] + dx,
                self._cube_pan_start_xlim[1] + dx]
        ylim = [self._cube_pan_start_ylim[0] + dy,
                self._cube_pan_start_ylim[1] + dy]
        xlim, ylim = self._cube_clamp_limits(xlim, ylim)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()
        self._cube_update_scrollbars()

    def _cube_pan_end(self, event):
        """End middle-mouse-button drag pan."""
        if event.button == 2:
            self._cube_pan_active = False
            self._cube_pan_start_pos = None

    def _cube_zoom_apply(self, zoom_level=None, mode=None):
        """Apply zoom to the cube axes.
        
        zoom_level: float multiplier (1.0 = fit window, 2.0 = 200%, etc.)
        mode: 'fit_window' | 'fit_width' | 'fit_height' | None (use zoom_level)
        """
        if not hasattr(self, 'ax') or self.ax is None:
            return

        # Natural extent of the image in data coords
        imgs = [c for c in self.ax.get_children()
                if hasattr(c, 'get_extent') and callable(c.get_extent)]
        if not imgs:
            return
        x0, x1, y0, y1 = imgs[0].get_extent()  # (left, right, bottom, top)
        img_w = abs(x1 - x0)
        img_h = abs(y1 - y0)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        canvas_w = self.canvas.width()
        canvas_h = self.canvas.height()
        if canvas_w == 0 or canvas_h == 0:
            return

        if mode == 'fit_window':
            self.ax.set_xlim(x0, x1)
            self.ax.set_ylim(y0, y1)
            self._cube_zoom_level = 1.0
        elif mode == 'fit_width':
            # Scale so image width fills canvas; crop/pad height
            scale = img_w / img_w  # width always fits
            half_h = (img_w / canvas_w * canvas_h) / 2
            self.ax.set_xlim(x0, x1)
            self.ax.set_ylim(cy - half_h, cy + half_h)
            self._cube_zoom_level = 1.0
        elif mode == 'fit_height':
            half_w = (img_h / canvas_h * canvas_w) / 2
            self.ax.set_xlim(cx - half_w, cx + half_w)
            self.ax.set_ylim(y0, y1)
            self._cube_zoom_level = 1.0
        else:
            # Percentage zoom: zoom_level=1 shows the full image,
            # zoom_level=2 shows half the image (2× magnification), etc.
            half_w = img_w / (2 * zoom_level)
            half_h = img_h / (2 * zoom_level)
            # Keep centre fixed
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            cur_cx = (cur_xlim[0] + cur_xlim[1]) / 2
            cur_cy = (cur_ylim[0] + cur_ylim[1]) / 2
            # Clamp centre so we don't pan outside the image
            cur_cx = np.clip(cur_cx, x0 + half_w, x1 - half_w)
            cur_cy = np.clip(cur_cy, y0 + half_h, y1 - half_h)
            self.ax.set_xlim(cur_cx - half_w, cur_cx + half_w)
            self.ax.set_ylim(cur_cy - half_h, cur_cy + half_h)
            self._cube_zoom_level = zoom_level

        self.canvas.draw_idle()
        self._cube_update_scrollbars()

    def _cube_zoom_in(self):
        lvls = self._zoom_levels
        idx = min(range(len(lvls)), key=lambda i: abs(lvls[i] - self._cube_zoom_level))
        new_idx = min(idx + 1, len(lvls) - 1)
        self._cube_zoom_level = lvls[new_idx]
        self._sync_zoom_combo()
        self._cube_zoom_apply(zoom_level=self._cube_zoom_level)

    def _cube_zoom_out(self):
        lvls = self._zoom_levels
        idx = min(range(len(lvls)), key=lambda i: abs(lvls[i] - self._cube_zoom_level))
        new_idx = max(idx - 1, 0)
        self._cube_zoom_level = lvls[new_idx]
        self._sync_zoom_combo()
        self._cube_zoom_apply(zoom_level=self._cube_zoom_level)

    def _sync_zoom_combo(self):
        """Update the combo text to reflect current zoom level without triggering signal."""
        pct_map = {
            1/32: "3.125%", 1/16: "6.25%", 1/8: "12.5%", 1/4: "25%",
            1/2: "50%", 1: "100%", 2: "200%", 4: "400%",
            8: "800%", 16: "1600%", 32: "3200%"
        }
        label = pct_map.get(self._cube_zoom_level, f"{int(self._cube_zoom_level*100)}%")
        self.zoom_combo.blockSignals(True)
        self.zoom_combo.setCurrentText(label)
        self.zoom_combo.blockSignals(False)

    def _cube_zoom_combo_changed(self, text):
        mode_map = {
            "Fit window": ("fit_window", 1.0),
            "Fit width":  ("fit_width",  1.0),
            "Fit height": ("fit_height", 1.0),
        }
        pct_map = {
            "3.125%": 1/32, "6.25%": 1/16, "12.5%": 1/8, "25%": 1/4,
            "50%": 1/2, "100%": 1, "200%": 2, "400%": 4,
            "800%": 8, "1600%": 16, "3200%": 32
        }
        if text in mode_map:
            mode, lvl = mode_map[text]
            self._cube_zoom_level = lvl
            self._cube_zoom_apply(mode=mode)
        elif text in pct_map:
            lvl = pct_map[text]
            self._cube_zoom_level = lvl
            self._cube_zoom_apply(zoom_level=lvl)

    # ── Cube image transform (rotate/flip) ───────────────────────────────────

    def _cube_get_transformed_data(self):
        """Apply current rotation and flip to _last_data and return the result.

        Verified empirically:
          k=1 → 90° CW visual rotation  (with imshow origin=lower)
          k=2 → 180°, k=3 → 270° CW
        _cube_rotation stores k directly (0-3).
        """
        if not hasattr(self, '_last_data') or self._last_data is None:
            return None
        data = self._last_data
        if getattr(self, '_last_from_fits', False):
            img = np.nansum(data, axis=0).astype(np.float64)
        else:
            img = np.array(data, dtype=np.float64)

        self._cube_orig_shape = img.shape  # (ny_orig, nx_orig)

        k = getattr(self, '_cube_rotation', 0) % 4
        if k:
            img = np.rot90(img, k=k)

        if getattr(self, '_cube_flip_h', False):
            img = np.fliplr(img)
        if getattr(self, '_cube_flip_v', False):
            img = np.flipud(img)
        return img

    def _cube_transform_coords(self, xd, yd):
        """Map display coords (after rotate/flip) back to original data coords.

        Inverse formulas verified against np.rot90 + imshow(origin=lower):
          k=0: x_orig=xd,          y_orig=yd
          k=1: x_orig=nx_orig-1-yd, y_orig=xd
          k=2: x_orig=nx_orig-1-xd, y_orig=ny_orig-1-yd
          k=3: x_orig=yd,           y_orig=ny_orig-1-xd
        Flips are applied before rotation in the forward pass, so undo after.
        """
        if not hasattr(self, '_cube_orig_shape'):
            return int(round(xd)), int(round(yd))
        ny_orig, nx_orig = self._cube_orig_shape

        # Undo flips first (they were applied after rotation in the forward pass)
        x, y = xd, yd
        # Need the rotated shape to undo flips correctly
        k = getattr(self, '_cube_rotation', 0) % 4
        if k in (1, 3):
            ny_rot, nx_rot = nx_orig, ny_orig
        else:
            ny_rot, nx_rot = ny_orig, nx_orig
        if getattr(self, '_cube_flip_h', False):
            x = nx_rot - 1 - x
        if getattr(self, '_cube_flip_v', False):
            y = ny_rot - 1 - y

        # Undo rotation
        if k == 0:
            x_orig, y_orig = x, y
        elif k == 1:
            x_orig, y_orig = nx_orig - 1 - y, x
        elif k == 2:
            x_orig, y_orig = nx_orig - 1 - x, ny_orig - 1 - y
        else:  # k == 3
            x_orig, y_orig = y, ny_orig - 1 - x

        return int(round(x_orig)), int(round(y_orig))

    def _cube_redraw_transformed(self):
        """Redraw the cube image with the current transform + toolbar settings."""
        img = self._cube_get_transformed_data()
        if img is None:
            return
        self._applying_transform = True
        try:
            self.draw_image(img, cmap=self._last_cmap, scale=self._last_scale, from_fits=False)
        finally:
            self._applying_transform = False

    def _cube_rotate(self, degrees):
        """Rotate the cube image. degrees=90 → CW, degrees=-90 → CCW."""
        steps = (degrees // 90) % 4
        self._cube_rotation = (getattr(self, '_cube_rotation', 0) + steps) % 4
        self._cube_redraw_transformed()

    def _cube_flip(self, axis):
        """Flip the cube image horizontally ('h') or vertically ('v')."""
        if axis == 'h':
            self._cube_flip_h = not getattr(self, '_cube_flip_h', False)
        else:
            self._cube_flip_v = not getattr(self, '_cube_flip_v', False)
        self._cube_redraw_transformed()

    # ── Cube axis coord label toggle (X/Y ↔ RA/Dec) ──────────────────────────

    def _cube_toggle_coords(self):
        """Toggle cube axis tick labels between pixel X/Y and RA/Dec."""
        if not hasattr(self, 'ax') or self.ax is None:
            return
        mode = getattr(self, '_cube_coords_mode', 'xy')
        if mode == 'xy':
            self._cube_coords_mode = 'radec'
            self.coords_toggle_btn.setText('RA/Dec')
            self._cube_apply_radec_labels()
        else:
            self._cube_coords_mode = 'xy'
            self.coords_toggle_btn.setText('X/Y')
            self._cube_apply_xy_labels()
        self.canvas.draw_idle()

    def _cube_apply_xy_labels(self):
        """Set cube axes to plain pixel X/Y ticks."""
        if not hasattr(self, 'ax') or self.ax is None:
            return
        self.ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        self.ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        self.ax.set_xlabel('X (pixels)')
        self.ax.set_ylabel('Y (pixels)')
        # Restore auto pixel ticks
        self.ax.xaxis.set_major_locator(plt.AutoLocator())
        self.ax.yaxis.set_major_locator(plt.AutoLocator())

    def _cube_apply_radec_labels(self):
        """Set cube axes to RA/Dec tick labels derived from WCS."""
        if not hasattr(self, 'ax') or self.ax is None:
            return
        # Build tick positions and labels from the current pixel grid
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        # Sample ~5 evenly-spaced pixel positions across each axis
        x_pix = np.linspace(max(0, int(xlim[0])), int(xlim[1]), 5).astype(int)
        y_pix = np.linspace(max(0, int(ylim[0])), int(ylim[1]), 5).astype(int)

        try:
            ra_vals,  _ = self.pixel_to_ra_dec(x_pix, np.full_like(x_pix, int((ylim[0]+ylim[1])/2)))
            _,  dec_vals = self.pixel_to_ra_dec(np.full_like(y_pix, int((xlim[0]+xlim[1])/2)), y_pix)

            x_labels = [self.decimal_to_sexagesimal(float(r), is_ra=True)  for r in ra_vals]
            y_labels = [self.decimal_to_sexagesimal(float(d), is_ra=False) for d in dec_vals]

            self.ax.set_xticks(x_pix)
            self.ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
            self.ax.set_yticks(y_pix)
            self.ax.set_yticklabels(y_labels, fontsize=7)
            self.ax.set_xlabel('RA (J2000)')
            self.ax.set_ylabel('Dec (J2000)')
        except Exception as e:
            print(f"RA/Dec label error: {e}")
            self._cube_apply_xy_labels()

    # ── Background HST image overlay ──────────────────────────────────────────

    def _show_bkg_image_dialog(self):
        """Open dialog to select/fetch an HST background image and adjust its display."""
        dialog = QDialog(self)
        dialog.setWindowTitle('Background Image')
        dialog.setMinimumWidth(380)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(6)

        # ── Band selection ────────────────────────────────────────────────────
        layout.addWidget(QLabel('Select HST HiPS band:'))
        _hips_i = 'https://alaskybis.cds.unistra.fr/HST-hips/filter_I_hips/'
        _hips_b = 'https://alaskybis.cds.unistra.fr/HST-hips/filter_B_hips/'
        i_btn   = QPushButton('HST  I-band');  i_btn.setFixedHeight(30)
        b_btn   = QPushButton('HST  B-band');  b_btn.setFixedHeight(30)
        _hips_cxo = 'https://cdaftp.cfa.harvard.edu/cxc-hips/'
        cxo_btn = QPushButton('Chandra X-ray (broadband RGB)'); cxo_btn.setFixedHeight(30)
        clr_btn = QPushButton('Clear background'); clr_btn.setFixedHeight(26)
        layout.addWidget(i_btn)
        layout.addWidget(b_btn)
        layout.addWidget(cxo_btn)
        layout.addWidget(clr_btn)
        status = QLabel(''); status.setWordWrap(True)
        layout.addWidget(status)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # ── Display settings ──────────────────────────────────────────────────
        layout.addWidget(QLabel('Background display settings:'))

        _combo_qss = (
            "QComboBox { color: #eff0f1; background-color: #3c3f41; border: 1px solid #555; padding: 0 4px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { color: #eff0f1; background-color: #3c3f41;"
            "  selection-background-color: #d7801a; selection-color: #000; border: 1px solid #555; }"
        )

        # Colormap
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel('Colormap:'))
        bkg_cmap_combo = QComboBox()
        bkg_cmap_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        bkg_cmap_combo.addItems(['gray','gray_r','viridis','plasma','inferno','magma','hot','cool','Blues','Reds','Greens','bwr','seismic'])
        bkg_cmap_combo.setCurrentText(getattr(self, '_bkg_cmap', 'gray'))
        bkg_cmap_combo.setStyleSheet(_combo_qss)
        cmap_row.addWidget(bkg_cmap_combo)
        layout.addLayout(cmap_row)

        # Scale
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel('Scale:'))
        bkg_scale_combo = QComboBox()
        bkg_scale_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        bkg_scale_combo.addItems(['Linear','Log','Square root','Asinh'])
        bkg_scale_combo.setCurrentText(getattr(self, '_bkg_scale', 'Linear'))
        bkg_scale_combo.setStyleSheet(_combo_qss)
        scale_row.addWidget(bkg_scale_combo)
        layout.addLayout(scale_row)

        # Brightness slider
        bright_row = QHBoxLayout()
        bright_row.addWidget(QLabel('Brightness:'))
        bkg_bright_slider = QtWidgets.QSlider(Qt.Horizontal)
        bkg_bright_slider.setRange(-100, 100)
        bkg_bright_slider.setValue(int(getattr(self, '_bkg_brightness', 0.0) * 100))
        bkg_bright_slider.setFixedHeight(20)
        bright_row.addWidget(bkg_bright_slider)
        layout.addLayout(bright_row)

        # Contrast slider
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(QLabel('Contrast:'))
        bkg_contrast_slider = QtWidgets.QSlider(Qt.Horizontal)
        bkg_contrast_slider.setRange(10, 300)
        bkg_contrast_slider.setValue(int(getattr(self, '_bkg_contrast', 1.0) * 100))
        bkg_contrast_slider.setFixedHeight(20)
        contrast_row.addWidget(bkg_contrast_slider)
        layout.addLayout(contrast_row)

        # ── Live update helpers ───────────────────────────────────────────────
        def _apply_display():
            self._bkg_cmap       = bkg_cmap_combo.currentText()
            self._bkg_scale      = bkg_scale_combo.currentText()
            self._bkg_brightness = bkg_bright_slider.value() / 100.0
            self._bkg_contrast   = bkg_contrast_slider.value() / 100.0
            if getattr(self, '_bkg_data', None) is not None:
                self._bkg_artist = None
                self._draw_bkg_overlay()

        bkg_cmap_combo.currentTextChanged.connect(lambda _: _apply_display())
        bkg_scale_combo.currentTextChanged.connect(lambda _: _apply_display())
        bkg_bright_slider.valueChanged.connect(lambda _: _apply_display())
        bkg_contrast_slider.valueChanged.connect(lambda _: _apply_display())

        def _fetch(hips_url, label):
            status.setText(f'Fetching {label} image…')
            dialog.repaint()
            try:
                self._load_hips_background(hips_url, label, status, dialog)
            except Exception as e:
                status.setText(f'Error: {e}')

        i_btn.clicked.connect(lambda: _fetch(_hips_i, 'HST I-band'))
        cxo_btn.clicked.connect(lambda: _fetch(_hips_cxo, 'Chandra X-ray'))
        b_btn.clicked.connect(lambda: _fetch(_hips_b, 'HST B-band'))
        clr_btn.clicked.connect(lambda: [self._clear_bkg_image(), dialog.accept()])

        close_btn = QPushButton('Close')
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec_()

    def _load_hips_background(self, hips_url, label, status_label, dialog):
        """Fetch a HiPS cutout matched to the cube WCS and display it behind the cube."""
        global FITS_HEADER, FITS_DATA

        # Build a minimal WCS from the cube header
        try:
            from astropy.wcs import WCS as AstroWCS
            from astropy.io.fits import Header as FITSHeader
            # Use only the 2D spatial axes from the header
            h = FITSHeader(FITS_HEADER)
            # Make a clean 2-axis WCS header
            wcs_keys = ['NAXIS1','NAXIS2','CRPIX1','CRPIX2','CRVAL1','CRVAL2',
                        'CDELT1','CDELT2','CTYPE1','CTYPE2','CD1_1','CD1_2',
                        'CD2_1','CD2_2','PC1_1','PC1_2','PC2_1','PC2_2']
            wcs_h = FITSHeader()
            wcs_h['NAXIS'] = 2
            wcs_h['WCSAXES'] = 2
            for k in wcs_keys:
                if k in h:
                    wcs_h[k] = h[k]
            if 'NAXIS1' not in wcs_h and FITS_DATA is not None:
                wcs_h['NAXIS1'] = FITS_DATA.shape[-1]
                wcs_h['NAXIS2'] = FITS_DATA.shape[-2]
            wcs2d = AstroWCS(wcs_h)
        except Exception as e:
            if status_label: status_label.setText(f'WCS error: {e}')
            return

        # Fetch via astroquery.hips2fits.query_with_wcs
        try:
            from astroquery.hips2fits import hips2fits
            print(f"Querying hips2fits for: {hips_url}")
            # PNG-only HiPS (e.g. Chandra RGB) needs format='jpg'; FITS HiPS use 'fits'
            _fmt = 'jpg' if 'cxc-hips' in hips_url or 'cxc_hips' in hips_url else 'fits'
            result = hips2fits.query_with_wcs(
                hips=hips_url,
                wcs=wcs2d,
                format=_fmt,
                min_cut=0.5,
                max_cut=99.5,
                stretch='linear',
            )
            # FITS → HDUList; JPG/PNG → numpy array
            if _fmt == 'fits':
                bkg_data = result[0].data if hasattr(result, '__getitem__') else result.data
            else:
                bkg_data = np.array(result)  # (ny, nx, 3) or (3, ny, nx)
            print(f"hips2fits returned data shape: {np.array(bkg_data).shape}")
        except ImportError:
            # Fallback: plain HTTP fetch of the hips2fits endpoint
            import urllib.request, io
            nx = int(wcs2d.pixel_shape[0] if wcs2d.pixel_shape else
                     FITS_DATA.shape[-1] if FITS_DATA is not None else 256)
            ny = int(wcs2d.pixel_shape[1] if wcs2d.pixel_shape else
                     FITS_DATA.shape[-2] if FITS_DATA is not None else 256)
            ra  = float(wcs2d.wcs.crval[0])
            dec = float(wcs2d.wcs.crval[1])
            cdelt = abs(float(wcs2d.wcs.cdelt[0])) if wcs2d.wcs.cdelt[0] != 0 else 1/3600
            fov = cdelt * max(nx, ny)
            # Strip trailing slash for the hips parameter
            _hips_param = hips_url.rstrip('/')
            url = (f'https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits?'
                   f'hips={urllib.request.quote(_hips_param)}'
                   f'&width={nx}&height={ny}&fov={fov:.6f}'
                   f'&ra={ra:.6f}&dec={dec:.6f}&projection=TAN'
                   f'&format=fits')
            req = urllib.request.Request(url, headers={'User-Agent': 'HyperCube/1.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            from astropy.io import fits as astrofits
            with astrofits.open(io.BytesIO(raw)) as hdul:
                bkg_data = hdul[0].data

        if bkg_data is None:
            if status_label: status_label.setText('No data returned.')
            return

        # If RGB/RGBA (PNG HiPS like Chandra), convert to grayscale luminosity
        bkg_data = np.array(bkg_data)
        if bkg_data.ndim == 3:
            if bkg_data.shape[0] in (3, 4):   # (bands, ny, nx) from astroquery
                rgb = bkg_data[:3].astype(np.float64)
            elif bkg_data.shape[2] in (3, 4): # (ny, nx, bands)
                rgb = bkg_data[:, :, :3].astype(np.float64).transpose(2, 0, 1)
            else:
                rgb = bkg_data.astype(np.float64)
            # ITU-R BT.601 luminosity
            bkg_data = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

        # Store and display
        self._bkg_data = bkg_data.astype(np.float64)
        self._bkg_label = label
        self._draw_bkg_overlay()
        self.cube_opacity_slider.setVisible(True)

        if status_label:
            status_label.setText(f'✓  {label} loaded ({bkg_data.shape[1]}×{bkg_data.shape[0]} px)')

    def _draw_bkg_overlay(self):
        """Render the background image and cube overlay on self.ax."""
        if not hasattr(self, '_bkg_data') or self._bkg_data is None:
            return
        if not hasattr(self, 'ax') or self.ax is None:
            return

        bkg = self._bkg_data.copy()
        bmin, bmax = np.nanpercentile(bkg, [1, 99])

        # Apply scale (transfer function)
        scale = getattr(self, '_bkg_scale', 'Linear')
        bkg = bkg - bmin  # shift to zero
        bkg = np.where(bkg > 0, bkg, 0.0)
        if scale == 'Log':
            bkg = np.log1p(bkg)
        elif scale == 'Square root':
            bkg = np.sqrt(bkg)
        elif scale == 'Asinh':
            med = np.nanmedian(bkg[bkg > 0]) if np.any(bkg > 0) else 1.0
            bkg = np.arcsinh(bkg / (med or 1.0))

        # Normalise to [0,1]
        vmax = np.nanpercentile(bkg, 99)
        if vmax > 0:
            bkg = np.clip(bkg / vmax, 0, 1)
        else:
            bkg = np.zeros_like(bkg)

        # Apply brightness and contrast
        brightness = getattr(self, '_bkg_brightness', 0.0)
        contrast   = getattr(self, '_bkg_contrast',   1.0)
        mid = 0.5
        bkg_norm = np.clip((bkg - mid) * contrast + mid + brightness, 0, 1)

        # Remove any existing bkg artist
        if hasattr(self, '_bkg_artist') and self._bkg_artist is not None:
            try: self._bkg_artist.remove()
            except: pass
        self._bkg_artist = self.ax.imshow(
            bkg_norm, origin='lower', cmap=getattr(self, '_bkg_cmap', 'gray'),
            extent=self.ax.images[0].get_extent() if self.ax.images else None,
            zorder=0, alpha=1.0
        )
        # Set cube image alpha from slider
        opacity = self.cube_opacity_slider.value() / 100.0
        for img in self.ax.images:
            if img is not self._bkg_artist:
                img.set_alpha(opacity)
                img.set_zorder(1)
        self.canvas.draw_idle()

    def _cube_opacity_changed(self, value):
        """Update cube overlay alpha when slider moves."""
        if not hasattr(self, 'ax') or self.ax is None:
            return
        alpha = value / 100.0
        bkg_artist = getattr(self, '_bkg_artist', None)
        for img in self.ax.images:
            if img is not bkg_artist:
                img.set_alpha(alpha)
        self.canvas.draw_idle()

    def _clear_bkg_image(self):
        """Remove the background image and reset cube opacity."""
        if hasattr(self, '_bkg_artist') and self._bkg_artist is not None:
            try: self._bkg_artist.remove()
            except: pass
            self._bkg_artist = None
        self._bkg_data = None
        # Restore full opacity on cube
        if hasattr(self, 'ax') and self.ax is not None:
            for img in self.ax.images:
                img.set_alpha(1.0)
            self.canvas.draw_idle()
        self.cube_opacity_slider.setVisible(False)
        self.cube_opacity_slider.setValue(100)

    def _cube_reset_view(self):
        """Reset the cube viewport to its state at FITS load time:
        - B/W (gray) colormap
        - Linear scaling, 99% stretch
        - Fit-window zoom (full image visible)
        """
        if not hasattr(self, '_last_data') or self._last_data is None:
            return

        # Reset toolbar combos (blockSignals to avoid double-redraw)
        self.scale_combo.blockSignals(True)
        self.stretch_combo.blockSignals(True)
        self.scale_combo.setCurrentText("Linear")
        self.stretch_combo.setCurrentText("99%")
        self.scale_combo.blockSignals(False)
        self.stretch_combo.blockSignals(False)

        # Redraw with gray cmap, original data
        self.draw_image(
            FITS_DATA if hasattr(self, '_last_from_fits') and self._last_from_fits else self._last_data,
            cmap='gray',
            scale='linear',
            from_fits=True
        )

        # Reset zoom to fit window
        self._cube_zoom_apply(mode='fit_window')
        self._sync_zoom_combo()

    def _cube_redraw_with_current_settings(self):
        """Redraw the cube image using current toolbar stretch and scale settings."""
        if not hasattr(self, '_last_data') or self._last_data is None:
            return
        self.draw_image(
            self._last_data,
            cmap=self._last_cmap,
            scale=self._last_scale,
            from_fits=self._last_from_fits
        )

    def _init_placeholder_canvas(self):
        """Draw the HyperCube logo centred on a dark canvas before a FITS file is loaded."""
        fig = self.canvas.figure
        fig.patch.set_facecolor('#2b2b2b')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for spine in ax.spines.values():
            spine.set_visible(False)

        try:
            logo_path = resource_path('hypercube_logo.png')
            logo = plt.imread(logo_path)
            # Place logo in a centred inset axes so it scales with the window
            logo_ax = fig.add_axes([0.25, 0.2, 0.5, 0.5])
            logo_ax.imshow(logo)
            logo_ax.axis('off')
        except Exception:
            # Fallback to text if image can't be loaded
            ax.text(0.5, 0.55, 'HYPERCUBE',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=28, fontweight='bold', color='#d7801a',
                    fontfamily='monospace')

        ax.text(0.5, 0.1, 'Open a FITS file to begin',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, color='#888888')

        fig.tight_layout(pad=0)
        self.canvas.draw()

    def set_fit_params_window(self, fit_params_window):
        """Store the reference to FitParamsWindow after it's created."""
        self.fit_params_window = fit_params_window


    def on_mouse_move(self, event):
        global spectrum
        """Update rectangle position and spectrum in right panel."""
        if self.locked:
            return  # Do nothing if locked
        if getattr(self, '_bc_drag_active', False):
            return  # Suppress spaxel update during brightness/contrast drag
        
        if (event.inaxes) and (self.is_1d_spectrum == False):
            # Round to nearest integer spaxel centre in display coords.
            _xd = int(np.floor(event.xdata + 0.5))
            _yd = int(np.floor(event.ydata + 0.5))
            # Map display coords back to original data coords (undo rotate/flip)
            x, y = self._cube_transform_coords(_xd, _yd)
            self.cursor_pos = (x, y)
            
            if self.fits_data is not None:
                self.current_spaxel = (x, y)
                spectrum = self.get_spectrum_at_spaxel(x, y)
                self.update_spectrum_panel(spectrum)
                
                # Update red rectangle position
                self.red_rect.set_xy((x - 0.5, y - 0.5))
                self.red_rect.set_visible(True)
                self.canvas.draw_idle()
                
                # Always update the spaxel info overlay on the image
                self.update_spaxel_overlay(x, y)
                
                # Update fit result buttons in the side panel (only when it exists)
                if self.fit_params_window is not None:
                    self.update_buttons(x, y)




    def _spectrum_hbar_pan(self, value):
        """Pan the spectrum axes horizontally from the scrollbar position (0-1000)."""
        if not hasattr(self, 'spectrum_ax') or self.spectrum_ax is None:
            return
        if not hasattr(self, '_spec_wav_min') or self._spec_wav_min is None:
            return
        xlim = self.spectrum_ax.get_xlim()
        view_w = xlim[1] - xlim[0]
        full_w = self._spec_wav_max - self._spec_wav_min
        pan_range = full_w - view_w
        if pan_range <= 0:
            return
        frac = value / 1000.0
        new_x0 = self._spec_wav_min + frac * pan_range
        self.spectrum_ax.set_xlim(new_x0, new_x0 + view_w)
        self.spectrum_canvas.draw_idle()

    def _spectrum_update_hbar(self):
        """Sync the spectrum scrollbar to the current xlim.
        Shows/hides it based on whether the view is zoomed in."""
        if not hasattr(self, 'spec_hbar') or not hasattr(self, 'spectrum_ax'):
            return
        if self.spectrum_ax is None or not hasattr(self, '_spec_wav_min'):
            return
        xlim = self.spectrum_ax.get_xlim()
        view_w = xlim[1] - xlim[0]
        full_w = self._spec_wav_max - self._spec_wav_min
        pan_range = full_w - view_w
        if pan_range <= 1e-6:
            self.spec_hbar.hide()
            return
        self.spec_hbar.show()
        frac = (xlim[0] - self._spec_wav_min) / pan_range
        self.spec_hbar.blockSignals(True)
        self.spec_hbar.setValue(int(np.clip(frac, 0, 1) * 1000))
        self.spec_hbar.blockSignals(False)

    def on_spectrum_mouse_move(self, event):
        """Handle mouse movement in the spectrum plot."""
        if event.inaxes == self.spectrum_ax:
            # Update the cursor position in the spectrum plot
            self.spectrum_cursor_pos = (event.xdata, event.ydata)

            # Update channel map span while C is held
            if self._chanmap_active and self._chanmap_span is not None and event.xdata is not None:
                pass  # handled below
            # Update any active subtraction window spans
            if event.xdata is not None:
                for _key, _sm in self._submap.items():
                    if _sm['active'] and _sm['span'] is not None:
                        _x0 = min(_sm['start'], event.xdata)
                        _x1 = max(_sm['start'], event.xdata)
                        _xy = _sm['span'].get_xy()
                        _n = len(_xy)
                        _xy[:, 0] = [_x0, _x0, _x1, _x1, _x0][:_n]
                        _sm['span'].set_xy(_xy)
            # Update channel map span while C is held (main)
            if self._chanmap_active and self._chanmap_span is not None and event.xdata is not None:
                wav = event.xdata
                x0 = min(self._chanmap_start, wav)
                x1 = max(self._chanmap_start, wav)
                xy = self._chanmap_span.get_xy()
                # axvspan polygon is either 4 or 5 vertices depending on mpl version
                n = len(xy)
                xs = [x0, x0, x1, x1, x0][:n]
                xy[:, 0] = xs
                self._chanmap_span.set_xy(xy)
                self.spectrum_canvas.draw_idle()

            # Update wavelength/flux overlay
            if hasattr(self, 'spectrum_info_text') and event.xdata is not None:
                _pix = int(np.argmin(np.abs(wavelengths - event.xdata))) if wavelengths is not None else "—"
                self.spectrum_info_text.set_text(
                    f"px  {_pix}\n\u03bb  {event.xdata:.4f}\nF  {event.ydata:.4e}"
                )
                self.spectrum_canvas.draw_idle()
    
            # If we are in the middle of drawing a line, update the line's end position
            if self.drawing_line and self.current_line:
                # Get current position and update the line end coordinates
                end_x, end_y = self.spectrum_cursor_pos
                self.current_line.set_xdata([self.line_start[0], end_x])
                self.current_line.set_ydata([self.line_start[1], end_y])
    
                # Redraw the canvas to reflect the updated line
                self.spectrum_canvas.draw()


    def on_mouse_press(self, event):
        """Start zoom selection."""
        if event.button == 1 and event.inaxes == self.spectrum_ax:
            self.zoom_start_x = event.xdata
            if self.zoom_rect:
                self.zoom_rect.remove()
            self.zoom_rect = self.spectrum_ax.axvspan(self.zoom_start_x, self.zoom_start_x, color='grey', alpha=0.3)
            self.spectrum_canvas.draw_idle()

    def on_mouse_drag(self, event):
        """Update zoom selection rectangle dynamically."""
        if event.button == 1 and event.inaxes == self.spectrum_ax and self.zoom_start_x is not None:
            x_end = event.xdata
            if x_end is not None:
                if self.zoom_rect:
                    self.zoom_rect.remove()
                self.zoom_rect = self.spectrum_ax.axvspan(self.zoom_start_x, x_end, color='grey', alpha=0.3)
                self.spectrum_canvas.draw_idle()

    def on_mouse_release(self, event):
        """Apply zoom when mouse is released."""
        if event.button == 1 and event.inaxes == self.spectrum_ax and self.zoom_start_x is not None:
            zoom_end_x = event.xdata
            if zoom_end_x is not None and self.zoom_start_x != zoom_end_x:
                x_min, x_max = min(self.zoom_start_x, zoom_end_x), max(self.zoom_start_x, zoom_end_x)
                self.spectrum_ax.set_xlim(x_min, x_max)
                self._spectrum_update_hbar()
                self.zoom_limits = (x_min, x_max)  # Store zoom limits globally
                self.zoom_active = True
    
                # Dynamically adjust the y-axis based on the zoomed x-axis range
                mask = (wavelengths >= x_min) & (wavelengths <= x_max)
                if np.any(mask):  # Ensure there are valid points in the selected range
                    y_min, y_max = np.nanmin(self.spectrum_line.get_ydata()[mask]), np.nanmax(self.spectrum_line.get_ydata()[mask])
                else:  # If no valid points, use full y-range
                    y_min, y_max = np.nanmin(self.spectrum_line.get_ydata()), np.nanmax(self.spectrum_line.get_ydata())
    
                padding = self.zoom_pad * (y_max - y_min)
                self.spectrum_ax.set_ylim(y_min - padding/2, y_max + padding)
    
                self.spectrum_ax.figure.canvas.draw_idle()  # Ensure immediate update
    
            if self.zoom_rect:
                self.zoom_rect.remove()
                self.zoom_rect = None
            self.zoom_start_x = None

    def show_context_menu(self, event):
        if event.button == 3:  # Right-click
            menu = QMenu(self)
            reset_action = menu.addAction("Reset Zoom")
            action = menu.exec_(self.spectrum_canvas.mapToGlobal(event.guiEvent.pos()))
            if action == reset_action:
                self.reset_zoom()

    def reset_zoom(self):
        """Reset zoom to the full spectrum range for both x and y axes."""
        self.spectrum_ax.set_xlim(wavelengths.min(), wavelengths.max())
    
        # Compute full y-range and apply padding
        y_min, y_max = np.nanmin(self.spectrum_line.get_ydata()), np.nanmax(self.spectrum_line.get_ydata())
        padding = self.zoom_pad * (y_max - y_min)
        self.spectrum_ax.set_ylim(y_min - padding/2, y_max + padding)
    
        self.zoom_active = False
        self.spectrum_canvas.draw_idle()  # Ensure the reset happens immediately

    

    def press_scaleflux_button(self):
        text_box = QLineEdit()
        text_box.setText(str(self.fluxscalefactor))  # Set the initial value from the dataframe
        text_box.show()
        text_box.returnPressed.connect(lambda: self.scale_flux(text_box))
        
    def press_scaleWL_button(self):
        text_box = QLineEdit()
        text_box.setText(str(self.WLscalefactor))  # Set the initial value from the dataframe
        text_box.show()
        text_box.returnPressed.connect(lambda: self.scale_WL(text_box))

    def scale_flux(self,text_box):
        global FITS_DATA
        self.fluxscalefactor = text_box.text()
        FITS_DATA = FITS_DATA*np.float64(self.fluxscalefactor)
        text_box.hide()
        
    def scale_WL(self,text_box):
        global wavelengths
        self.WLscalefactor = text_box.text()
        wavelengths = wavelengths*np.float64(self.WLscalefactor)
        text_box.hide()

    def open_fits_file(self):
        """Opens a FITS file and stores header & data, handling 1D spectra, 3D cubes, and bintables"""
        global FITS_HEADER, FITS_DATA, wavelengths, snr_map, spectrum
        print('Opening file...')
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "FITS Files (*.fits);;All Files (*)", options=options)
        
        if file_path:
            with fits.open(file_path) as hdul:
                self.fits_header = None
                self.fits_data = None
                self.is_1d_spectrum = False
                self.is_bintable = False
                self.is_3d_cube = False
                self.column_names = []
                self.has_explicit_wavelengths = False
    
                # Check extensions for data type
                for ext in range(len(hdul)):
                    header = hdul[ext].header
                    data = hdul[ext].data
    
                    if data is None:
                        continue
                        
                    # Check for binary table
                    if header.get('XTENSION') == 'BINTABLE':
                        print(f"Found binary table in extension {ext}")
                        self.fits_header = header
                        self.fits_data = data
                        self.is_bintable = True
                        spectrum = data
                        self.is_1d_spectrum = True
                        self.column_names = [header.get(f'TTYPE{i}') for i in range(1, header.get('TFIELDS', 0)+1)]
                        if any(col and 'wave' in col.lower() for col in self.column_names):
                            self.has_explicit_wavelengths = True
                        break
                    
                    # Check for 1D spectrum
                    elif data.ndim == 1:
                        print(f"Found 1D spectrum in extension {ext}")
                        self.fits_header = header
                        self.fits_data = data
                        spectrum = data
                        self.is_1d_spectrum = True
                        self.has_explicit_wavelengths = True
                        if any(key in header for key in ['CRVAL1', 'CDELT1', 'CD1_1']):
                            wavelengths = self._construct_wavelengths_from_header()
                        break
                    
                    # Check for 3D/4D cube (like ALMA data)
                    elif data.ndim in [3, 4] and 'CTYPE3' in header and 'CRVAL3' in header:
                        print(f"Found {data.ndim}D cube in extension {ext}")
                        self.fits_header = header
                        self.fits_data = data
                        self.is_3d_cube = True
                        
                        # For ALMA-style cubes (1, nchan, ny, nx)
                        if data.ndim == 4 and data.shape[0] == 1:  # Stokes axis
                            self.fits_data = data[0]  # Remove stokes dimension
                        
                        # Generate wavelength array from header
                        if 'CTYPE3' in header and header['CTYPE3'].startswith(('FREQ','VELO','VRAD','WAVE')):
                            wavelengths = self._construct_wavelengths_from_header(axis=3)
                            self.has_explicit_wavelengths = True
                        break
    
                # Fallback to primary extension
                if self.fits_header is None:
                    print("No valid extension found, defaulting to primary.")
                    self.fits_header = hdul[0].header
                    self.fits_data = hdul[0].data if hdul[0].data is not None else hdul[1].data
                    if self.fits_data.ndim == 1:
                        self.is_1d_spectrum = True
                        if any(key in self.fits_header for key in ['CRVAL1', 'CDELT1', 'CD1_1']):
                            self.has_explicit_wavelengths = True
    
                FITS_HEADER = self.fits_header
                FITS_DATA = self.fits_data
                
                # Handle different data types
                if self.is_bintable:
                    print("Binary table detected - showing column selection")
                    self.show_column_selection_dialog()
                    return
                    
                elif self.is_1d_spectrum:
                    print("1D spectrum detected")
                    if not self.has_explicit_wavelengths:
                        print("Warning: No wavelength info found - using pixel indices")
                        wavelengths = np.arange(len(FITS_DATA))
                    self.process_1d_spectrum()
                    
                elif self.is_3d_cube:
                    print(f"{FITS_DATA.ndim}D cube detected")
                    if FITS_DATA.ndim == 3:  # (nchan, ny, nx)
                        self.process_3d_cube()
                        # For 3D cubes, create field selection dialog
                    #     self.column_names = ['Flux']  # Default name
                    #     self.show_column_selection_dialog()
                    # else:
                    #     self.process_3d_cube()
    
                # Extract observation info
                self.extract_observation_info()
    
    def show_column_selection_dialog(self):
        """Create a dialog for selecting wavelength and flux columns from bintable"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Table Columns")
        dialog.setMinimumWidth(500)  # Slightly wider for better button spacing
        
        layout = QVBoxLayout()
        info_label = QLabel("This FITS file contains multiple columns. Please select:")
        info_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Wavelength selection
        wave_group = QGroupBox("Select Wavelength Vector")
        wave_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        wave_layout = QVBoxLayout()
        wave_layout.setSpacing(10)
        wave_layout.setContentsMargins(10, 25, 10, 15)  # More vertical padding
        
        self.wave_buttons = []
        for i, name in enumerate(self.column_names):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    padding: 5px;
                    border: 1px solid #888;
                    border-radius: 4px;
                    min-width: 80px;
                }
                QPushButton:checked {
                    background-color: lime;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            btn.clicked.connect(lambda _, x=i: self.select_column(x, 'wave'))
            self.wave_buttons.append(btn)
            wave_layout.addWidget(btn)
        wave_group.setLayout(wave_layout)
        layout.addWidget(wave_group)
        
        # Flux selection
        flux_group = QGroupBox("Select Flux Vector")
        flux_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        flux_layout = QVBoxLayout()
        flux_layout.setSpacing(10)
        flux_layout.setContentsMargins(10, 25, 10, 15)  # More vertical padding
        
        self.flux_buttons = []
        for i, name in enumerate(self.column_names):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    padding: 5px;
                    border: 1px solid #888;
                    border-radius: 4px;
                    min-width: 80px;
                }
                QPushButton:checked {
                    background-color: lime;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            btn.clicked.connect(lambda _, x=i: self.select_column(x, 'flux'))
            self.flux_buttons.append(btn)
            flux_layout.addWidget(btn)
        flux_group.setLayout(flux_layout)
        layout.addWidget(flux_group)
        
        # Confirm button
        confirm_btn = QPushButton("Confirm Selection")
        confirm_btn.setStyleSheet("""
            QPushButton {
                padding: 8px;
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        confirm_btn.clicked.connect(lambda: self.finalize_bintable_selection(dialog))
        layout.addWidget(confirm_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def select_column(self, index, col_type):
        """Handle column selection (single selection per type)"""
        buttons = self.wave_buttons if col_type == 'wave' else self.flux_buttons
        for i, btn in enumerate(buttons):
            # Store current style to preserve other properties
            current_style = btn.styleSheet()
            if i == index:
                btn.setChecked(True)
                # Add the checked style while preserving other styles
                btn.setStyleSheet(current_style + """
                    QPushButton {
                        background-color: lime;
                        font-weight: bold;
                    }
                """)
            else:
                btn.setChecked(False)
                # Reset to base style
                btn.setStyleSheet("""
                    QPushButton {
                        padding: 5px;
                        border: 1px solid #888;
                        border-radius: 4px;
                        min-width: 80px;
                    }
                """)
    
    def finalize_bintable_selection(self, dialog):
        """Process selected columns and continue with file loading"""
        global wavelengths, FITS_DATA  # Declare globals
        
        try:
            # Get selected columns
            wave_idx = next((i for i, btn in enumerate(self.wave_buttons) if btn.isChecked()), -1)
            flux_idx = next((i for i, btn in enumerate(self.flux_buttons) if btn.isChecked()), -1)
            
            if wave_idx == -1 or flux_idx == -1:
                QMessageBox.warning(self, "Selection Error", 
                                  "Please select both wavelength and flux columns")
                return
            
            # Extract selected data - proper FITS binary table access
            
            # print(f'wavelength index selected: {wave_idx}')
            # print(f'flux index selected: {flux_idx}')
            wavelengths = self.fits_data[0][wave_idx]  # Direct column access
            flux = self.fits_data[0][flux_idx]
            
            # Update global data references
            FITS_DATA = flux
            self.process_1d_spectrum()
            self.extract_observation_info()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               f"Failed to load selected columns:\n{str(e)}")
        finally:
            dialog.close()
    
    def _construct_wavelengths_from_header(self, axis=1):
        """Construct wavelength array from FITS header for given axis"""
        header = self.fits_header
        naxis = header[f'NAXIS{axis}']
        
        # Try standard WCS keywords
        crval = header.get(f'CRVAL{axis}')
        cdelt = header.get(f'CDELT{axis}')
        crpix = header.get(f'CRPIX{axis}', 1)
        
        if None not in (crval, cdelt):
            return crval + cdelt * (np.arange(naxis) + 1 - crpix)
        
        # Try CD matrix
        cd = header.get(f'CD{axis}_{axis}')
        if cd is not None:
            return crval + cd * (np.arange(naxis) + 1 - crpix)
        
        # Try PC matrix
        pc = header.get(f'PC{axis}_{axis}')
        if pc is not None and cdelt is not None:
            return crval + pc * cdelt * (np.arange(naxis) + 1 - crpix)
        
        # Fallback for frequency axes
        if header.get(f'CTYPE{axis}','').startswith('FREQ'):
            restfreq = header.get('RESTFRQ', header.get('RESTFREQ'))
            if restfreq is not None:
                return (1 - np.arange(naxis) * header.get('CDELT3', 1) / restfreq * 299792.458)
        
        print(f"Warning: No valid WCS found for axis {axis} - using pixel indices")
        return np.arange(naxis)
    
    
    
    def process_1d_spectrum(self):
        """Common processing for all 1D spectra"""
        global wavelengths, snr_map
        
        # Create dummy 2D array for visualization
        dummy_2d = np.tile(FITS_DATA, (10, 1)).T
        
        # Initialize dummy WCS and SNR map
        self.wcs = WCS(naxis=2)
        snr_map = np.zeros((10, 10))
        
        # Show the spectrum
        self.update_spectrum_panel(FITS_DATA)
    
    def process_3d_cube(self):
        """Processing for 3D cubes"""
        global wavelengths, snr_map
        
        self.draw_image(FITS_DATA, cmap='gray', scale='linear', from_fits=True)
        self.wcs = WCS(self.fits_header)
        
        # Initialize SNR map
        _, nx, ny = FITS_DATA.shape
        snr_map = np.zeros((nx, ny)) + 1E8
        
        # Extract spectral axis info
        spectral_sampling = FITS_HEADER.get('CDELT3', FITS_HEADER.get('CD3_3', 1))
        wavelengths = (FITS_HEADER['CRVAL3'] + 
                     spectral_sampling * 
                     (np.arange(FITS_DATA.shape[0]) - FITS_HEADER.get('CRPIX3', 1) + 1))
    
    def extract_observation_info(self):
        """Extract common observation info"""
        global df_obs
        
        source_name = FITS_HEADER.get('OBJECT', '')
        source_redshift = FITS_HEADER.get('REDSHIFT', '')
        resolving_power = FITS_HEADER.get('RESOLVINGP', '')
        
        df_obs.loc[0] = [source_name, source_redshift, resolving_power]
        print("FITS file loaded successfully!")
        # Enable data-dependent toolbar buttons now that a file is loaded
        self.WLscalefactor_button.setEnabled(True)
        self.fluxscalefactor_button.setEnabled(True)
        # Prompt user to confirm / resolve the source name from NED
        # Use QTimer to let the main window finish painting before showing the dialog
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(300, self._prompt_source_on_load)



    def _prompt_source_on_load(self):
        """Show the source information dialog after a FITS file is loaded.
        If FitParamsWindow is open, delegate to it; otherwise call standalone version.
        """
        if self.fit_params_window is not None:
            self.fit_params_window._show_source_dialog()
        else:
            # Build a minimal standalone dialog matching FitParamsWindow._show_source_dialog
            global df_obs, FITS_HEADER
            current_name = str(df_obs.loc[0, 'sourcename']) if len(df_obs) > 0 else ''
            dialog = QDialog(self)
            dialog.setWindowTitle('Source Information')
            dialog.setMinimumWidth(420)
            layout = QVBoxLayout(dialog)
            layout.setSpacing(10)
            layout.addWidget(QLabel('Source name:'))
            name_edit = QLineEdit(current_name)
            name_edit.setPlaceholderText('e.g. NGC 1068')
            layout.addWidget(name_edit)
            resolve_name_btn  = QPushButton('Resolve name  (NED)')
            resolve_coord_btn = QPushButton('Resolve coordinates  (NED)')
            layout.addWidget(resolve_name_btn)
            layout.addWidget(resolve_coord_btn)
            status = QLabel(''); status.setWordWrap(True)
            layout.addWidget(status)
            btn_row = QHBoxLayout()
            ok_btn = QPushButton('OK'); ok_btn.setDefault(True)
            cancel_btn = QPushButton('Cancel')
            btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
            layout.addLayout(btn_row)

            def _apply(ned_name, ned_z):
                name_edit.setText(ned_name)
                df_obs.loc[0, 'sourcename'] = ned_name
                df_obs.loc[0, 'redshift']   = ned_z
                # Enable the background image button now that source is resolved
                if hasattr(self, 'bkg_image_btn'):
                    self.bkg_image_btn.setEnabled(True)

            def _by_name():
                name = name_edit.text().strip()
                if not name: status.setText('Enter a name first.'); return
                status.setText(f'Querying NED for "{name}"...'); dialog.repaint()
                try:
                    from astroquery.ipac.ned import Ned
                    r = Ned.query_object(name)
                    ned_name = str(r['Object Name'][0])
                    ned_z    = float(r['Redshift'][0])
                    _apply(ned_name, ned_z)
                    status.setText(f'✓  {ned_name}   z = {ned_z:.6f}')
                except Exception as e:
                    status.setText(f'NED query failed: {e}')

            def _by_coords():
                try:
                    ra  = float(FITS_HEADER.get('CRVAL1', ''))
                    dec = float(FITS_HEADER.get('CRVAL2', ''))
                except: status.setText('Could not read RA/Dec from FITS header.'); return
                status.setText(f'Querying NED at RA={ra:.4f}, Dec={dec:.4f}...'); dialog.repaint()
                try:
                    from astroquery.ipac.ned import Ned
                    import astropy.units as u
                    from astropy.coordinates import SkyCoord
                    coord = SkyCoord(ra=ra, dec=dec, unit='deg')
                    r = Ned.query_region(coord, radius=0.5 * u.arcmin)
                    if len(r) == 0: raise ValueError('No objects found')
                    ned_name = str(r['Object Name'][0])
                    ned_z    = float(r['Redshift'][0])
                    _apply(ned_name, ned_z)
                    status.setText(f'✓  {ned_name}   z = {ned_z:.6f}')
                except Exception as e:
                    status.setText(f'NED query failed: {e}')

            def _commit():
                df_obs.loc[0, 'sourcename'] = name_edit.text().strip()
                dialog.accept()

            resolve_name_btn.clicked.connect(_by_name)
            resolve_coord_btn.clicked.connect(_by_coords)
            ok_btn.clicked.connect(_commit)
            name_edit.returnPressed.connect(_commit)
            cancel_btn.clicked.connect(dialog.reject)
            dialog.exec_()

    def pixel_to_ra_dec(self, x, y):
        """Convert pixel coordinates (x, y) to RA, Dec using the WCS information from FITS header.
        Handles both 2D and 4D WCS (like ALMA data with Stokes parameter)."""
        
        # Ensure inputs are numpy arrays
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        # Check WCS dimensionality
        if self.wcs.naxis == 4:
            # For 4D WCS (RA, Dec, Freq, Stokes) - ALMA case
            z = np.zeros_like(x)  # Frequency axis (use 0 or reference pixel)
            stokes = np.zeros_like(x)  # Stokes axis (I=0)
            ra, dec, _, _ = self.wcs.all_pix2world(x, y, z, stokes, 0)
        elif self.wcs.naxis == 3:
            # For 3D WCS (RA, Dec, Freq)
            z = np.zeros_like(x)  # Frequency axis
            ra, dec, _ = self.wcs.all_pix2world(x, y, z, 0)
        else:
            # Standard 2D WCS
            ra, dec = self.wcs.all_pix2world(x, y, 0)
        
        return ra, dec


    # Helper function to convert decimal to sexagesimal
    def decimal_to_sexagesimal(self, ra_dec, is_ra=True):
        """
        Convert decimal RA or Dec to sexagesimal format (RA: h:m:s, Dec: d:m:s).
        :param ra_dec: decimal RA or Dec.
        :param is_ra: True if it's RA (hours), False if it's Dec (degrees).
        :return: sexagesimal string.
        """
        if is_ra:
            # Convert RA from degrees to hours, minutes, seconds
            total_seconds = ra_dec * 3600.0 / 15.0  # RA in hours, 15 degrees per hour
        else:
            # Convert Dec from degrees to degrees, arcminutes, arcseconds
            total_seconds = ra_dec * 3600.0
    
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
    
        if is_ra:
            return f"{hours:02}:{minutes:02}:{seconds:05.2f}"  # RA: h:m:s
        else:
            sign = '+' if ra_dec >= 0 else '-'
            degrees = abs(int(ra_dec))
            minutes = int((abs(ra_dec) * 60) % 60)
            seconds = (abs(ra_dec) * 3600) % 60
            return f"{sign}{degrees:02}:{minutes:02}:{seconds:05.2f}"  # Dec: ±d:m:s


    
    def update_spaxel_overlay(self, x, y):
        """Update the RA/Dec/X/Y/N text overlay on the cube image. Always called on mouse move."""
        if self.is_1d_spectrum:
            return
        ra, dec = self.pixel_to_ra_dec(x, y)
        ra_sexagesimal  = self.decimal_to_sexagesimal(ra[0],  is_ra=True)
        dec_sexagesimal = self.decimal_to_sexagesimal(dec[0], is_ra=False)
        n = self.get_spaxel_number(x, y)
        if hasattr(self, 'spaxel_info_text'):
            self.spaxel_info_text.set_text(
                f"RA  {ra_sexagesimal}\n"
                f"Dec {dec_sexagesimal}\n"
                f"X {x}   Y {y}   N {n}"
            )
            self.canvas.draw_idle()

    def update_buttons(self, x, y):
        """Refresh fit-result values in the Fit Parameters side panel for spaxel (x, y).
        Only called when fit_params_window exists."""
        # Only clear non-spectrum curves when NOT on the init-guess spaxel,
        # so interactive initial guesses persist while the user stays there.
        on_init_spaxel = (self._init_guess_spaxel is not None and
                          int(x) == self._init_guess_spaxel[0] and
                          int(y) == self._init_guess_spaxel[1])
        if not on_init_spaxel:
            for line in self.spectrum_ax.get_lines():
                if line.get_label() != "_child0":
                    line.remove()
            self.gaussian_component_lines.clear()
            self.spectrum_canvas.draw()

        for region_ID in np.unique(np.int64(df_cont['region_ID'])):
            show_init = on_init_spaxel  # redraw init curves when back on home spaxel
            self.rebuild_plot(region_ID, from_file=False,
                              show_init=show_init, show_fit=(not show_init), x=x, y=y)

    def get_spaxel_number(self, x, y):
        """Get the spaxel number for the given pixel coordinates."""
        # Example: convert (x, y) to a spaxel index (could be linear index or 2D)
        return y #* self.image_width + x  # Example of converting to a 1D spaxel number




    def update_spectrum_panel(self, spectrum):
        """Efficiently update the right panel to show the 1D spectrum."""
        if spectrum is not None:
            if not hasattr(self, 'spectrum_fig'):
                # Initialize the spectrum figure and canvas
                self.spectrum_fig = plt.Figure(figsize=(5, 4))
                self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
                self.spectrum_ax = self.spectrum_fig.add_subplot(111)
                
                # Create the spectrum line (step plot)
                self.spectrum_line, = self.spectrum_ax.step([], [], lw=0.5, color='w')
                
                # Apply the updated QSS styling
                apply_mpl_qss_style(self.spectrum_fig, self.spectrum_ax, self.spectrum_line)
                
                # Set axis labels and title
                self.spectrum_ax.set_title("1D Spectrum at Cursor Position")
                self.spectrum_ax.set_xlabel("Wavelength")
                self.spectrum_ax.set_ylabel("Flux")
                
                # Mouse event connections
                self.spectrum_canvas.mpl_connect('button_press_event', self.on_mouse_press)
                self.spectrum_canvas.mpl_connect('motion_notify_event', self.on_mouse_drag)
                self.spectrum_canvas.mpl_connect('button_release_event', self.on_mouse_release)
                self.spectrum_canvas.mpl_connect("button_press_event", self.show_context_menu)
                self.spectrum_canvas.mpl_connect('motion_notify_event', self.on_spectrum_mouse_move)
                
                # Ensure canvas reflects the QSS style (PyQt5-level styling)
                self.spectrum_canvas.setStyleSheet("background-color: #2E3440; border: none;")
                
                # Wavelength/flux overlay (top-right corner, mirrors the cube RA/Dec overlay)
                self.spectrum_info_text = self.spectrum_ax.text(
                    0.01, 0.99, '',
                    transform=self.spectrum_ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    fontsize=8, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor='none')
                )

                # Reveal the spectrum toolbar now that we have a spectrum
                self.spec_bar_widget.show()
                # Add the Matplotlib canvas to the layout
                layout = self.right_panel.layout()
                layout.addWidget(self.spectrum_canvas)

                # Horizontal pan scrollbar below the spectrum
                self.spec_hbar = QScrollBar(Qt.Horizontal)
                self.spec_hbar.setRange(0, 1000)
                self.spec_hbar.setValue(500)
                self.spec_hbar.setSingleStep(10)
                self.spec_hbar.setPageStep(100)
                self.spec_hbar.setFixedHeight(14)
                self.spec_hbar.setStyleSheet(
                    "QScrollBar:horizontal { height: 14px; }"
                    "QScrollBar::handle:horizontal { min-width: 30px; }"
                )
                self.spec_hbar.valueChanged.connect(self._spectrum_hbar_pan)
                layout.addWidget(self.spec_hbar)

            # Update the data instead of recreating the plot
            self.spectrum_line.set_xdata(wavelengths)
            self.spectrum_line.set_ydata(spectrum)

            if not self.zoom_active:
                # Default full range
                self.spectrum_ax.set_xlim(wavelengths.min(), wavelengths.max())
                self._spec_wav_min = float(wavelengths.min())
                self._spec_wav_max = float(wavelengths.max())
                self._spectrum_update_hbar()
                y_min, y_max = np.nanmin(spectrum), np.nanmax(spectrum)
            else:
                # Get the zoomed x-axis range
                xlim_min, xlim_max = self.spectrum_ax.get_xlim()
    
                # Select only the part of the spectrum within the zoomed range
                mask = (wavelengths >= xlim_min) & (wavelengths <= xlim_max)
                if np.any(mask):  # Ensure there are valid values in range
                    y_min, y_max = np.nanmin(spectrum[mask]), np.nanmax(spectrum[mask])
                else:  # If no valid data, use default
                    y_min, y_max = np.nanmin(spectrum), np.nanmax(spectrum)
    
            # Apply 10% padding to the y-limits
            padding = self.zoom_pad * (y_max - y_min)
            self.spectrum_ax.set_ylim(y_min - padding/2, y_max + padding)

            # Refresh the figure
            self.spectrum_canvas.draw()

    def _chanmap_update_centre_display(self):
        """Update the centre-pixel button label from the current span limits."""
        global wavelengths
        if self._chanmap_span is None or wavelengths is None:
            self.chanmap_centre_btn.setText("Pixel: —")
            return
        xy = self._chanmap_span.get_xy()
        wav0 = xy[:, 0].min()
        wav1 = xy[:, 0].max()
        wav_centre = (wav0 + wav1) / 2
        pix = np.argmin(np.abs(wavelengths - wav_centre))
        self.chanmap_centre_btn.setText(f"Pixel: {pix}")

    def _chanmap_shift(self, delta_pix):
        """Shift the locked channel map selection by delta_pix pixels."""
        global wavelengths
        if not self._chanmap_locked or self._chanmap_span is None or wavelengths is None:
            return
        xy = self._chanmap_span.get_xy()
        wav0 = xy[:, 0].min()
        wav1 = xy[:, 0].max()
        half_w = (wav1 - wav0) / 2
        wav_centre = (wav0 + wav1) / 2

        pix_centre = np.argmin(np.abs(wavelengths - wav_centre))
        new_pix = int(np.clip(pix_centre + delta_pix, 0, len(wavelengths) - 1))
        new_centre = wavelengths[new_pix]

        new_wav0 = new_centre - half_w
        new_wav1 = new_centre + half_w
        n = len(xy)
        xs = [new_wav0, new_wav0, new_wav1, new_wav1, new_wav0][:n]
        xy[:, 0] = xs
        self._chanmap_span.set_xy(xy)
        self.spectrum_canvas.draw_idle()
        self._chanmap_update_centre_display()
        self._compute_channel_map(new_wav0, new_wav1)

    def _chanmap_centre_clicked(self):
        """Open a dialog to enter a new centre pixel — avoids keyboard focus stealing."""
        global wavelengths
        if not self._chanmap_locked or self._chanmap_span is None or wavelengths is None:
            return
        xy = self._chanmap_span.get_xy()
        wav0 = xy[:, 0].min()
        wav1 = xy[:, 0].max()
        current_pix = int(np.argmin(np.abs(wavelengths - (wav0 + wav1) / 2)))

        dialog = QDialog(self)
        dialog.setWindowTitle("Set centre pixel")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"Centre pixel (current: {current_pix})"))
        text_box = QLineEdit(str(current_pix))
        text_box.selectAll()
        layout.addWidget(text_box)
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        def _apply():
            try:
                pix = int(np.clip(int(text_box.text()), 0, len(wavelengths) - 1))
            except ValueError:
                return
            half_w = (wav1 - wav0) / 2
            new_centre = wavelengths[pix]
            new_wav0 = new_centre - half_w
            new_wav1 = new_centre + half_w
            n = len(xy)
            xs = [new_wav0, new_wav0, new_wav1, new_wav1, new_wav0][:n]
            xy[:, 0] = xs
            self._chanmap_span.set_xy(xy)
            self.spectrum_canvas.draw_idle()
            self._chanmap_update_centre_display()
            self._compute_channel_map(new_wav0, new_wav1)
            dialog.accept()

        ok_btn.clicked.connect(_apply)
        text_box.returnPressed.connect(_apply)
        cancel_btn.clicked.connect(dialog.reject)
        dialog.exec_()

    def keyReleaseEvent(self, event):
        """Release C/X/V to lock selections; C=line window, X/V=continuum windows."""
        # Handle subtraction window release (X or V)
        if event.key() in (Qt.Key_X, Qt.Key_V) and not event.isAutoRepeat():
            key_char = 'x' if event.key() == Qt.Key_X else 'v'
            sm = self._submap[key_char]
            if sm['active']:
                sm['active'] = False
                if self.spectrum_cursor_pos and sm['start'] is not None:
                    wav_end = self.spectrum_cursor_pos[0]
                    w0 = min(sm['start'], wav_end)
                    w1 = max(sm['start'], wav_end)
                    if abs(w1 - w0) < 1e-6:
                        # Tap = clear this window
                        if sm['span'] is not None:
                            try: sm['span'].remove()
                            except ValueError: pass
                        sm['span'] = None
                        sm['locked'] = False
                        self.spectrum_canvas.draw_idle()
                    else:
                        sm['locked'] = True
                self._compute_channel_map_with_subtraction()

        if event.key() == Qt.Key_C and self._chanmap_active and not event.isAutoRepeat():
            self._chanmap_active = False
            if self.spectrum_cursor_pos and self._chanmap_start is not None:
                wav_end = self.spectrum_cursor_pos[0]
                wav0 = min(self._chanmap_start, wav_end)
                wav1 = max(self._chanmap_start, wav_end)
                if abs(wav1 - wav0) < 1e-6:
                    # Zero-width tap: clear selection and restore white-light
                    if self._chanmap_span is not None:
                        try: self._chanmap_span.remove()
                        except ValueError: pass
                        self._chanmap_span = None
                    self._chanmap_locked = False
                    # Also clear all subtraction windows
                    for _sm in self._submap.values():
                        if _sm['span'] is not None:
                            try: _sm['span'].remove()
                            except ValueError: pass
                        _sm.update({'span': None, 'locked': False, 'active': False, 'start': None})
                    self.chanmap_centre_btn.setText("Pixel: —")
                    self.spectrum_canvas.draw_idle()
                    self.draw_image(FITS_DATA, self._last_cmap, self._last_scale, from_fits=True)
                else:
                    self._chanmap_locked = True
                    self._compute_channel_map(wav0, wav1)
                    self._chanmap_update_centre_display()

    def _compute_channel_map(self, wav0, wav1):
        """Store the current line window limits and trigger the full (subtracted) map."""
        self._chanmap_wav0 = wav0
        self._chanmap_wav1 = wav1
        self._compute_channel_map_with_subtraction()

    def _compute_channel_map_with_subtraction(self):
        """Compute the (optionally continuum-subtracted) channel map and display it.

        Line map  = sum of flux within the C window [wav0, wav1] per spaxel.
        Continuum = mean flux per channel, averaged across any locked X/V windows,
                    then scaled to the number of channels in the C window.
        Result    = Line map − Continuum  (if at least one subtraction window exists)
                  = Line map              (if no subtraction windows are locked)
        """
        global wavelengths
        if self.fits_data is None or wavelengths is None:
            return
        if not hasattr(self, '_chanmap_wav0'):
            return

        wav0, wav1 = self._chanmap_wav0, self._chanmap_wav1
        line_mask = (wavelengths >= wav0) & (wavelengths <= wav1)
        if not np.any(line_mask):
            return

        n_line = line_mask.sum()
        line_map = np.nansum(self.fits_data[line_mask, :, :], axis=0)

        # Gather locked subtraction windows
        cont_estimates = []
        for key_char, sm in self._submap.items():
            if sm['locked'] and sm['span'] is not None:
                xy = sm['span'].get_xy()
                sw0, sw1 = xy[:, 0].min(), xy[:, 0].max()
                cont_mask = (wavelengths >= sw0) & (wavelengths <= sw1)
                n_cont = cont_mask.sum()
                if n_cont > 0:
                    # Mean flux per channel in this window → scale to line width
                    mean_per_chan = np.nansum(self.fits_data[cont_mask, :, :], axis=0) / n_cont
                    cont_estimates.append(mean_per_chan)

        cmap  = getattr(self, '_last_cmap',  'gray')
        scale = getattr(self, '_last_scale', 'linear')

        if cont_estimates:
            # Average the sideband estimates then scale to line window width
            cont_map = np.mean(cont_estimates, axis=0) * n_line
            result = line_map - cont_map
            print(f"Continuum-subtracted map: {n_line} line ch, "
                  f"{len(cont_estimates)} sideband(s)")
        else:
            result = line_map
            print(f"Channel map: {n_line} channels [{wav0:.2f} – {wav1:.2f}]")

        self.draw_image(result, cmap, scale)

    def keyPressEvent(self, event):
        key = event.text().lower()

        global df_cont, df
        """Handle key press events for locking and drawing a line."""
        if event.key() == Qt.Key_L:
            # Toggle the locking state
            self.locked = not self.locked
            print(f"Lock {'enabled' if self.locked else 'disabled'}")
            # Red when locked, orange when free
            if hasattr(self, 'red_rect'):
                self.red_rect.set_edgecolor('#cc2222' if self.locked else '#d7801a')
                self.canvas.draw_idle()
        
        if event.key() == Qt.Key_F:
            print(df)
            # for line in self.spectrum_ax.get_lines():
            #     print(line)
                
            # self.active_line.remove()
            # self.spectrum_canvas.draw_idle()  # Efficient redrawing
            
            
        if event.key() == Qt.Key_D and not event.isAutoRepeat():
            # Start or finish drawing a line in the spectrum plot
            if not self.drawing_line:
                # Start drawing a line: store the first point (cursor position)
                self.line_start = self.spectrum_cursor_pos
                self.drawing_line = True
                
                # Create a new line object in the spectrum plot (only x-coordinates for the start)
                start_x, start_y = self.line_start
                self.current_line, = self.spectrum_ax.plot([start_x, start_x], [start_y, start_y], color='lime', lw=1)
                self.spectrum_canvas.draw()  # Refresh the canvas with the new line
            else:
                # Finish the line at the current cursor position
                self.finalize_line()
                self.drawing_line = False
                # print(f"Line finalized from {self.line_start} to {self.spectrum_cursor_pos}")
                self.slope = (self.spectrum_cursor_pos[1]-self.line_start[1])/(self.spectrum_cursor_pos[0]-self.line_start[0])
                self.intercept = self.line_start[1] - self.slope * self.line_start[0]
                data_cont = {'Continuum Name': ['Continuum'],
                             'x1': [round(self.line_start[0],2)],
                             'x2': [round(self.spectrum_cursor_pos[0],2)],
                             'Slope_0': [self.slope],
                             'Intercept_0': [self.intercept],
                             'Slope_fit': [np.nan],
                             'Intercept_fit': [np.nan],
                             'region_ID': [len(df_cont)],
                             'lineactor': [self.current_line]}
                df_cont_new = pd.DataFrame(data_cont)
                if len(df_cont) == 0:
                    df_cont = df_cont_new
                else:
                    df_cont = pd.concat([df_cont, df_cont_new], ignore_index=True)
                
                # Ensure the Fit Parameters dock is open, then add the new spectral region frame
                self._ensure_fit_params_open_and_add_region()
                # Mark this spaxel with a blue square immediately on continuum placement
                if self.current_spaxel is not None:
                    self._init_guess_spaxel = (int(self.current_spaxel[0]), int(self.current_spaxel[1]))
                    if self._blue_rect is not None:
                        self._blue_rect.set_xy((self._init_guess_spaxel[0] - 0.5,
                                                self._init_guess_spaxel[1] - 0.5))
                        self._blue_rect.set_visible(True)
                        self.canvas.draw_idle()

        if event.key() == Qt.Key_C and not event.isAutoRepeat():
            # Always start a fresh selection immediately on press.
            # If a previous locked box exists, discard it without a separate clear step.
            if not hasattr(self, 'spectrum_cursor_pos') or not self.spectrum_cursor_pos:
                pass  # cursor not over spectrum — do nothing
            else:
                wav = self.spectrum_cursor_pos[0]
                if wav is not None:
                    # Clear any existing span
                    if self._chanmap_span is not None:
                        try: self._chanmap_span.remove()
                        except ValueError: pass
                        self._chanmap_span = None
                    # Start new selection immediately at current cursor wavelength
                    self._chanmap_active = True
                    self._chanmap_locked = False
                    self._chanmap_start  = wav
                    self._chanmap_span = self.spectrum_ax.axvspan(
                        wav, wav,
                        alpha=0.25, color='#4fc3f7', zorder=0
                    )
                    self.spectrum_canvas.draw_idle()

        # X and V keys: subtraction windows (only when channel map is locked)
        if event.key() in (Qt.Key_X, Qt.Key_V) and not event.isAutoRepeat():
            if not self._chanmap_locked:
                pass  # No channel window yet — ignore
            elif hasattr(self, 'spectrum_cursor_pos') and self.spectrum_cursor_pos:
                wav = self.spectrum_cursor_pos[0]
                key_char = 'x' if event.key() == Qt.Key_X else 'v'
                sm = self._submap[key_char]
                # Clear existing span for this key
                if sm['span'] is not None:
                    try: sm['span'].remove()
                    except ValueError: pass
                    sm['span'] = None
                sm['active'] = True
                sm['locked'] = False
                sm['start']  = wav
                sm['span'] = self.spectrum_ax.axvspan(
                    wav, wav, alpha=0.20, color='#ef5350', zorder=0
                )
                self.spectrum_canvas.draw_idle()

        if event.key() == Qt.Key_G and not event.isAutoRepeat():
            if not self.gaussian_active:
                # First press of 'g': Start Gaussian fitting
                x_cursor, y_cursor = self.spectrum_cursor_pos  # Get cursor position
                self.line_region = self.get_line_region(x_cursor)  # Check if cursor is in x1, x2 range
                if self.line_region is not None:
                    # self.gaussian_active = True
                    self.initial_x = x_cursor  # Lock centroid position
                    self.current_sigma = 10
                    self.current_amplitude = y_cursor
                    
                    # Plot initial Gaussian
                    self.start_gaussian_adjustment(self.line_region['region_ID'])
                    # self.draw_gaussian(line_region, self.initial_x, self.current_sigma, self.current_amplitude)
            else:
                # Second press of 'g': Lock the Gaussian
                self.gaussian_active = False
                self.save_gaussian(self.line_region['region_ID'])


    def start_gaussian_adjustment(self, region_ID):
        """Initialize Gaussian drawing when 'G' is pressed, linked to mouse motion."""
        if self.gaussian_active:
            # If Gaussian adjustment is already active, finalize it
            self.gaussian_active = False
            return  
        
        self.gaussian_active = True
        
        # Retrieve the line associated with frame_id
        line_obj = df_cont.loc[df_cont["region_ID"] == region_ID, 'lineactor'].item()
    
        if not isinstance(line_obj, plt.Line2D):
            print(f"Error: No valid line object found.")
            return
    
        # Store the line and initialize Gaussian parameters
        self.active_line = line_obj
        region = df_cont.loc[df_cont["region_ID"] == region_ID]
        self.x1, self.x2 = region["x1"].item(), region["x2"].item()
        self.m, self.b = region["Slope_0"].item(), region["Intercept_0"].item()
    
        # Capture cursor position for Gaussian center
        if self.spectrum_cursor_pos:
            self.gaussian_x0, _ = self.spectrum_cursor_pos  # x0 is fixed at cursor position
    
        # self.gaussian_sigma = 5  # Initial sigma
        # self.gaussian_amplitude = 1  # Initial amplitude
        self.gaussian_sigma = 20 * abs(wavelengths[1]-wavelengths[0])
        self.gaussian_amplitude = 0.1 * (np.diff(self.spectrum_ax.get_ylim())[0])
    
        # Store Gaussian component lines for individual display
        self.gaussian_component_lines = []
    
        # Set flag to indicate active adjustment
        self.gaussian_active = True
    
        # Connect the mouse move event
        self.spectrum_canvas.mpl_connect("motion_notify_event", self.update_gaussian_dynamic)
    
    
    def update_gaussian_dynamic(self, event):
        """Update the Gaussian shape dynamically based on cursor movement."""
        if not self.gaussian_active or event.inaxes != self.spectrum_ax:
            return
    
        # Adjust sigma (horizontal motion) and amplitude (vertical motion)
        dx = event.xdata - self.gaussian_x0
        dy = event.ydata
    
        self.gaussian_sigma = max(0.0001, abs(dx))  # Sigma changes with horizontal movement
    
        # Compute the continuum level at the Gaussian's center position
        y_continuum_x0 = self.m * self.gaussian_x0 + self.b
    
        # Adjust amplitude so that peak of the Gaussian reaches dy
        self.gaussian_amplitude = dy - y_continuum_x0#max(1E-20, dy - y_continuum_x0)
    
        # Generate updated Gaussian + Line function
        x_vals = np.linspace(self.x1, self.x2, 1000)
        y_line = self.m * x_vals + self.b  # The baseline (without Gaussians)
    
        # Recompute all Gaussians from scratch (prevents runaway summing)
        y_gaussian_total = np.zeros_like(x_vals)
    
        # Clear existing individual component lines
        while self.gaussian_component_lines:
            line = self.gaussian_component_lines.pop()
            line.remove()
    
        # Sum over all existing Gaussians in df and plot them
        for _, row in df.iterrows():
            y_gaussian = row["Amp_0"] * np.exp(-((x_vals - row["Centroid_0"]) ** 2) / (2 * row["Sigma_0"] ** 2))
            y_gaussian_total += y_gaussian
            
            # Draw individual Gaussian component (dashed green line)
            component_line, = self.spectrum_ax.plot(x_vals, y_gaussian + y_line, color='#d7801a', linestyle='--', linewidth=1)
            self.gaussian_component_lines.append(component_line)
    
        # If we're actively adjusting a new Gaussian, include and plot it
        y_gaussian_new = self.gaussian_amplitude * np.exp(-((x_vals - self.gaussian_x0) ** 2) / (2 * self.gaussian_sigma ** 2))
        y_gaussian_total += y_gaussian_new
    
        new_component_line, = self.spectrum_ax.plot(x_vals, y_gaussian_new + y_line, color='#d7801a', linestyle='--', linewidth=1)
        self.gaussian_component_lines.append(new_component_line)
    
        # Final y-values for the sum (solid green line)
        y_new = y_line + y_gaussian_total
    
        # Update the existing line object for the summed model
        self.active_line.set_data(x_vals, y_new)
    
        # Redraw dynamically
        self.spectrum_ax.figure.canvas.draw_idle()





    def save_gaussian(self, region_ID):
        global df
        """Save final Gaussian parameters to df_cont"""
        
        if len(df) == 0: 
            line_id = 0
        else:
            line_id = np.max(np.int64(df['Line_ID']))+1
        
        # Auto-identify the line using the line library and source redshift
        _z        = df_obs.loc[0, 'redshift'] if len(df_obs) > 0 else ''
        _ion, _rest_wav = _identify_line(self.gaussian_x0, _z)
        # Build base name: ion + rest wavelength subscript
        if _ion and np.isfinite(_rest_wav):
            _base_name = f'{_ion}_{int(round(_rest_wav))}'
        elif _ion:
            _base_name = _ion
        else:
            _base_name = f'Line {len(df.loc[df["region_ID"]==region_ID])}'
        # If the base name already exists in df, append _b, _c, _d … to disambiguate
        existing_names = set(df['Line_Name'].astype(str).tolist())
        _line_name = _base_name
        if _line_name in existing_names:
            for _letter in 'bcdefghijklmnopqrstuvwxyz':
                _candidate = f'{_base_name}_{_letter}'
                if _candidate not in existing_names:
                    _line_name = _candidate
                    break
        _rest_wav_store = _rest_wav if np.isfinite(_rest_wav) else np.nan

        df_new = pd.DataFrame({'Line_ID': [line_id],
                'Line_Name': [_line_name],
                'SNR': [0],
                'Rest Wavelength': [_rest_wav_store],
                'Amp_0': [self.gaussian_amplitude],
                'Amp_0_lowlim': [0],
                'Amp_0_highlim': [np.inf],
                'Centroid_0': [self.gaussian_x0],
                'Centroid_0_lowlim': [-np.inf],
                'Centroid_0_highlim': [np.inf],
                'Sigma_0': [self.gaussian_sigma],
                'Sigma_0_lowlim': [0],
                'Sigma_0_highlim': [np.inf],
                'Amp_fit': [np.nan],
                'Centroid_fit': [np.nan],
                'Sigma_fit': [np.nan],
                'region_ID': [region_ID],
                'curveactor': [self.active_line]})
        
        
        
        
        df = pd.concat([df,df_new])

        # Mark this spaxel with a blue square on the cube viewport
        if self.current_spaxel is not None:
            self._init_guess_spaxel = (int(self.current_spaxel[0]), int(self.current_spaxel[1]))
            if self._blue_rect is not None:
                self._blue_rect.set_xy((self._init_guess_spaxel[0] - 0.5,
                                        self._init_guess_spaxel[1] - 0.5))
                self._blue_rect.set_visible(True)
                self.canvas.draw_idle()

        self.fit_params_window.on_addline_button_click(df,frame_id=region_ID)
        
        
        
    def rebuild_plot(self,region_ID, from_file, show_init, show_fit, x, y):
        """used after deleting emission line, adding new line, or editing line"""
        global df, df_cont
        
        region = df_cont.loc[np.int64(df_cont["region_ID"]) == np.int64(region_ID)]
        # '''
        
        if show_init == True:
            self.x1, self.x2 = region["x1"].item(), region["x2"].item()
            self.m, self.b = region["Slope_0"].item(), region["Intercept_0"].item()
            
            self.x1 = np.float64(self.x1)
            self.x2 = np.float64(self.x2)
            self.m = np.float64(self.m)
            self.b = np.float64(self.b)
            
            # Generate updated Gaussian + Line function
            x_vals = np.linspace(self.x1, self.x2, 1000)
            y_line = self.m * x_vals + self.b  # The baseline (without Gaussians)
            
            # '''
            gaussians = df.loc[np.int64(df["region_ID"]) == region_ID]
    
            # Recompute all Gaussians from scratch (prevents runaway summing)
            y_gaussian_total = np.zeros_like(x_vals)
        
            # Sum over all existing Gaussians in df
            i = 0
            for _, row in gaussians.iterrows():
                row['Amp_0'] = np.float64(row['Amp_0'])
                row['Centroid_0'] = np.float64(row['Centroid_0'])
                row['Sigma_0'] = np.float64(row['Sigma_0'])
                gaussian = row["Amp_0"] * np.exp(-((x_vals - row["Centroid_0"]) ** 2) / (2 * row["Sigma_0"] ** 2))
                new_curve, = self.spectrum_ax.plot(x_vals, gaussian+y_line, color='#d7801a', lw=1,linestyle='--')
                df.loc[(np.int64(df['region_ID']==region_ID)) & (df['Line_ID'] == i), 'curveactor'] = new_curve
                y_gaussian_total += gaussian
                i = i+1
                
        
            # Final y-values
            y_new = y_line + y_gaussian_total
        
            # Update the existing line object
            if from_file == False:
                line_obj = df_cont.loc[np.int64(df_cont["region_ID"]) == region_ID, 'lineactor'].item()
                line_obj.set_data(x_vals, y_new)
                
            
            new_line, = self.spectrum_ax.plot(x_vals, y_new, color='lime', lw=1)
            df_cont.loc[np.int64(df_cont["region_ID"]) == region_ID, 'lineactor'] = new_line  # Update reference


        if (show_fit == True) and (len(df_fit) > 0):
            region = df_fit.loc[(np.int64(df_fit['spaxel_x'])==x) &
                                 (np.int64(df_fit['spaxel_y'])==y) &
                                 (np.int64(df_fit['region_ID'])==np.int64(region_ID))]
            
            
            self.x1, self.x2 = region.iloc[0]["cont_region"+str(region_ID+1)+"_x_start"].item(), region.iloc[0]["cont_region"+str(region_ID+1)+"_x_end"].item()
            self.m, self.b = region.iloc[0]["cont_region"+str(region_ID+1)+"_slope_fit"].item(), region.iloc[0]["cont_region"+str(region_ID+1)+"_intercept_fit"].item()
            
            self.x1 = np.float64(self.x1)
            self.x2 = np.float64(self.x2)
            self.m = np.float64(self.m)
            self.b = np.float64(self.b)
            
            # Generate updated Gaussian + Line function
            x_vals = np.linspace(self.x1, self.x2, 1000)
            y_line = self.m * x_vals + self.b  # The baseline (without Gaussians)
            
            # '''
            # gaussians = df.loc[np.int64(df["region_ID"]) == region_ID]
    
            # Recompute all Gaussians from scratch (prevents runaway summing)
            y_gaussian_total = np.zeros_like(x_vals)
        
            # Sum over all existing Gaussians in df
            for _, row in region.iterrows():
                row['amp_fit'] = np.float64(row['amp_fit'])
                row['cen_fit'] = np.float64(row['cen_fit'])
                row['sigma_fit'] = np.float64(row['sigma_fit'])
                plotcolor = row['color']
                y_gaussian = row["amp_fit"] * np.exp(-((x_vals - row["cen_fit"]) ** 2) / (2 * row["sigma_fit"] ** 2))
                y_gaussian_total += y_gaussian
                fit_component, = self.spectrum_ax.plot(x_vals,y_line+y_gaussian,lw=1,linestyle='--',color=plotcolor)
                
            
            # Final y-values
            y_new = y_line + y_gaussian_total
        
            # Update the existing line object
            if from_file == False:
                line_obj = df_cont.loc[np.int64(df_cont["region_ID"]) == np.int64(region_ID), 'lineactor'].item()
                line_obj.set_data(x_vals, y_new)
                
            
            if self.is_1d_spectrum == False:
                new_line, = self.spectrum_ax.plot(x_vals, y_new, color='red', lw=0.5, label=f'rChi2 = {row["rchisq"]}')
            else:
                new_line, = self.spectrum_ax.plot(x_vals, y_new, color='red', lw=0.5)
            df_cont.loc[np.int64(df_cont["region_ID"]) == region_ID, 'lineactor'] = new_line  # Update reference
            
            self.spectrum_ax.legend()

            # ── Push fitted values into the parameter buttons ─────────────
            # df_fit uses 'LineID' and df uses 'Line_ID'; match on both.
            if self.fit_params_window is not None:
                fp = self.fit_params_window
                for _, frow in region.iterrows():
                    lid = np.int64(frow['LineID'])
                    # Button key prefix is the Line_ID from df
                    prefix = str(lid)
                    _map = {
                        f'{prefix}~Amp_fit':      frow.get('amp_fit', np.nan),
                        f'{prefix}~Centroid_fit': frow.get('cen_fit', np.nan),
                        f'{prefix}~v_fit':        frow.get('vel_fit', np.nan),
                        f'{prefix}~Sigma_fit':    frow.get('sigma_fit', np.nan),
                    }
                    for btn_name, val in _map.items():
                        key = (np.int64(region_ID), btn_name)
                        if key in fp.buttons_dict:
                            fp.buttons_dict[key].setText(
                                _fmt(val) if np.isfinite(float(val)) else "—"
                            )

        # Redraw dynamically
        self.spectrum_ax.figure.canvas.draw_idle()
        
        
        
    def get_line_region(self, x_cursor):
        """Check if cursor is within x1, x2 of any existing lines"""
        for index, row in df_cont.iterrows():
            if row["x1"] <= x_cursor <= row["x2"]:
                return row  # Return the matching line's data
        return None

                
    def draw_image(self, data, cmap, scale, from_fits=False):
        """Generates and displays the white-light image in the left panel."""
        self._last_cmap  = cmap
        self._last_scale = scale
        self._last_from_fits = from_fits
        # Only update _last_data when called from a real data load (not from
        # _cube_redraw_transformed, which passes already-transformed 2D data).
        # _cube_redraw_transformed sets _applying_transform=True before calling.
        if not getattr(self, '_applying_transform', False):
            self._last_data = data
            # Cache the original 2D shape for coord inverse transform
            if from_fits and hasattr(data, 'ndim') and data.ndim == 3:
                self._cube_orig_shape = (data.shape[1], data.shape[2])  # (ny, nx)
            elif not from_fits and hasattr(data, 'ndim') and data.ndim == 2:
                self._cube_orig_shape = data.shape
        global npix_x, npix_y
        if from_fits:
            image = np.nansum(data, axis=0).astype(np.float64)
            npix_x = np.shape(image)[0]
            npix_y = np.shape(image)[1]
        else:
            image = np.array(data, dtype=np.float64)

        # ── Apply transfer function (scale) ───────────────────────────
        # Prefer toolbar setting; fall back to passed-in scale arg
        tf = getattr(self, 'scale_combo', None)
        tf_name = tf.currentText() if tf else scale

        # Shift to positive before non-linear transforms
        min_val = np.nanmin(image)
        if min_val < 0:
            image = image - min_val

        if tf_name == 'Log':
            image = np.log10(np.where(image > 0, image, np.nan))
        elif tf_name == 'Square root':
            image = np.sqrt(np.where(image >= 0, image, np.nan))
        elif tf_name == 'Squared':
            image = image ** 2
        elif tf_name == 'Asinh':
            scale_factor = np.nanpercentile(image, 99) or 1.0
            image = np.arcsinh(image / scale_factor)
        # Linear: no transform

        # ── Apply stretch (clip level) ────────────────────────────────
        st = getattr(self, 'stretch_combo', None)
        st_name = st.currentText() if st else 'minmax'

        stretch_map = {
            'minmax': (0, 100),
            '99.9%':  (0.05, 99.95),
            '99.5%':  (0.25, 99.75),
            '99%':    (0.5,  99.5),
            '98%':    (1,    99),
            '95%':    (2.5,  97.5),
        }
        if st_name == 'manual':
            vmin = getattr(self, '_manual_vmin', np.nanmin(image))
            vmax = getattr(self, '_manual_vmax', np.nanmax(image))
        else:
            lo_pct, hi_pct = stretch_map.get(st_name, (0.5, 99.5))  # default 99% clip
            finite = image[np.isfinite(image)]
            if len(finite) == 0:
                vmin, vmax = 0, 1
            else:
                vmin = np.percentile(finite, lo_pct)
                vmax = np.percentile(finite, hi_pct)

        self.canvas.figure.clear()
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        apply_mpl_qss_style(self.canvas.figure, self.ax, None)
        self.ax.grid(False)  # No grid on the cube image

        # Initialize red rectangle but keep it hidden initially
        self.red_rect = patches.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='#d7801a', facecolor='none', visible=False)
        self.ax.add_patch(self.red_rect)
        self._blue_rect = patches.Rectangle((0, 0), 1, 1, linewidth=2.0, edgecolor='#4fc3f7', facecolor='none', visible=False)
        self.ax.add_patch(self._blue_rect)
        # Restore blue rect position if an init-guess spaxel was previously set
        if getattr(self, '_init_guess_spaxel', None) is not None:
            _ix, _iy = self._init_guess_spaxel
            self._blue_rect.set_xy((_ix - 0.5, _iy - 0.5))
            self._blue_rect.set_visible(True)

        # Overlay text for spaxel info (top-left corner of the image axes)
        self.spaxel_info_text = self.ax.text(
            0.01, 0.99, '',
            transform=self.ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontsize=8, color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor='none')
        )
        # Restore background image overlay if one was loaded
        if getattr(self, '_bkg_data', None) is not None:
            self._bkg_artist = None  # force redraw since axes were cleared
            self._draw_bkg_overlay()

        self.canvas.draw()





    def get_spectrum_at_spaxel(self, x, y):
        """Fetch the 1D spectrum corresponding to the given spaxel (x, y)."""
        spectrum = FITS_DATA[:, y, x]
        return spectrum


    def finalize_line(self):
        """Finalize the drawn line when the 'd' key is pressed again."""
        if self.drawing_line and self.line_start is not None:
            # Get the final position of the cursor in the spectrum plot
            end_x, end_y = self.spectrum_cursor_pos
            self.current_line.set_xdata([self.line_start[0], end_x])
            self.current_line.set_ydata([self.line_start[1], end_y])
            
            # Redraw the canvas to reflect the final line
            self.spectrum_canvas.draw()
    
    def update_line(self, x1, x2):
        """Update the line in the plot while keeping y1 and y2 fixed."""
        if hasattr(self, 'current_line') and hasattr(self, 'line_start') and hasattr(self, 'spectrum_cursor_pos'):
            y1 = self.line_start[1]  # Keep the original y1
            y2 = self.spectrum_cursor_pos[1]  # Keep the original y2
            
            # Recalculate the slope and intercept to maintain (y1, y2)
            if x1 != x2:  # Avoid division by zero
                self.slope = (y2 - y1) / (x2 - x1)
                self.intercept = y1 - self.slope * x1
            else:
                print("Warning: x1 and x2 are the same. Vertical line assumed.")
                self.slope = float('inf')  # Indicates a vertical line
                self.intercept = None  # Not defined for vertical lines
    
            self.current_line.set_data([x1, x2], [y1, y2])  # Update with new x1, x2
            self.spectrum_canvas.draw()  # Refresh the plot
            # print(f"Updated line: x1={x1}, y1={y1}, x2={x2}, y2={y2}, slope={self.slope}, intercept={self.intercept}")

    
    def update_line_with_slope_intercept(self, x1, x2, slope, intercept):
        """Update the line in the plot based on the updated slope and intercept."""
        if hasattr(self, 'current_line'):
            y1 = slope * x1 + intercept
            y2 = slope * x2 + intercept
            print(y1,y2)
            self.current_line.set_data([x1, x2], [y1, y2])
            self.spectrum_canvas.draw()  # Refresh the plot
            print(f"Redrawing line with updated slope={slope} and intercept={intercept}")
        
        # Store the slope and intercept for future updates
        self.slope = slope
        self.intercept = intercept






    def create_menu_bar_VisualizerWindow(self):
        """Creates and returns the menu bar for ViewerWindow"""
        menu_bar = QMenuBar(self)

        # File Menu
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Import (Cmd+I)", self)
        open_action.setShortcut(QKeySequence("Cmd+O" if sys.platform == "darwin" else "Ctrl+O"))
        save_action = QAction("Save (Cmd+S)", self)
        save_action.setShortcut(QKeySequence("Cmd+S" if sys.platform == "darwin" else "Ctrl+S"))
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # View Menu
        view_menu = menu_bar.addMenu("View")
        view_menu.addAction(QAction("Zoom In", self))
        view_menu.addAction(QAction("Zoom Out", self))

        # Help Menu
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction(QAction("About", self))

        return menu_bar

    def showEvent(self, event):
        """Ensure ViewerWindow always has its own menu bar when focused"""
        self.setMenuBar(self.create_menu_bar_VisualizerWindow())
        super().showEvent(event)


    def _ensure_fit_params_open_and_add_region(self):
        """Called after each interactive continuum draw (D-key).
        
        Guarantees:
        1. The Fit Parameters dock is created and visible (never accidentally closes it).
        2. The newly-added spectral region frame is always appended to the panel.
        """
        new_id = len(df_cont) - 1  # index of the region just added

        if self.fit_params_dock is None:
            # First ever open: create the dock. FitParamsWindow.__init__ will
            # call add_spectral_frame for region 0 via its own init logic.
            self.toggle_fit_params_dock()
            # __init__ only adds region 0; if somehow more regions exist, add them
            for extra_id in range(1, len(df_cont)):
                self.fit_params_window.add_spectral_frame(
                    f"Spectral Region {extra_id + 1}", df_cont, df, ID=extra_id, addframe=False
                )
        else:
            # Dock already exists — make sure it is visible (never hide it here)
            if not self.fit_params_dock.isVisible():
                self.fit_params_dock.show()
                self.resizeDocks([self.fit_params_dock], [self.width() // 2], Qt.Horizontal)
                self.open_fit_params_button.setText("Fit Params \u25c0")
            # Always add the new region frame explicitly
            self.fit_params_window.add_spectral_frame(
                f"Spectral Region {new_id + 1}", df_cont, df, ID=new_id, addframe=False
            )

    def toggle_fit_params_dock(self):
        """Toggle the Fit Parameters panel between docked and hidden states.
        
        First call: creates FitParamsWindow and docks it on the right side of
        ViewerWindow as a QDockWidget.  Subsequent calls show/hide the dock so
        the viewer layout is never obscured by a floating window.
        """
        # --- First-time creation ---
        if self.fit_params_dock is None:
            # Create the FitParamsWindow (still a QMainWindow; we just embed it)
            self.fit_params_window = FitParamsWindow(
                viewer_window=self, df=df, df_cont=df_cont, current_line=''
            )

            # Wrap it in a QDockWidget
            self.fit_params_dock = QDockWidget("Fit Parameters", self)
            self.fit_params_dock.setObjectName("FitParamsDock")
            self.fit_params_dock.setAllowedAreas(
                Qt.BottomDockWidgetArea | Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
            )
            # Embed the central widget of FitParamsWindow (the scroll area etc.)
            self.fit_params_dock.setWidget(self.fit_params_window.centralWidget())
            self.fit_params_dock.setFeatures(
                QDockWidget.DockWidgetMovable |
                QDockWidget.DockWidgetFloatable |
                QDockWidget.DockWidgetClosable
            )
            # Keep the button label in sync when the user closes the dock via its X
            self.fit_params_dock.visibilityChanged.connect(self._on_dock_visibility_changed)

            self.addDockWidget(Qt.BottomDockWidgetArea, self.fit_params_dock)
            # Size the dock to ~40% of the window height
            self.resizeDocks([self.fit_params_dock], [self.height() * 2 // 5], Qt.Vertical)
            self.open_fit_params_button.setText("▲  Fit Parameters")
            return

        # --- Toggle visibility on subsequent calls ---
        if self.fit_params_dock.isVisible():
            self.fit_params_dock.hide()
            self.open_fit_params_button.setText("▼  Fit Parameters")
        else:
            self.fit_params_dock.show()
            # Re-apply height whenever the dock is shown again
            self.resizeDocks([self.fit_params_dock], [self.height() * 2 // 5], Qt.Vertical)
            self.open_fit_params_button.setText("▲  Fit Parameters")

    def _on_dock_visibility_changed(self, visible):
        """Keep the toggle button label in sync when dock is closed via its own X button."""
        if visible:
            self.open_fit_params_button.setText("▲  Fit Parameters")
        else:
            self.open_fit_params_button.setText("▼  Fit Parameters")

    def open_fit_params_window(self):
        """Opens the FitParamsWindow when the button is clicked.
        
        Kept for backwards-compatibility (e.g. calls from other code paths).
        Delegates to the dock toggle so behaviour is consistent.
        """
        self.toggle_fit_params_dock()

    def closeEvent(self, event):
        """Override closeEvent to ensure the console is still active after closing the window"""
        app = QApplication.instance()
        if app:
            app.quit()  # Quit the QApplication when the window is closed
        event.accept()  # Accept the event and close the window
        

class FitParamsWindow(QtWidgets.QMainWindow):
    buttons_dict = {}

    def __init__(self, viewer_window, df_cont, df, current_line):
        super().__init__()
        self.viewer_window = viewer_window
        df_cont = df_cont
        df = df
        self.fitloaded = False
        self.current_line = current_line
        self.setWindowTitle("Fit Parameters")
        
        # self.create_menu_bar_FitParamsWindow()  # Add menu bar
        # Assign the menu bar explicitly to this window
        self.setMenuBar(self.create_menu_bar_FitParamsWindow())

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        # Scroll Area
        self.scroll_area = QScrollArea(self.main_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        layout.addWidget(self.scroll_area)

        # Container for frames inside the scroll area
        self.scroll_container = QWidget()
        # Vertical layout: obs toolbar at top, then one spectral region card per row
        self.outer_scroll_layout = QVBoxLayout(self.scroll_container)
        self.outer_scroll_layout.setSpacing(8)
        self.outer_scroll_layout.setContentsMargins(4, 4, 4, 4)
        # scroll_layout alias points to same layout for compatibility
        self.scroll_layout = self.outer_scroll_layout
        self.scroll_container.setLayout(self.outer_scroll_layout)
        self.scroll_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.scroll_area.setWidget(self.scroll_container)

        self._build_obs_and_fit_toolbar(df_obs)
        
        if len(df_cont) == 0:
            self.spectral_count = 0
        else:
            self.spectral_count = 1
            self.add_spectral_frame("Spectral Region 1", df_cont, df, ID=0, addframe=False)

        self.setGeometry(100, 100, 1200, 400)
        # Note: visibility is now managed by the QDockWidget in ViewerWindow.

    def create_menu_bar_FitParamsWindow(self):
        """Creates the menu bar for this window"""
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)  # Assign it to the window

        # File Menu
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open (Cmd+O)", self)
        open_action.setShortcut(QKeySequence("Cmd+O" if sys.platform == "darwin" else "Ctrl+O"))
        save_action = QAction("Save (Cmd+S)", self)
        save_action.setShortcut(QKeySequence("Cmd+S" if sys.platform == "darwin" else "Ctrl+S"))
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # Edit Menu
        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addAction(QAction("Undo", self))
        edit_menu.addAction(QAction("Redo", self))

        # View Menu
        view_menu = menu_bar.addMenu("View")
        view_menu.addAction(QAction("Zoom In", self))
        view_menu.addAction(QAction("Zoom Out", self))

        # Help Menu
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction(QAction("About", self))

        return menu_bar  # Return the menu bar

    def add_continuum_buttons(self, regionID, grid_layout):
        # Add Labels (continuum button labels)
        for col, col_name in enumerate(df_cont.columns):
            if (col_name != 'region_ID') & (col_name != 'lineactor'):
                label = QLabel(col_name)
                # label.setStyleSheet("""
                #     font-family: "Segoe UI";
                #     font-size: 12pt;
                #     border: none;
                #     padding-right: 10px;
                #     padding-left: 10px;
                #     padding-top: 5px;
                #     padding-bottom: 5px;
                #     background-color: snow;
                #     color: black;
                #     font: bold;
                #     width: 64px;
                # """)
                grid_layout.addWidget(label, 0, col)
            
        # Add Data (continuum buttons)
        for row in range(df_cont.shape[0]):
           for col, col_name in enumerate(df_cont.columns):
               if (col_name != 'region_ID') & (col_name != 'lineactor'):
                   if np.int64(df_cont.iloc[row,7]) == np.int64(regionID):
                       button_name = 'continuum~'+col_name
                       # button_text = str(df_cont.iloc[row, col]) if 'fit' not in col_name.lower() else ''
                       if (col_name == 'Continuum Name') or (type(df_cont.iloc[row, col]) == str):
                             button_text = str(df_cont.iloc[row, col]) if 'fit' not in col_name.lower() else ''
                       else:
                             button_text = _fmt(df_cont.iloc[row, col]) if 'fit' not in col_name.lower() else ''
                       button = FrameButton(button_text, row, col, regionID, button_name)
                       # button.setStyleSheet("""
                       #     font-family: "Segoe UI";
                       #     font-size: 10pt;
                       #     border: 2px solid;
                       #     border-color: steelblue;
                       #     border-radius: 5px;
                       #     padding-right: 10px;
                       #     padding-left: 10px;
                       #     padding-top: 5px;
                       #     padding-bottom: 5px;
                       #     background-color: lightskyblue;
                       #     highlight-color; cyan;
                       #     color: k;
                       #     font: bold;
                       #     width: 64px;
                       # """)
                       
                       if col >= 5:  # Check the 'fit' column in df_cont
                           # button.setStyleSheet("""
                           #     font-family: "Segoe UI";
                           #     font-size: 12pt;
                           #     border: 2px solid;
                           #     border-color: black;
                           #     border-radius: 5px;
                           #     padding-right: 10px;
                           #     padding-left: 10px;
                           #     padding-top: 5px;
                           #     padding-bottom: 5px;
                           #     background-color: darkgrey;
                           #     color: white;
                           #     font: bold;
                           #     width: 64px;
                           # """)
                           button.setDisabled(True)  # Disable the button
                       else:
                           button.clicked.connect(partial(self.on_button_click, data_frame=df_cont, frame_id=regionID, button_name=button_name))
                       
                       grid_layout.addWidget(button, 1, col)
       
                       # Store the button reference in buttons_dict by row/col
                       self.buttons_dict[(regionID, button_name)] = button
                       
                       if col == 6:
                            button_name = 'continuum_delete'
                            delete_region_button = FrameButton('x', row, col, regionID, button_name)
                            delete_region_button.setStyleSheet("""
                                background-color: lightcoral; color: black;
                                color: k;
                            """)
                            
                            delete_region_button.clicked.connect(partial(self.on_deleteregion_button_click, frame_id=regionID))

                            grid_layout.addWidget(delete_region_button, 1, col+1)
                            # Store the button reference in buttons_dict by row/col
                            self.buttons_dict[(regionID, button_name)] = delete_region_button
                       
                       
    def add_spectral_lines_button_header(self, grid_layout):
        button_columns = [
            'Line Name', 
            'SNR',
            '<i>&lambda;</i><sub>rest</sub>', 
            '<i>f</i><sub>&lambda;,0</sub>', 
            '<i>f</i><sub>&lambda;,0,min</sub>', 
            '<i>f</i><sub>&lambda;,0,max</sub>', 
            '<i>&lambda;</i><sub>obs,0</sub>', 
            '<i>&lambda;</i><sub>obs,0,min</sub>',
            '<i>&lambda;</i><sub>obs,0,max</sub>',
            '&sigma;<sub>0</sub>', 
            '&sigma;<sub>0,min</sub>', 
            '&sigma;<sub>0,max</sub>', 
            '<i>f</i><sub>&lambda;,fit</sub>', 
            '<i>&lambda;</i><sub>obs,fit</sub>', 
            'v<sub>fit</sub>', 
            '&sigma;<sub>fit</sub>'
        ]
    
        for col, col_name in enumerate(button_columns):
            label = QLabel(col_name)
            label.setTextFormat(Qt.RichText)  # Enable HTML formatting
            if col == 0:
                label.setStyleSheet("""
                    font-size: 12pt;
                """)
            else:
                label.setStyleSheet("""
                    font-size: 12pt;
                    padding-left: 10px;
                """)
            # label.setStyleSheet("""
            #     font-family: "Segoe UI";
            #     font-size: 12pt;
            #     border: none;
            #     padding-right: 10px;
            #     padding-left: 10px;
            #     padding-top: 5px;
            #     padding-bottom: 5px;
            #     background-color: snow;
            #     color: black;
            #     font: bold;
            #     width: 64px;
            # """)
            grid_layout.addWidget(label, 2, col)



    def add_spectral_lines(self, regionID, grid_layout):
        global df_cont, df
        # Add Data (spectral line buttons)
        df_region = df.loc[np.int64(df['region_ID']) == np.int64(regionID)]
        
        for row in range(df_region.shape[0]):
            button_columns = ['Line_Name', 'SNR','Rest Wavelength', 
                              'Amp_0', 'Amp_0_lowlim', 'Amp_0_highlim',
                              'Centroid_0', 'Centroid_0_lowlim', 'Centroid_0_highlim',
                              'Sigma_0','Sigma_0_lowlim','Sigma_0_highlim',
                              'Amp_fit', 'Centroid_fit', 'v_fit', 'Sigma_fit']
            
            main_row_index = 3 + row * 2  # Reserve extra row space
            limits_row_index = main_row_index + 1  # Place limit buttons in a separate row
            
            button_height = 24  # Standardized button height
            
            for col, col_name in enumerate(button_columns):
                # print(type(df_region.iloc[row][col_name]),df_region.iloc[row][col_name])
                button_name = str(np.int64(df_region.iloc[row]['Line_ID']))+'~'+str(col_name)
                if col_name in ['Rest Wavelength', 'Amp_0', 'Centroid_0', 'Sigma_0']:
                    df_region.iloc[row][col_name] = np.float64(df_region.iloc[row][col_name])
                    button_text = _fmt(df_region.iloc[row][col_name])
                else:
                    
                    if ('fit' in col_name.lower()) | ('SNR' in col_name) | ('Rest' in col_name):
                        button_text = ''
                    else: 
                        button_text = str(df_region.iloc[row][col_name])
                        
                # button_text = str(df_region.iloc[row][col_name]) if 'fit' not in col_name.lower() else ''
                button = FrameButton(button_text, row, col, regionID, button_name)
                # button.setStyleSheet("""
                #     font-family: "Segoe UI";
                #     font-size: 8pt;
                #     border: 2px solid;
                #     border-color: steelblue;
                #     border-radius: 5px;
                #     padding: 5px 10px;
                #     background-color: lightskyblue;
                #     color: black;
                #     font: bold;
                # """)
                if 'fit' in col_name:  # Check the 'fit' column
                    # button.setStyleSheet("""
                    #     font-family: "Segoe UI";
                    #     font-size: 12pt;
                    #     border: 2px solid;
                    #     border-color: black;
                    #     border-radius: 5px;
                    #     padding: 5px 10px;
                    #     background-color: darkgrey;
                    #     color: white;
                    #     font: bold;
                    # """)
                    # button.setDisabled(True)  # Disable the button
                    button.clicked.connect(partial(self.on_fit_button_click, frame_id=regionID, button_name=button_name))
                else:
                    button.clicked.connect(partial(self.on_button_click, data_frame=df, frame_id=regionID, button_name=button_name))
    
                grid_layout.addWidget(button, main_row_index, col)
                
                # Store button reference
                self.buttons_dict[(regionID, button_name)] = button
                
                line_id = np.float64(df_region.iloc[row]['Line_ID'])
                line_id = np.int64(line_id)

            # Delete button at the rightmost position of each line row
            del_line_btn = FrameButton('x', row, len(button_columns), regionID, f'del_line_{line_id}')
            del_line_btn.setFixedHeight(24)
            del_line_btn.setStyleSheet("background-color: lightcoral; color: black;")
            del_line_btn.clicked.connect(partial(self.on_deleteline_button_click, frame_id=regionID, line_id=line_id))
            grid_layout.addWidget(del_line_btn, main_row_index, len(button_columns))
            self.buttons_dict[(regionID, f'del_line_{line_id}')] = del_line_btn
                
    def on_fit_button_click(self, frame_id, button_name):
        print(f'frame ID: {frame_id}, button name: {button_name}')

        if len(df_fit) > 0:
            print(df_fit.iloc[0])
            gaussian_number = np.float64(button_name.split('~')[0])
            print(f'gaussian number = {gaussian_number}')
            # Filter df_fit for the specified line_name and region_ID
            filtered_df = df_fit[(df_fit['LineID'] == np.float64(gaussian_number)) & (df_fit['region_ID'] == frame_id)]

            # Convert spaxel_x and spaxel_y to integers for indexing
            filtered_df['spaxel_x'] = filtered_df['spaxel_x'].astype(int)
            filtered_df['spaxel_y'] = filtered_df['spaxel_y'].astype(int)
            
            # Determine image dimensions
            x_max = filtered_df['spaxel_x'].max() + 1
            y_max = filtered_df['spaxel_y'].max() + 1
            
            # Create an empty image array
            image_array = np.full((y_max, x_max), np.nan)  # Using NaN for unfilled values
            
            
            
            
            # Mapping of button key parts to the column names in df_fit
            key_to_column_mapping = {
                'continuum~Slope_fit': 'slope_fit',
                'continuum~Intercept_fit': 'intercept_fit',
                'Line_Name': 'LineName',
                'Amp_fit': 'amp_fit',
                'Centroid_fit': 'cen_fit',
                'Sigma_fit': 'sigma_fit',
                'v_fit': 'vel_fit'
            }
            
            df_param = key_to_column_mapping[button_name.split('~')[1]]
            
            
            print(f'df_param: {df_param}')
            # Populate the image array with amp_fit values
            for _, row in filtered_df.iterrows():
                # print(row[df_param])
                image_array[row['spaxel_y'], row['spaxel_x']] = row[df_param]
                

            # print('shape of img array'+str(np.shape(image_array)))

            if df_param == 'amp_fit':
                cmap = 'viridis'
                # Use linear scale: log10 of ~1e-18 values collapses to NaN after clipping
                scale = 'linear'
            if df_param == 'cen_fit': 
                cmap='bwr'
                scale='linear'
            if df_param == 'vel_fit': 
                cmap='bwr'
                scale='linear'
            if df_param == 'sigma_fit': 
                cmap='plasma'
                scale='linear'
            
            # fig,ax=plt.subplots()
            # ax.imshow(image_array,origin='lower',aspect='auto',cmap=cmap)
            self.viewer_window.draw_image(image_array, cmap=cmap, scale=scale,from_fits=False)


    def save_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
    
                # Convert region_ID to int64
                df_cont['region_ID'] = np.int64(df_cont['region_ID'])
                df['region_ID'] = np.int64(df['region_ID'])
    
                # Write df_obs (Observation parameters)
                writer.writerow(df_obs.columns)  # Write header
                for row in df_obs.itertuples(index=False, name=None):
                    writer.writerow(row)  # Flatten and write each row
    
                # Insert an empty row to separate the sections
                writer.writerow([])
                
                writer.writerow(['wavelength scale factor', 'flux scale factor'])
                
                writer.writerow([self.viewer_window.WLscalefactor,self.viewer_window.fluxscalefactor])
    
                # Insert an empty row to separate the sections
                writer.writerow([])
                
                # Write df_cont (Continuum parameters)
                writer.writerow(df_cont.columns)  # Header for df_cont
                writer.writerows(df_cont.values)  # Data for df_cont
    
                # Insert an empty row to separate the sections
                writer.writerow([])
    
                # Write df header (Emission line parameters)
                writer.writerow(df.columns)  # Header for df
    
                # Sort df by 'region_ID' before writing
                df_sorted = df.sort_values(by='region_ID', ascending=True)
                writer.writerows(df_sorted.values)  # Data for df
    
            print(f"File saved: {file_path}")

    
    def open_file(self):
        global df, df_cont, df_obs
        """Open a CSV file and load it into the dataframes"""
        print('Opening file...')
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    
        # Clear existing plot actors
        for frame_id in np.unique(df_cont['region_ID']):
            df_cont.loc[(df_cont["region_ID"] == frame_id)]['lineactor'].item().remove()
    
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    reader = csv.reader(file)
                    rows = list(reader)
    
                # Identify section indices (empty rows)
                empty_indices = [i for i, row in enumerate(rows) if not any(row)]
    
                if len(empty_indices) < 2:
                    print("Error: Invalid file format.")
                    return
    
                # Split the CSV data into three sections
                obs_data = rows[:empty_indices[0]]
                scale_data = rows[empty_indices[0]+1:empty_indices[1]]
                cont_data = rows[empty_indices[1]+1:empty_indices[2]]
                line_data = rows[empty_indices[2]+1:]
    
                # Load Observation Data (df_obs)
                df_obs_columns = obs_data[0]
                df_obs_values = obs_data[1:]
                df_obs = pd.DataFrame(df_obs_values, columns=df_obs_columns)
    
                # Ensure numeric columns are properly cast
                for col in ['redshift', 'resolvingpower']:
                    if col in df_obs.columns:
                        df_obs[col] = pd.to_numeric(df_obs[col], errors='coerce')
    
                # Load Continuum Data (df_cont)
                df_cont_columns = cont_data[0]
                df_cont_values = cont_data[1:]
                df_cont = pd.DataFrame(df_cont_values, columns=df_cont_columns)
    
                # Convert data types for df_cont
                df_cont['x1'] = np.float64(df_cont['x1'])
                df_cont['x2'] = np.float64(df_cont['x2'])
                df_cont['Slope_0'] = np.float64(df_cont['Slope_0'])
                df_cont['Intercept_0'] = np.float64(df_cont['Intercept_0'])
                df_cont['Slope_fit'] = pd.to_numeric(df_cont['Slope_fit'], errors='coerce')
                df_cont['Intercept_fit'] = pd.to_numeric(df_cont['Intercept_fit'], errors='coerce')
                df_cont['region_ID'] = np.int64(df_cont['region_ID'])
    
                # Load Emission Line Data (df)
                df_columns = line_data[0]
                df_values = line_data[1:]
                df = pd.DataFrame(df_values, columns=df_columns)
    
                # Convert numeric columns in df
                float_columns = [
                    'spaxel_x', 'spaxel_y', 'Line_ID', 'Rest Wavelength',
                    'Amp_0', 'Amp_0_lowlim', 'Amp_0_highlim',
                    'Centroid_0', 'Centroid_0_lowlim', 'Centroid_0_highlim',
                    'Sigma_0', 'Sigma_0_lowlim', 'Sigma_0_highlim', 'region_ID'
                ]
                for col in float_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if col in ['Line_ID', 'region_ID']:
                            df[col] = np.int64(df[col])
    
                # Clear the existing layout
                for i in reversed(range(self.scroll_layout.count())):
                    widget = self.scroll_layout.itemAt(i).widget()
                    if isinstance(widget, QFrame):
                        self.scroll_layout.removeWidget(widget)
                        widget.deleteLater()
    
                # Populate the UI with the loaded data
                self.add_spaxel_info_frame('Observation Data', df_cont, df, df_obs)
                for ID in np.unique(df_cont['region_ID']):
                    ID = np.int64(ID)
                    self.add_spectral_frame(f'Spectral Region {ID+1}', df_cont, df, ID, addframe=False)
                    self.viewer_window.rebuild_plot(region_ID=ID, from_file=True, show_init=True, show_fit=False, x=0, y=0)
    
                print(f"File loaded: {file_path}")
    
            except Exception as e:
                print(f"Error opening file: {e}")


            
    def toggle_fullscreen(self):
        """Toggles full screen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_S and (event.modifiers() & Qt.ControlModifier):  # Check for Cmd/Ctrl + S
            self.save_file()  # Trigger save file dialog
        if event.key() == Qt.Key_O and (event.modifiers() & Qt.ControlModifier):  # Check for Cmd/Ctrl + S
            self.open_file()  # Trigger save file dialog
        # if event.key() == Qt.Key_D:  # If 'D' key is pressed
        #     self.spectral_count += 1
        #     ID = self.spectral_count - 1
        #     IDName = ID +1
        #     self.add_spectral_frame(f"Spectral Region {IDName}", df_cont, df, ID, addframe=True)
        #debugging
        if event.key() == Qt.Key_F:
            print(df)
            # print(self.buttons_dict)
            # print(df.iloc[0])
            # print(df_fit.iloc[0])
            
            # for col, val in df_cont.iloc[0].items():  # Assuming a single row
            #     print(f"{col}: {val}")

    def _build_obs_and_fit_toolbar(self, df_obs):
        """Build a compact single-row toolbar at the top of the scroll area.
        
        Left side: editable observation fields (Source, z, R).
        Right side: fit action buttons in a horizontal row.
        A '+ Spectral Region' button sits at the far right.
        Everything lives in a slim QFrame added to scroll_layout.
        """
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setObjectName('obs_toolbar_frame')
        frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        row = QHBoxLayout(frame)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(6)

        # ── Observation fields ──────────────────────────────────────────────
        _name = df_obs.loc[0, 'sourcename']    if len(df_obs) > 0 else ''
        _z    = df_obs.loc[0, 'redshift']      if len(df_obs) > 0 else ''
        _rp   = df_obs.loc[0, 'resolvingpower'] if len(df_obs) > 0 else ''

        self.source_name_button    = SpaxelButton(f'Source: {_name}',  'Source Name')
        self.source_redshift_button= SpaxelButton(f'z: {_z}',          'Source Redshift')
        self.resolving_power_button= SpaxelButton(f'R: {_rp}',         'Resolving Power')

        for btn in [self.source_name_button,
                    self.source_redshift_button,
                    self.resolving_power_button]:
            btn.setFixedHeight(28)
            row.addWidget(btn)

        self.source_name_button.clicked.connect(
            partial(self.on_button_click, data_frame=df_obs, frame_id=0, button_name='sourcename'))
        self.source_redshift_button.clicked.connect(
            partial(self.on_button_click, data_frame=df_obs, frame_id=0, button_name='redshift'))
        self.resolving_power_button.clicked.connect(
            partial(self.on_button_click, data_frame=df_obs, frame_id=0, button_name='resolvingpower'))

        # Separator
        sep = QFrame(); sep.setFrameShape(QFrame.VLine); sep.setFrameShadow(QFrame.Sunken)
        row.addWidget(sep)

        # ── Fit action buttons ──────────────────────────────────────────────
        self.fit_spaxel_button           = QPushButton('Fit Spaxel')
        self.fit_cube_button             = QPushButton('Fit Cube')
        self.fix_fit_button              = QPushButton('Rectify Bad Fits')
        self.save_cube_fit_button        = QPushButton('Save Fit (CSV)')
        self.save_cube_fit_fitsfile_button = QPushButton('Save Fit (FITS)')
        self.load_cube_fit_button        = QPushButton('Load Fit')

        self.fit_spaxel_button.clicked.connect(partial(self.fit_spaxel))
        self.fit_cube_button.clicked.connect(partial(self.fit_cube))
        self.fix_fit_button.clicked.connect(partial(self.fix_fits))
        self.save_cube_fit_button.clicked.connect(partial(self.save_cube_fit))
        self.save_cube_fit_fitsfile_button.clicked.connect(partial(self.save_fit_result_fitsfile))
        self.load_cube_fit_button.clicked.connect(partial(self.load_cube_fit))

        for btn in [self.fit_spaxel_button, self.fit_cube_button,
                    self.fix_fit_button, self.save_cube_fit_button,
                    self.save_cube_fit_fitsfile_button, self.load_cube_fit_button]:
            btn.setFixedHeight(28)
            row.addWidget(btn)

        # Separator
        sep2 = QFrame(); sep2.setFrameShape(QFrame.VLine); sep2.setFrameShadow(QFrame.Sunken)
        row.addWidget(sep2)

        # ── + Spectral Region ───────────────────────────────────────────────
        self.add_region_button = QPushButton('+ Region')
        self.add_region_button.setFixedHeight(28)
        self.add_region_button.setToolTip(
            'Add a new blank spectral region.\n'
            'You can also draw a continuum with the D key.'
        )
        self.add_region_button.clicked.connect(self._on_add_region_button_click)
        row.addWidget(self.add_region_button)

        row.addStretch()
        self.outer_scroll_layout.insertWidget(0, frame)  # toolbar always at top

    def add_spaxel_info_frame(self, title_text, df_cont, df, df_obs):
        """Legacy wrapper — calls the new compact toolbar builder."""
        self._build_obs_and_fit_toolbar(df_obs)

    def _on_add_region_button_click(self):
        """Manually add a new blank spectral region panel without drawing on the spectrum.
        
        Creates a placeholder continuum entry in df_cont with NaN geometry so the
        spectral region frame appears in the dock immediately.  The user can then
        populate x1/x2/slope/intercept by clicking the parameter buttons, or draw
        a real continuum with the 'D' key which will override these values.
        """
        global df_cont, df
        new_id = len(df_cont)  # Next region ID
        
        # Add a placeholder continuum row so add_spectral_frame has data to display
        placeholder_cont = pd.DataFrame({
            'Continuum Name': ['Continuum'],
            'x1': [np.nan],
            'x2': [np.nan],
            'Slope_0': [0.0],
            'Intercept_0': [0.0],
            'Slope_fit': [np.nan],
            'Intercept_fit': [np.nan],
            'region_ID': [new_id],
            'lineactor': [None],
        })
        
        if len(df_cont) == 0:
            df_cont = placeholder_cont
        else:
            df_cont = pd.concat([df_cont, placeholder_cont], ignore_index=True)
        
        # Add the spectral region frame (addframe=False because we already updated df_cont above)
        self.add_spectral_frame(
            f"Spectral Region {new_id + 1}", df_cont, df, ID=new_id, addframe=False
        )

    def fix_fits(self):
        global df, df_cont, df_fit
        
        # identify spaxels with bad fit
        if len(df_fit) > 0:
            rchisq_max = 0.01
            self.fit_cube(refit=True, rchisq_thresh=rchisq_max)
        
                


    def save_fit_result_fitsfile(self):
        """Save df_fit results as a multi-extension FITS file with WCS-based RA/Dec mapping."""
        wcs = self.viewer_window.wcs
        print('Saving cube fit to FITS file')
    
        # Prompt user for file save location
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "FITS Files (*.fits);;All Files (*)", options=options)
        
        if not file_path:
            print("File save canceled.")
            return
    
        # Replace "nan" and "NaN" strings with actual NaNs
        df_fit.replace(["nan", "NaN"], np.nan, inplace=True)
    
        # Convert coordinate columns to int
        df_fit[["spaxel_x", "spaxel_y"]] = df_fit[["spaxel_x", "spaxel_y"]].astype(int)
    
        #   Convert spaxel coordinates to RA, Dec using the 3D WCS
        spaxel_x = df_fit["spaxel_x"].values
        spaxel_y = df_fit["spaxel_y"].values
        spaxel_z = np.zeros_like(spaxel_x)  # Placeholder for spectral axis
        
        ra_vals, dec_vals, _ = wcs.all_pix2world(spaxel_x, spaxel_y, spaxel_z, 0)

        
        # Identify unique spectral lines
        unique_lines = np.unique(df_fit['LineName'])
        extensions = []
    
        # Create a Primary HDU
        primary_hdu = fits.PrimaryHDU()
        extensions.append(primary_hdu)
    
        # Generate image maps for each line and each fit parameter
        for line in unique_lines:
            line_df = df_fit[df_fit['LineName'] == line]
            for param in ["amp_fit", "cen_fit", "vel_fit", "sigma_fit"]:
                # Create an empty map with NaNs
                image_map = np.full((max(df_fit["spaxel_y"]) + 1, max(df_fit["spaxel_x"]) + 1), np.nan)
    
                # Populate map using RA/Dec indices
                for (_, row), ra, dec in zip(line_df.iterrows(), ra_vals, dec_vals):
                    x, y = int(row["spaxel_x"]), int(row["spaxel_y"])
                    image_map[y, x] = row[param]
    
                # Create FITS HDU with WCS header
                hdu = fits.ImageHDU(data=image_map, name=f"{param[:-4]}_{line}")
                hdu.header.update(wcs.to_header())  # Add WCS info
                extensions.append(hdu)
    
        # Write all extensions to FITS file
        hdulist = fits.HDUList(extensions)
        hdulist.writeto(file_path, overwrite=True)
    
        print(f"FITS file saved to {file_path}")
        

    def save_cube_fit(self):
        global df, df_cont, df_obs, df_fit
        """Save csv storing galaxy, observation, and cube fit data"""
        print('Saving cube fit to csv file')
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
    
                # Write df_obs (Observation information)
                writer.writerow(df_obs.columns)  # Header for df_obs
                writer.writerows(df_obs.values)  # Data for df_obs
    
                # Insert an empty row to separate the sections
                writer.writerow([])
                
                writer.writerow(['wavelength scale factor', 'flux scale factor'])
                
                writer.writerow([self.viewer_window.WLscalefactor,self.viewer_window.fluxscalefactor])
    
                # Insert an empty row to separate the sections
                writer.writerow([])
                
                # Write df_cont (continuum information)
                writer.writerow(df_cont.columns)  # Header for df_cont
                writer.writerows(df_cont.values)  # Data for df_cont
    
                # Insert an empty row for separation
                writer.writerow([])
    
                # Write df_cont (line information)
                writer.writerow(df.columns)  # Header for df
                writer.writerows(df.values)  # Data for df
    
                # Insert an empty row for separation
                writer.writerow([])
                
                # Write df_fit (Fitting results)
                writer.writerow(df_fit.columns)  # Header for df_fit
                writer.writerows(df_fit.values)  # Data for df_fit
    
            print(f"CSV File saved: {file_path}")
            
            
        
    def load_cube_fit(self):
        global df, df_cont, df_fit, df_obs, viewing_fit
        """Open a CSV file and load it into the dataframes"""
        print('opening file')
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    
        if file_path:
            with open(file_path, 'r', newline='') as file:
                reader = csv.reader(file)
                rows = [row for row in reader]  # Don't remove empty rows yet
    
            # Find all the empty rows that separate the dataframes
            split_indices = [i for i, row in enumerate(rows) if not any(row)]
            
            if len(split_indices) < 3:
                raise ValueError("CSV file doesn't have enough sections separated by blank rows")
            
            # Read df_obs (first section)
            df_obs = pd.DataFrame(rows[:split_indices[0]][1:], columns=rows[0])  # Exclude header row from data
            df_obs = df_obs.apply(pd.to_numeric, errors='ignore')  # Convert numeric columns if possible
    
            self.source_name_button.setText('Source: '+str(df_obs['sourcename'].item()))
            self.source_redshift_button.setText('z: '+str(df_obs['redshift'].item()))
            self.resolving_power_button.setText('R: '+str(df_obs['resolvingpower'].item()))
            
            df_scale_header_index = split_indices[0] + 1
            while df_scale_header_index < len(rows) and not any(rows[df_scale_header_index]):
                df_scale_header_index += 1
            
            df_scale_data_start = df_scale_header_index + 1
            df_scale_data_end = split_indices[1]
            
            df_scale = pd.DataFrame(rows[df_scale_data_start:df_scale_data_end], 
                                  columns=rows[df_scale_header_index])
            df_scale = df_scale.apply(pd.to_numeric, errors='ignore')
            
            
            
            # Read df_cont (second section, after first empty row)
            df_cont_header_index = split_indices[1] + 1
            while df_cont_header_index < len(rows) and not any(rows[df_cont_header_index]):
                df_cont_header_index += 1
                
            df_cont_data_start = df_cont_header_index + 1
            df_cont_data_end = split_indices[2]
            
            df_cont = pd.DataFrame(rows[df_cont_data_start:df_cont_data_end], 
                                  columns=rows[df_cont_header_index])
            df_cont = df_cont.apply(pd.to_numeric, errors='ignore')
            
            # Read df (third section, after second empty row)
            df_header_index = split_indices[2] + 1
            while df_header_index < len(rows) and not any(rows[df_header_index]):
                df_header_index += 1
                
            df_data_start = df_header_index + 1
            df_data_end = split_indices[3]
            
            df = pd.DataFrame(rows[df_data_start:df_data_end], 
                             columns=rows[df_header_index])
            df = df.apply(pd.to_numeric, errors='ignore')
            
            # Read df_fit (fourth section, after third empty row)
            df_fit_header_index = split_indices[3] + 1
            while df_fit_header_index < len(rows) and not any(rows[df_fit_header_index]):
                df_fit_header_index += 1
                
            df_fit_data_start = df_fit_header_index + 1
            
            df_fit = pd.DataFrame(rows[df_fit_data_start:], 
                                 columns=rows[df_fit_header_index])
            df_fit = df_fit.apply(pd.to_numeric, errors='ignore')
            
            if 'RA' not in df_fit.columns:
                df_fit['RA'] = self.viewer_window.pixel_to_ra_dec(df_fit['spaxel_x'],df_fit['spaxel_y'])[0]
                df_fit['Dec'] = self.viewer_window.pixel_to_ra_dec(df_fit['spaxel_x'],df_fit['spaxel_y'])[1]
            
            # Define column groups for conversion
            float_columns = [
                col for col in df_fit.columns 
                if col not in ["color", "success", "LineName", "LineID", "spaxel_x", "spaxel_y", "region_ID"]
            ]
            
            int_columns = ["LineID", "spaxel_x", "spaxel_y", "region_ID"]
            
            # Replace "nan" and "NaN" strings with actual NaNs
            df_fit.replace(["nan", "NaN"], np.nan, inplace=True)
            
            # Convert columns to float
            df_fit[float_columns] = df_fit[float_columns].astype(float)
            
            # Convert integer columns (use Int64 dtype to allow NaNs)
            df_fit[int_columns] = df_fit[int_columns].astype("Int64")
            
            # Remove any current actors from the spectrum plot
            # for frame_id in np.unique(df_cont['region_ID']):
            #     df_cont.loc[(df_cont["region_ID"] == frame_id)]['lineactor'].item().remove()
            # self.viewer_window.spectrum_canvas.draw()
            
            # Iterate over all items in the layout and remove them
            for i in reversed(range(self.scroll_layout.count())):  # Reverse to avoid index shifting
                widget = self.scroll_layout.itemAt(i).widget()
                if isinstance(widget, QFrame):
                    self.scroll_layout.removeWidget(widget)
                    widget.deleteLater()
                    
            # Regenerate the UI using updated df and df_cont
            self.add_spaxel_info_frame('Observation Data', df_cont, df, df_obs)
            for ID in np.unique(df_cont['region_ID']):
                ID = np.int64(ID)
                self.add_spectral_frame('Spectral Region '+str(ID+1), df_cont, df, ID, addframe=False)
                self.viewer_window.rebuild_plot(region_ID=ID,from_file=True,show_init=False,show_fit=True,x=df_fit.iloc[0]['spaxel_x'],y=df_fit.iloc[0]['spaxel_y'])
                    
            self.fitloaded = True
            viewing_fit = True
            print(f"File loaded: {file_path}")



    def line(self, x, m, b):
        return m*x+b
    
    # Define a Gaussian function
    def gaussian_function(self, x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def generalized_model(self, x, m, b, *gaussian_params):
        """
        Model combining a linear continuum and multiple Gaussian components.
    
        Parameters:
        - x: ndarray, input x values
        - m: float, slope of the linear continuum
        - b: float, intercept of the linear continuum
        - *gaussian_params: list, flattened parameters for each Gaussian in the form (A1, mu1, sigma1, A2, mu2, sigma2, ...)
    
        Returns:
        - y: ndarray, model values
        """
        # Linear component
        y = self.line(x, m, b)
        
        # Add each Gaussian
        num_gaussians = len(gaussian_params) // 3
        for i in range(num_gaussians):
            A = gaussian_params[3 * i]
            mu = gaussian_params[3 * i + 1]
            sigma = gaussian_params[3 * i + 2]
            y += self.gaussian_function(x, A, mu, sigma)
        
        return y
    


    def fit_spaxel(self,z,max_nfev=256,params_to_use=None):
        global df_fit, df, df_cont
        
        if self.viewer_window.is_1d_spectrum == False:
            spectrum = self.viewer_window.get_spectrum_at_spaxel(
                self.viewer_window.current_spaxel[0], self.viewer_window.current_spaxel[1]
            )
        else:
            spectrum = FITS_DATA
            
        spectrum = np.nan_to_num(spectrum)
        
        # ── Flux rescaling ────────────────────────────────────────────────
        # lmfit's Levenberg-Marquardt uses finite differences ~1e-8 * value
        # for the Jacobian.  When flux values are ~1e-18, that step underflows
        # to zero, making the Jacobian singular and all amplitude fits zero.
        # Rescaling to order-unity before the fit and back afterward avoids this.
        flux_scale = np.nanpercentile(np.abs(spectrum[spectrum != 0]), 95) if np.any(spectrum != 0) else 1.0
        if flux_scale == 0 or not np.isfinite(flux_scale):
            flux_scale = 1.0
        spectrum_scaled = spectrum / flux_scale

        # Scale amplitude params and their bounds
        params_scaled = params_to_use.copy()
        for pname, param in params_scaled.items():
            if pname.startswith('amp'):
                param.set(value=param.value / flux_scale)
                if param.min is not None:
                    param.min = param.min / flux_scale
                if param.max is not None:
                    param.max = param.max / flux_scale
            elif pname.startswith('intercept'):
                param.set(value=param.value / flux_scale)
        # ─────────────────────────────────────────────────────────────────

        # Constants for velocity calculation
        c = 299792.458  # km/s
        
        # Perform the fit using lmfit
        try:
            result = piecewise_model.fit(spectrum_scaled, params_scaled, x=wavelengths, max_nfev=max_nfev)
            
            # Map parameter prefixes to continuum regions
            cont_map = {
                f"x{i + 1}_start": i + 1 for i in range(len(df_cont))
            }
            
            # Loop through each emission line in df
            for line_idx, line in enumerate(df.itertuples(), start=1):
                line_id = line.Line_ID
                line_name = line.Line_Name
                rest_wavelength = np.float64(df.loc[df['Line_ID']==line_id]['Rest Wavelength'])
            
                # Identify corresponding region for this line
                region_id = line.region_ID
                region_index = df_cont[df_cont['region_ID'] == region_id].index[0] + 1
            
                # Collect continuum model parameters
                cont_params = {
                    f'cont_region{region_index}_x_start': params_to_use[f'x{region_index}_start'].value,
                    f'cont_region{region_index}_x_end': params_to_use[f'x{region_index}_end'].value,
                    f'cont_region{region_index}_slope_init': params_to_use[f'slope{region_index}'].init_value,
                    f'cont_region{region_index}_slope_fit': result.params[f'slope{region_index}'].value * flux_scale,
                    f'cont_region{region_index}_intercept_init': params_to_use[f'intercept{region_index}'].init_value,
                    f'cont_region{region_index}_intercept_fit': result.params[f'intercept{region_index}'].value * flux_scale
                }
            
                # If there is an intermediate region, capture its parameters
                if region_index < len(df_cont):
                    cont_params.update({
                        f'cont_region{region_index}_x_int_start': params_to_use[f'x_int_{region_index}_start'].value,
                        f'cont_region{region_index}_x_int_end': params_to_use[f'x_int_{region_index}_end'].value,
                        f'cont_region{region_index}_slope_int_init': params_to_use[f'slope_int_{region_index}'].init_value,
                        f'cont_region{region_index}_slope_int_fit': result.params[f'slope_int_{region_index}'].value,
                    })
            
                # Get parameter keys for this line
                amp_key = f'amp{line_idx}'
                cen_key = f'cen{line_idx}'
                sigma_key = f'sigma{line_idx}'
                
                # Calculate velocities
                if np.isfinite(rest_wavelength):
                    # Initial velocity from initial parameters
                    vel_init = c * ((params_to_use[cen_key].init_value / (rest_wavelength*(z+1))) - 1)
                    # Fitted velocity from fitted parameters
                    vel_fit = c * ((result.params[cen_key].value / (rest_wavelength*(z+1))) - 1)
                    # Velocity from parameter if it exists
                    # vel_param = result.params[vel_key].value if vel_key in result.params else vel_fit
                    # vel_std = result.params[vel_key].stderr if vel_key in result.params else np.nan
                else:
                    vel_init = vel_fit = vel_param = vel_std = np.nan
            
                fit_entry = {
                    'spaxel_x': self.viewer_window.current_spaxel[0],
                    'spaxel_y': self.viewer_window.current_spaxel[1],
                    'RA': self.viewer_window.pixel_to_ra_dec(self.viewer_window.current_spaxel[0],self.viewer_window.current_spaxel[1])[0],
                    'Dec': self.viewer_window.pixel_to_ra_dec(self.viewer_window.current_spaxel[0],self.viewer_window.current_spaxel[1])[1],
                    'region_ID': region_id,
                    'LineName': line_name,
                    'LineID': line_id,
            
                    # Amplitude information
                    'amp_init': params_to_use[amp_key].init_value,
                    'amp_fit': result.params[amp_key].value * flux_scale,
                    'amp_std': (result.params[amp_key].stderr or 0) * flux_scale,
            
                    # Centroid information
                    'cen_init': params_to_use[cen_key].init_value,
                    'cen_fit': result.params[cen_key].value,
                    'cen_std': result.params[cen_key].stderr,
                    
                    # Velocity information
                    'vel_init': vel_init,
                    'vel_fit': vel_fit,
                    # 'vel_param': vel_param,
                    # 'vel_std': vel_std,
                    'rest_wavelength': rest_wavelength,
                    
                    # Sigma (width) information
                    'sigma_init': params_to_use[sigma_key].init_value,
                    'sigma_fit': result.params[sigma_key].value,
                    'sigma_std': result.params[sigma_key].stderr,
            
                    # Fit statistics
                    'BIC': result.bic,
                    'rchisq': result.redchi,
                    'success': result.success
                }
            
                # Add continuum parameters
                fit_entry.update(cont_params)
            
                fit_results.append(fit_entry)
    
        except RuntimeError as e:
            # print(f"Fit failed for spaxel {self.viewer_window.current_spaxel}: {e}")
            # Add a failed entry with basic information
            fit_results.append({
                'spaxel_x': self.viewer_window.current_spaxel[0],
                'spaxel_y': self.viewer_window.current_spaxel[1],
                'fit_success': False,
                'error': str(e)
            })
                
            

    def calculate_snr_map(self, linewl, Nsigma, search_window_width=None, continuum_offset=None, continuum_width=None):
        """
        Calculate an SNR map using wavelength units (same as 'wavelengths' array).
        
        Parameters:
            linewl (float or array-like): Central wavelength(s) of the emission line(s) 
            Nsigma (float): SNR threshold for contour drawing
            search_window_width (float): Width of search window around line center (in wavelength units)
            continuum_offset (float): Offset from line center to start continuum regions (in wavelength units)
            continuum_width (float): Width of continuum regions (in wavelength units)
        
        Returns:
            snr_map (numpy.ndarray): 2D array of max SNR values across emission lines for each spaxel
        """
        global snr_map
        num_wavelengths, nx, ny = np.shape(FITS_DATA)
        snr_map = np.zeros((nx, ny))
        
        # Set default window sizes if not specified
        if search_window_width is None:
            search_window_width = 50 * np.median(np.diff(wavelengths))  # 10x wavelength step
        
        if continuum_offset is None:
            continuum_offset = 60 * np.median(np.diff(wavelengths))  # 20x wavelength step
        
        if continuum_width is None:
            continuum_width = 70 * np.median(np.diff(wavelengths))  # 30x wavelength step
    
        print(f'Calculating S/N map for line_wl = {linewl}, masking at {Nsigma}-sigma!')
    
        for i in range(nx):
            for j in range(ny):
                spectrum = FITS_DATA[:, i, j]
                snr_values = []
                
                for index, row in df.iterrows():
                    line_center = row['Centroid_0']  # Line center in wavelength units
                    
                    # Define line region
                    line_mask = (wavelengths >= line_center - search_window_width/2) & \
                               (wavelengths <= line_center + search_window_width/2)
                    line_region = spectrum[line_mask]
                    
                    # Use robust maximum for peak flux
                    peak_flux = np.percentile(line_region, 95) if len(line_region) > 0 else 0
                    
                    # Define continuum regions (both sides of line)
                    left_cont_mask = (wavelengths >= line_center - continuum_offset - continuum_width) & \
                                    (wavelengths <= line_center - continuum_offset)
                    right_cont_mask = (wavelengths <= line_center + continuum_offset + continuum_width) & \
                                     (wavelengths >= line_center + continuum_offset)
                    
                    # Combine continuum regions
                    continuum_flux = np.concatenate([
                        spectrum[left_cont_mask],
                        spectrum[right_cont_mask]
                    ])
                    
                    # Compute noise as MAD scaled to STD (robust against outliers)
                    if len(continuum_flux) > 1:
                        noise_level = 1.4826 * np.median(np.abs(continuum_flux - np.median(continuum_flux)))
                    else:
                        noise_level = np.nan
                    
                    # Compute SNR if valid
                    if noise_level > 0 and not np.isnan(noise_level):
                        snr = peak_flux / noise_level
                        snr_values.append(snr)
                    else:
                        snr_values.append(0)
                
                # Store maximum SNR for this spaxel
                snr_map[i, j] = np.nanmax(snr_values) if snr_values else 0
    
        # Draw the contour on the viewer window at the specified Nsigma level
        contour = self.viewer_window.ax.contour(snr_map, levels=[Nsigma], colors='red', linewidths=1.5)
    
        # Update the canvas
        self.viewer_window.canvas.draw()
        self.viewer_window.spectrum_canvas.draw_idle()
        
        return snr_map




    def fit_cube(self, refit=False, rchisq_thresh=None):
        global df, df_fit, fit_results, snr_mask, piecewise_model, line, new_results#,params
        
        if self.viewer_window.is_1d_spectrum == False:
            nx, ny = np.shape(FITS_DATA)[2], np.shape(FITS_DATA)[1]
        else:
            nx = 1
            ny = 1
            
        total_spaxels = nx * ny
    
        app = QApplication.instance() or QApplication([])
    
        # Progress UI
        progress_window = QWidget()
        progress_window.setWindowTitle("Fitting Progress")
    
        progress_bar = QProgressBar()
        progress_bar.setRange(0, total_spaxels)
    
        status_label = QLabel(f"Fitting spaxel 0 / {total_spaxels}")
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
    
        layout = QVBoxLayout()
        layout.addWidget(progress_bar)
        layout.addWidget(status_label)
        progress_window.setLayout(layout)
        progress_window.show()
    
        # Constants for velocity calculation
        c = 299792.458  # km/s
        
        # Get galaxy redshift
        if df_obs['redshift'].item() == '':
            z = 0
        else:
            z = df_obs['redshift'].item()
    
        # Model parameters
        params = Parameters()
        
        # Sort continuum regions by x1 (start wavelength)
        df_cont_sort = df_cont.sort_values(by='x1').reset_index(drop=True)
        
        # Ensure lines are sorted by Centroid_0 within each region
        df_sort = df.sort_values(by=['region_ID', 'Centroid_0']).reset_index(drop=True)
    
        # Loop through each continuum region and add parameters
        for i, row in df_cont.iterrows():
            region_id = row['region_ID']
    
            # Set x-limits for each region (fixed)
            params.add(f'x{i + 1}_start', value=row['x1'], vary=False)
            params.add(f'x{i + 1}_end', value=row['x2'], vary=False)
    
            # Linear continuum parameters for each region
            slope = row['Slope_0'] if np.isfinite(row['Slope_0']) else 0
            intercept = row['Intercept_0'] if np.isfinite(row['Intercept_0']) else 0
    
            params.add(f'slope{i + 1}', value=slope, vary=True)
            params.add(f'intercept{i + 1}', value=intercept, vary=True)
    
            # Add parameters for the linear region between this and the next region
            if i < len(df_cont) - 1:
                next_row = df_cont.iloc[i + 1]
                params.add(f'x_int_{i + 1}_start', value=row['x2'], vary=False)
                params.add(f'x_int_{i + 1}_end', value=next_row['x1'], vary=False)
    
                params.add(f'slope_int_{i + 1}', value=slope, vary=True)
                params.add(f'intercept_int_{i + 1}', value=intercept, vary=True)
    
            # Extract Gaussians associated with this region
            region_lines = df[df['region_ID'] == region_id]
    
            # Add Gaussian parameters for this region
            for j, line in enumerate(df.itertuples(), start=1):
                # Cast all parameter values to float64 explicitly.
                # After pd.concat operations, columns can be object dtype;
                # lmfit silently misbehaves when given non-float initial values.
                _amp   = np.float64(line.Amp_0)
                _cen   = np.float64(line.Centroid_0)
                _sigma = np.float64(line.Sigma_0)
                _amp_lo  = np.float64(line.Amp_0_lowlim)   if np.isfinite(np.float64(line.Amp_0_lowlim))  else None
                _amp_hi  = np.float64(line.Amp_0_highlim)  if np.isfinite(np.float64(line.Amp_0_highlim)) else None
                _cen_lo  = np.float64(line.Centroid_0_lowlim)  if np.isfinite(np.float64(line.Centroid_0_lowlim))  else None
                _cen_hi  = np.float64(line.Centroid_0_highlim) if np.isfinite(np.float64(line.Centroid_0_highlim)) else None
                _sig_lo  = np.float64(line.Sigma_0_lowlim)  if np.isfinite(np.float64(line.Sigma_0_lowlim))  else None
                _sig_hi  = np.float64(line.Sigma_0_highlim) if np.isfinite(np.float64(line.Sigma_0_highlim)) else None
                # Guard: if initial value is exactly 0, use a small non-zero starting point
                # so the optimizer has something to scale against
                if _amp == 0:
                    _amp = 1e-30
                params.add(f'amp{j}',   value=_amp,   vary=True, min=_amp_lo,  max=_amp_hi)
                params.add(f'cen{j}',   value=_cen,   vary=True, min=_cen_lo,  max=_cen_hi)
                params.add(f'sigma{j}', value=_sigma, vary=True, min=_sig_lo,  max=_sig_hi)
                
                
                # Add velocity parameter for this line
                if df.iloc[j-1]['Rest Wavelength']:
                    if type(df.iloc[j-1]['Rest Wavelength']) == str:
                        rest_wavelength = ast.literal_eval(df.iloc[j-1]['Rest Wavelength'])
                    else:
                        rest_wavelength = df.iloc[j-1]['Rest Wavelength']
                    
                else:
                    rest_wavelength = np.nan
                
                if (np.isfinite(rest_wavelength)) & ('constraints' in df.columns):
                    # Update constraints in df
                    df = update_constraints_with_velocity(df, z)
    
            # Add a "fake" parameter to store the number of Gaussians in this region
            params.add(f'NR{i + 1}', value=len(region_lines), vary=False)
    
        # choose the model based on parameters
        Nregions = len(df_cont)
        Nlines = len(np.unique(df['Line_ID']))
        
        # parameter constraints 
        add_dataframe_constraints_to_params(df, params)
    
        # print("\nParameters with Constraints:")
        # params.pretty_print()
        print(f'Nregions={Nregions}, Nlines={Nlines}')
        model_maker = HyperCube_ModelFunctions.PiecewiseModel(n_regions=Nregions, n_gaussians=Nlines)
        piecewise_model = Model(model_maker.model_function)
        # piecewise_model = Model(HyperCube_ModelFunctions.model_chooser(Nregions, Nlines))
        
        # Save the spaxel the user has locked before the loop mutates current_spaxel
        _locked_spaxel = self.viewer_window.current_spaxel

        if len(fit_results) > 0 and not refit:
            fit_results = []
            
        # Handle refit case with rchisq threshold
        if refit and rchisq_thresh is not None and 'df_fit' in globals() and 'rchisq' in df_fit.columns:
            fit_results = []
            
            # Get indices of spaxels that need refitting
            bad_fits = df_fit[df_fit['rchisq'] > rchisq_thresh]
            bad_spaxels = bad_fits[['spaxel_x', 'spaxel_y']].drop_duplicates()
            
            # Create a copy of original parameters to perturb
            refit_params = params.copy()
            
            substrings = ['amp', 'cen', 'sigma']
            
            # Perturb all varying parameters by ±10%
            for name, param in refit_params.items():
                if param.vary:
                    if any(sub in name for sub in substrings):
                        current_val = param.value
                        perturbation = np.random.uniform(0.9, 1.1)  # 10% random variation
                        refit_params[name].value = current_val * perturbation
                        
                    # print('original parameters:')
                    # params.pretty_print()
                    # print('refit parameters:')
                    # refit_params.pretty_print()
                        
            # Refit only the bad spaxels with perturbed parameters
            for i, j in bad_spaxels.itertuples(index=False):
                if snr_map[j, i] >= snr_value:
                    # Get original amplitudes for this spaxel - properly extract as dict
                    original_data = df_fit[
                        (df_fit['spaxel_x'] == i) & 
                        (df_fit['spaxel_y'] == j)
                    ]
                    original_amplitudes = dict(zip(original_data['LineID'], original_data['amp_fit']))
                    
                    # Perform the fit
                    self.viewer_window.current_spaxel = (i, j)
                    self.fit_spaxel(z, max_nfev=512, params_to_use=refit_params)
                    
                    # Get new results (last n entries where n=number of lines in spaxel)
                    new_results = [r for r in fit_results[-len(original_amplitudes):] 
                                  if r['spaxel_x'] == i and r['spaxel_y'] == j]
                    
                    # Print comparison
                    print(f"\nSpaxel ({i}, {j}) - Amplitude Comparison:")
                    print(f"{'LineID':<10} | {'Old Amp':<10} | {'New Amp':<10} | {'Change (%)':<10}")
                    print("-" * 45)
                    
                    for new_r in new_results:
                        line_id = new_r['LineID']
                        try:
                            old_amp = original_amplitudes[line_id]
                            new_amp = new_r['amp_fit']
                            change_pct = (new_amp - old_amp)/old_amp * 100
                            print(
                                f"{line_id:<10} | {old_amp:<10.3f} | {new_amp:<10.3f} | "
                                f"{change_pct:>+10.1f}%"
                            )
                        except KeyError:
                            print(f"{line_id:<10} | {'N/A':<10} | {new_amp:<10.3f} | {'New':>10}")

        
                    current_spaxel = i * ny + j + 1
                    progress_bar.setValue(current_spaxel)
                    status_label.setText(f"Refitting spaxel {current_spaxel} / {total_spaxels}")
                    QApplication.processEvents()
        
                    if current_spaxel % 500 == 0 and psutil.virtual_memory().percent > 80:
                        gc.collect()
            
            # Merge results (keeping LineID matching)
            new_results = pd.DataFrame(fit_results)
            
            # First, drop the rows from df_fit that match the rows in df_new (on the three columns)
            df_fit_filtered = df_fit.merge(
                new_results[['spaxel_x', 'spaxel_y', 'LineID']],
                on=['spaxel_x', 'spaxel_y', 'LineID'],
                how='left',
                indicator=True
            ).query('_merge == "left_only"').drop(columns='_merge')
            
            # Then, append the new rows
            df_fit = pd.concat([df_fit_filtered, new_results], ignore_index=True)
            # new_results = pd.DataFrame(fit_results[-len(bad_spaxels):])
            # composite_keys = (df_fit['spaxel_x'].astype(str) + '_' + 
            #                  df_fit['spaxel_y'].astype(str) + '_' + 
            #                  df_fit['LineID'].astype(str))
            # new_keys = (new_results['spaxel_x'].astype(str) + '_' + 
            #             new_results['spaxel_y'].astype(str) + '_' + 
            #             new_results['LineID'].astype(str))
            
            # df_fit = df_fit[~composite_keys.isin(new_keys)]
            # df_fit = pd.concat([df_fit, new_results], ignore_index=True)
            
        else:
            # Normal fitting procedure for all spaxels
            for i in range(nx):
                for j in range(ny):
                    if snr_map[j,i] >= snr_value:
                        if progress_window.isVisible() and psutil.virtual_memory().percent > 80:
                            QApplication.processEvents()
        
                        self.viewer_window.current_spaxel = (i, j)
                        self.fit_spaxel(z, max_nfev=512, params_to_use=params)
        
                        current_spaxel = i * ny + j + 1
                        progress_bar.setValue(current_spaxel)
                        status_label.setText(f"Fitting spaxel {current_spaxel} / {total_spaxels}")
                        QApplication.processEvents()
        
                        if current_spaxel % 500 == 0 and psutil.virtual_memory().percent > 80:
                            gc.collect()
            
            # After fitting, create the results dataframe
            df_fit = pd.DataFrame(fit_results)
        
        # Clean up the dataframe
        if 'fit_success' in df_fit.columns:
            # Handle failed fits
            df_fit['success'] = df_fit.get('success', True) & ~df_fit['fit_success'].isna()
            df_fit.drop(columns=['fit_success'], inplace=True, errors='ignore')
        
        # Assign colors to the lines for plotting:
        df_fit['LineID'] = df_fit['LineID'].astype(float)
        unique_lines = df_fit['LineID'].unique()
        
        svg_colors = [
            'dodgerblue', 'mediumseagreen', 'darkorange', 'mediumpurple',
            'deepskyblue', 'gold', 'steelblue', 'mediumaquamarine',
            'peru', 'cornflowerblue'
        ]
        
        lineid_to_color = {line: svg_colors[i % len(svg_colors)] for i, line in enumerate(unique_lines)}
        df_fit['color'] = df_fit['LineID'].map(lineid_to_color)
        
        progress_window.close()

        # Refresh the fitted model on the CURRENTLY DISPLAYED spaxel
        vw = self.viewer_window
        # Restore the spaxel the user had locked — fit_cube overwrites current_spaxel
        vw.current_spaxel = _locked_spaxel
        cx, cy = int(_locked_spaxel[0]), int(_locked_spaxel[1])

        # ── Wipe every curve from the spectrum (both dashed and solid init-guess) ──
        # Keep only the raw spectrum step-line (_child0 label)
        for _line in vw.spectrum_ax.lines[:]:
            if _line.get_label() != '_child0':
                try: _line.remove()
                except ValueError: pass
        vw.gaussian_component_lines.clear()
        # Also remove the lineactor refs so rebuild_plot plots fresh lines
        df_cont['lineactor'] = None

        # ── Guard: only rebuild if this spaxel was actually fitted ──────────────
        fitted_spaxels = set(
            zip(df_fit['spaxel_x'].astype(int), df_fit['spaxel_y'].astype(int))
        ) if len(df_fit) > 0 else set()

        if (cx, cy) in fitted_spaxels:
            for region_ID in np.unique(np.int64(df_cont['region_ID'])):
                vw.rebuild_plot(region_ID, from_file=True, show_init=False, show_fit=True, x=cx, y=cy)
        vw.spectrum_canvas.draw_idle()
        # Remove the blue init-guess spaxel marker now that the fit is done
        vw._init_guess_spaxel = None
        if getattr(vw, '_blue_rect', None) is not None:
            vw._blue_rect.set_visible(False)
            vw.canvas.draw_idle()

        print("Fitting complete for entire cube.")

        

    def add_spectral_frame(self, title_text, df_cont, df, ID, addframe):
        """Creates a spectral region frame and appends it vertically to the scroll area.

        Each frame has:
          - a compact title bar (region name + delete button)
          - a QGridLayout with continuum params in a header row, then one row per line
          - a + Line button at the bottom
        The frame expands horizontally to fill the dock width (scroll is vertical only).
        """
        # df_cont and df arrive as parameters; use local aliases so we can
        # assign back to the module-level globals when addframe is True.
        _df_cont = df_cont
        _df = df

        # ── Init placeholder data if addframe requested ─────────────────────
        data_cont_initreg = {
            'Continuum Name': ['Continuum'],
            'x1': [5000], 'x2': [5010],
            'Slope_0': [0], 'Intercept_0': [0.002],
            'Slope_fit': [np.nan], 'Intercept_fit': [np.nan],
            'region_ID': [ID], 'lineactor': [self.current_line]
        }
        data_lines_initreg = {
            'Line_ID': [0, 1], 'Line_Name': ['line 1', 'line 2'],
            'SNR': [np.nan, np.nan],
            'Amp_0': [0.2348, 0.343], 'Centroid_0': [5504, 6533], 'Sigma_0': [0.345, 0.45],
            'Amp_fit': [np.nan, np.nan], 'Centroid_fit': [np.nan, np.nan],
            'Sigma_fit': [np.nan, np.nan], 'region_ID': [ID, ID]
        }
        if addframe:
            # Write back to module-level globals (can't use 'global' because
            # df_cont and df are also parameter names in this function).
            import sys as _sys
            _g = vars(_sys.modules[__name__])
            _g['df_cont'] = pd.concat([_df_cont, pd.DataFrame(data_cont_initreg)], ignore_index=True)
            _g['df']      = pd.concat([_df,      pd.DataFrame(data_lines_initreg)], ignore_index=True)
            _df_cont = _g['df_cont']
            _df      = _g['df']

        # ── Outer frame ─────────────────────────────────────────────────────
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setObjectName(f"frame_{ID}")
        frame.setMinimumWidth(0)
        frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        frame_layout = QVBoxLayout(frame)
        frame_layout.setSpacing(4)
        frame_layout.setContentsMargins(6, 6, 6, 6)

        # ── Title bar ───────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        title_lbl = QLabel(title_text)
        title_lbl.setStyleSheet("font-size: 12pt; font-weight: bold;")
        title_lbl.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        title_row.addWidget(title_lbl)
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(22, 22)
        del_btn.setToolTip("Delete this spectral region")
        del_btn.setStyleSheet("background-color: lightcoral; color: black;")
        del_btn.clicked.connect(partial(self.on_deleteregion_button_click, frame_id=ID))
        title_row.addWidget(del_btn)
        frame_layout.addLayout(title_row)

        # ── Single grid: continuum header + continuum row + line header + line rows
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(6)
        grid_layout.setVerticalSpacing(3)
        grid_layout.setContentsMargins(4, 4, 4, 4)
        frame_layout.addLayout(grid_layout)

        self.add_continuum_buttons(ID, grid_layout)
        self.add_spectral_lines_button_header(grid_layout)
        self.add_spectral_lines(ID, grid_layout)

        # Equal stretch on every column so content fills the frame width
        n_cols = 17
        for col in range(n_cols):
            grid_layout.setColumnStretch(col, 1)
            grid_layout.setColumnMinimumWidth(col, 0)

        # ── + Line button ───────────────────────────────────────────────────
        add_line_btn = QPushButton('+ Line')
        add_line_btn.setFixedHeight(26)
        add_line_btn.clicked.connect(partial(self.on_addline_widget, data_frame=df, frame_id=ID))
        # Row = header(1) + continuum_header(1) + continuum_row(1) + line_header(1) + line_rows
        n_lines = len(df.loc[df['region_ID'] == ID]) if 'region_ID' in df.columns else 0
        grid_layout.addWidget(add_line_btn, 3 + n_lines * 2, 0)
        self.buttons_dict[(ID, '__add_line__')] = add_line_btn

        # ── Append to vertical scroll layout ────────────────────────────────
        # Insert before the trailing stretch if one exists, otherwise just append
        count = self.scroll_layout.count()
        last = self.scroll_layout.itemAt(count - 1) if count > 0 else None
        if last and last.spacerItem():
            self.scroll_layout.insertWidget(count - 1, frame)
        else:
            self.scroll_layout.addWidget(frame)

    def update_button_value(self, frame_id, button_name, new_value):
        """Update the button text based on row and column"""
        if any(x in button_name for x in ['sourcename','redshift','resolvingpower']):
            self.source_name_button.setText(f"Source: {df_obs.loc[0, 'sourcename']}")
            self.source_redshift_button.setText(f"z: {df_obs.loc[0, 'redshift']}")
            self.resolving_power_button.setText(f"R: {df_obs.loc[0, 'resolvingpower']}")
        else:
        
            if (frame_id, button_name) in self.buttons_dict:
                button = self.buttons_dict[(frame_id, button_name)]
                button.setText(str(new_value))
    
    def on_deleteregion_button_click(self, frame_id):
        global df, df_cont
        """Deletes an entire spectral region: removes all its curves from the plot,
        drops it from df and df_cont, and regenerates the UI."""
        print(f"Deleting region {frame_id}")

        vw = self.viewer_window
        ax = vw.spectrum_ax

        # Remove all dashed component curves belonging to this region
        for line in ax.lines[:]:
            if line.get_linestyle() in ('--', 'dashed'):
                try:
                    line.remove()
                except ValueError:
                    pass
        vw.gaussian_component_lines.clear()

        # Remove the solid total-model lineactor for this region
        line_actor = df_cont.loc[np.int64(df_cont["region_ID"]) == frame_id, 'lineactor'].item()
        if line_actor is not None and line_actor in ax.lines:
            try:
                line_actor.remove()
            except ValueError:
                pass

        # Drop region from dataframes
        df = df.reset_index(drop=True)
        df = df.drop(df.index[np.int64(df["region_ID"]) == frame_id]).reset_index(drop=True)
        df_cont = df_cont.drop(df_cont.index[np.int64(df_cont["region_ID"]) == frame_id]).reset_index(drop=True)

        # Rebuild UI frames
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, QFrame):
                self.scroll_layout.removeWidget(widget)
                widget.deleteLater()
        self.add_spaxel_info_frame('Observation Data', df_cont, df, df_obs)
        for ID in np.unique(df_cont['region_ID']):
            ID = np.int64(ID)
            self.add_spectral_frame('Spectral Region ' + str(ID + 1), df_cont, df, ID, addframe=False)

        vw.spectrum_canvas.draw_idle()
        # Remove the blue init-guess marker — region is gone
        if len(df_cont) == 0:
            vw._init_guess_spaxel = None
            if getattr(vw, '_blue_rect', None) is not None:
                vw._blue_rect.set_visible(False)
                vw.canvas.draw_idle()

    def on_deleteline_button_click(self, frame_id, line_id):
        global df, df_cont
        """Deletes a spectral line from df and redraws surviving components."""
        print(f"Deleting line {line_id} from frame {frame_id}")

        vw = self.viewer_window
        ax = vw.spectrum_ax

        # Step 1: wipe every dashed line from the spectrum axes.
        # curveactor refs fall out of sync after reindexing so we nuke all
        # dashed lines and let rebuild_plot redraw survivors cleanly.
        for line in ax.lines[:]:
            if line.get_linestyle() in ('--', 'dashed'):
                try:
                    line.remove()
                except ValueError:
                    pass
        vw.gaussian_component_lines.clear()

        # Step 2: remove the solid total-model lineactor for this region
        for actor in df_cont.loc[df_cont["region_ID"] == frame_id, 'lineactor']:
            if actor and actor in ax.lines:
                try:
                    actor.remove()
                except ValueError:
                    pass

        # Step 3: drop the line from df
        df = df.reset_index(drop=True)
        df = df.drop(df[(df["region_ID"].astype(int) == frame_id) &
                        (df["Line_ID"].astype(int) == line_id)].index)
        df = df.reset_index(drop=True)

        # Step 4: rebuild UI frames
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, QFrame):
                self.scroll_layout.removeWidget(widget)
                widget.deleteLater()
        self.add_spaxel_info_frame('Observation Data', df_cont, df, df_obs)
        for ID in np.unique(df_cont['region_ID']):
            ID = np.int64(ID)
            self.add_spectral_frame(f'Spectral Region {ID + 1}', df_cont, df, ID, addframe=False)

        # Step 5: redraw survivors
        vw.spectrum_canvas.draw_idle()
        vw.rebuild_plot(region_ID=frame_id, from_file=False, show_init=True, show_fit=False, x=0, y=0)

    def on_addline_widget(self, data_frame, frame_id):
        global df, df_cont
        # Retrieve the line associated with frame_id
        line_obj = df_cont.loc[df_cont["region_ID"] == frame_id, 'lineactor'].item()
    
        if not isinstance(line_obj, plt.Line2D):
            print(f"Error: No valid line object found.")
            return
    
        # Store the line and initialize Gaussian parameters
        self.active_line = line_obj
        region = df_cont.loc[df_cont["region_ID"] == frame_id]
        self.x1, self.x2 = region["x1"].item(), region["x2"].item()
        self.m, self.b = region["Slope_0"].item(), region["Intercept_0"].item()
        # Generate updated Gaussian + Line function
        x_vals = np.linspace(self.x1, self.x2, 1000)
        y_line = self.m * x_vals + self.b  # The baseline (without Gaussians)
        
        
        # Recompute all Gaussians from scratch (prevents runaway summing)
        y_gaussian_total = np.zeros_like(x_vals)
    
        # Sum over all existing Gaussians in df
        for _, row in df.iterrows():
            y_gaussian_total += row["Amp_0"] * np.exp(-((x_vals - row["Centroid_0"]) ** 2) / (2 * row["Sigma_0"] ** 2))
    
        self.gaussian_amplitude = 0.1
        self.gaussian_x0 = np.mean(x_vals)
        self.gaussian_sigma = 1
        # If we're actively adjusting a new Gaussian, include it
        y_gaussian_new = self.gaussian_amplitude * np.exp(-((x_vals - self.gaussian_x0) ** 2) / (2 * self.gaussian_sigma ** 2))
        y_gaussian_total += y_gaussian_new
    
        # Final y-values
        y_new = y_line + y_gaussian_total
    
        # Update the existing line object
        self.active_line.set_data(x_vals, y_new)
    
        # Redraw dynamically
        self.viewer_window.spectrum_ax.figure.canvas.draw_idle()
        
        if len(df) == 0: 
            line_id = 0
        else:
            line_id = np.max(np.int64(df['Line_ID']))+1
        
        df_new = pd.DataFrame({'Line_ID': [line_id],
                'Line_Name': [f'Line '+str(len(df.loc[df["region_ID"]==frame_id]))],
                'SNR':[np.nan],
                'Rest Wavelength': [np.nan],
                'Amp_0': [self.gaussian_amplitude],
                'Amp_0_lowlim': [0],
                'Amp_0_highlim': [np.inf],
                'Centroid_0': [self.gaussian_x0],
                'Centroid_0_lowlim': [-np.inf],
                'Centroid_0_highlim': [np.inf],
                'Sigma_0': [self.gaussian_sigma],
                'Sigma_0_lowlim': [0],
                'Sigma_0_highlim': [np.inf],
                'Amp_fit': [np.nan],
                'Centroid_fit': [np.nan],
                'Sigma_fit': [np.nan],
                'region_ID': [frame_id],
                'curveactor': [self.active_line]})
        df = pd.concat([df,df_new])
            
        self.on_addline_button_click(data_frame=df, frame_id=frame_id)
        
        
        
    def print_active_frames(self):
        print("Active frames in the scroll window:")
        for i in range(self.scroll_layout.count()):
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, QFrame):
                print(f"Frame ID: {widget.objectName()}")
            else:
                print("Non-frame widget found in scroll layout.")
        
    
    def on_addline_button_click(self, data_frame, frame_id):
        global df

        # Find the corresponding frame in the UI
        frame = None
        for i in range(self.scroll_layout.count()):
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, QFrame) and str(frame_id) in widget.objectName():
                frame = widget
                break
    
        # self.print_active_frames()
    
        if not frame:
            print("Frame not found!")
            return
    
    
        # Get the grid layout inside the frame (second item in the card VBoxLayout)
        grid_layout = frame.layout().itemAt(1).layout()

        # Retrieve the stored + Line button and remove it from its current position
        add_line_button = self.buttons_dict.get((frame_id, '__add_line__'))
        if add_line_button:
            grid_layout.removeWidget(add_line_button)

        # Only add the NEW (last) line row — re-drawing all rows creates duplicates
        df_region = df.loc[np.int64(df['region_ID']) == np.int64(frame_id)]
        n_lines = len(df_region)
        last_row_idx = n_lines - 1  # 0-based index of the new line in df_region

        button_columns = ['Line_Name', 'SNR', 'Rest Wavelength',
                          'Amp_0', 'Amp_0_lowlim', 'Amp_0_highlim',
                          'Centroid_0', 'Centroid_0_lowlim', 'Centroid_0_highlim',
                          'Sigma_0', 'Sigma_0_lowlim', 'Sigma_0_highlim',
                          'Amp_fit', 'Centroid_fit', 'v_fit', 'Sigma_fit']

        # grid row for the new line:
        # row 0 = cont header, 1 = cont row, 2 = line header, 3..3+n-1 = line rows
        grid_row = 3 + last_row_idx * 2

        for col, col_name in enumerate(button_columns):
            button_name = str(np.int64(df_region.iloc[last_row_idx]['Line_ID'])) + '~' + col_name
            if col_name in ['Rest Wavelength', 'Amp_0', 'Centroid_0', 'Sigma_0']:
                button_text = _fmt(df_region.iloc[last_row_idx][col_name])
            elif 'fit' in col_name.lower() or col_name in ('SNR', 'Rest Wavelength'):
                button_text = ''
            else:
                button_text = str(df_region.iloc[last_row_idx][col_name])
            btn = FrameButton(button_text, last_row_idx, col, frame_id, button_name)
            btn.setFixedHeight(24)
            if 'fit' in col_name:
                btn.clicked.connect(partial(self.on_fit_button_click, frame_id=frame_id, button_name=button_name))
            else:
                btn.clicked.connect(partial(self.on_button_click, data_frame=df, frame_id=frame_id, button_name=button_name))
            grid_layout.addWidget(btn, grid_row, col)
            self.buttons_dict[(frame_id, button_name)] = btn

        # Delete button at rightmost position of the new line row
        new_line_id = np.int64(df_region.iloc[last_row_idx]['Line_ID'])
        del_btn = FrameButton('x', last_row_idx, len(button_columns), frame_id, f'del_line_{new_line_id}')
        del_btn.setFixedHeight(24)
        del_btn.setStyleSheet("background-color: lightcoral; color: black;")
        del_btn.clicked.connect(partial(self.on_deleteline_button_click, frame_id=frame_id, line_id=new_line_id))
        grid_layout.addWidget(del_btn, grid_row, len(button_columns))
        self.buttons_dict[(frame_id, f'del_line_{new_line_id}')] = del_btn

        # Re-place the + Line button immediately below the new last row
        if add_line_button:
            grid_layout.addWidget(add_line_button, grid_row + 1, 0)

        # """
    
    def clear_snr_visualizer(self, frame_id, button_name):
        global df
        self.viewer_window.draw_image(FITS_DATA,cmap='gray',scale='linear',from_fits=True)
        self.update_button_value(frame_id, button_name, np.nan)
        df.loc[(np.int64(df['region_ID'])==frame_id) & 
                        (np.float64(df['Line_ID'])==np.float64(button_name.split('~')[0])),'SNR'] = np.nan


    def _show_source_dialog(self):
        """Source name dialog with NED name-resolve and coordinate-resolve options."""
        global df_obs, FITS_HEADER

        current_name = str(df_obs.loc[0, 'sourcename']) if len(df_obs) > 0 else ''
        current_z    = str(df_obs.loc[0, 'redshift'])   if len(df_obs) > 0 else ''

        dialog = QDialog(self)
        dialog.setWindowTitle('Source Information')
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)

        # ── Name row ─────────────────────────────────────────────────────────
        layout.addWidget(QLabel('Source name:'))
        name_edit = QLineEdit(current_name)
        name_edit.setPlaceholderText('e.g. NGC 1068')
        layout.addWidget(name_edit)

        # ── Resolve buttons ───────────────────────────────────────────────────
        resolve_name_btn  = QPushButton('Resolve name  (NED)')
        resolve_coord_btn = QPushButton('Resolve coordinates  (NED)')
        layout.addWidget(resolve_name_btn)
        layout.addWidget(resolve_coord_btn)

        # ── Status label ──────────────────────────────────────────────────────
        status = QLabel('')
        status.setWordWrap(True)
        layout.addWidget(status)

        # ── OK / Cancel ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        ok_btn     = QPushButton('OK');     ok_btn.setDefault(True)
        cancel_btn = QPushButton('Cancel')
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        # ── Helpers ───────────────────────────────────────────────────────────
        def _apply_result(ned_name, ned_z):
            """Write NED name and redshift into HyperCube state."""
            name_edit.setText(ned_name)
            df_obs.loc[0, 'sourcename'] = ned_name
            df_obs.loc[0, 'redshift']   = ned_z
            self.source_name_button.setText(f'Source: {ned_name}')
            self.source_redshift_button.setText(f'z: {ned_z}')

        def _query_ned_by_name():
            name = name_edit.text().strip()
            if not name:
                status.setText('Please enter a source name first.')
                return
            status.setText(f'Querying NED for "{name}"…')
            dialog.repaint()
            try:
                # Prefer astroquery if available (handles NED API correctly)
                try:
                    from astroquery.ipac.ned import Ned
                    result = Ned.query_object(name)
                    ned_name = str(result['Object Name'][0])
                    ned_z = float(result['Redshift'][0])
                except ImportError:
                    # Fallback: classic NED CGI with VOTable-like text output
                    import urllib.request, urllib.parse
                    enc = urllib.parse.quote(name)
                    url = (f'https://ned.ipac.caltech.edu/cgi-bin/nph-objsearch'
                           f'?objname={enc}&extend=no&of=ascii_tab&list_limit=1'
                           f'&img_stamp=false&zv_breaker=30000'
                           f'&out_csys=Equatorial&out_equinox=J2000.0')
                    req = urllib.request.Request(url, headers={'User-Agent': 'HyperCube/1.0'})
                    with urllib.request.urlopen(req, timeout=10) as r:
                        text = r.read().decode('utf-8', errors='replace')
                    # Parse tab-separated NED output: columns include Object Name, Redshift
                    lines = [l for l in text.splitlines() if l and not l.startswith('#')]
                    if not lines:
                        raise ValueError('No results returned by NED')
                    header = lines[0].split('\t')
                    row    = lines[1].split('\t')
                    d = dict(zip(header, row))
                    ned_name = d.get('Object Name', name).strip()
                    z_str    = d.get('Redshift', '').strip()
                    ned_z    = float(z_str) if z_str else None

                if ned_z is None:
                    status.setText(f'Found: {ned_name} — no redshift in NED.')
                    name_edit.setText(ned_name)
                    df_obs.loc[0, 'sourcename'] = ned_name
                    self.source_name_button.setText(f'Source: {ned_name}')
                else:
                    _apply_result(ned_name, ned_z)
                    status.setText(f'✓  {ned_name}   z = {ned_z:.6f}')
            except Exception as e:
                status.setText(f'NED query failed: {e}')

        def _query_ned_by_coords():
            try:
                ra  = float(FITS_HEADER.get('CRVAL1', FITS_HEADER.get('RA', '')))
                dec = float(FITS_HEADER.get('CRVAL2', FITS_HEADER.get('DEC', '')))
            except (TypeError, ValueError):
                status.setText('Could not read RA/Dec from FITS header.')
                return
            status.setText(f'Querying NED at RA={ra:.5f}, Dec={dec:.5f}...')
            dialog.repaint()
            try:
                try:
                    from astroquery.ipac.ned import Ned
                    import astropy.units as u
                    from astropy.coordinates import SkyCoord
                    coord = SkyCoord(ra=ra, dec=dec, unit='deg')
                    result = Ned.query_region(coord, radius=0.5 * u.arcmin)
                    if len(result) == 0:
                        raise ValueError('No objects found')
                    ned_name = str(result['Object Name'][0])
                    ned_z    = float(result['Redshift'][0])
                except ImportError:
                    import urllib.request
                    url = (f'https://ned.ipac.caltech.edu/cgi-bin/nph-objsearch'
                           f'?search_type=Near+Position+Search&ra={ra}&dec={dec}'
                           f'&radius=0.5&of=ascii_tab&list_limit=1&img_stamp=false'
                           f'&zv_breaker=30000&out_csys=Equatorial&out_equinox=J2000.0')
                    req = urllib.request.Request(url, headers={'User-Agent': 'HyperCube/1.0'})
                    with urllib.request.urlopen(req, timeout=10) as r:
                        text = r.read().decode('utf-8', errors='replace')
                    lines_t = [l for l in text.splitlines() if l and not l.startswith('#')]
                    if len(lines_t) < 2:
                        raise ValueError('No NED object within 0.5 arcmin of coordinates.')
                    header = lines_t[0].split('\t')
                    row    = lines_t[1].split('\t')
                    d = dict(zip(header, row))
                    ned_name = d.get('Object Name', '').strip()
                    z_str    = d.get('Redshift', '').strip()
                    ned_z    = float(z_str) if z_str else None

                if ned_z is None:
                    status.setText(f'Found: {ned_name} — no redshift in NED.')
                    name_edit.setText(ned_name)
                    df_obs.loc[0, 'sourcename'] = ned_name
                    self.source_name_button.setText(f'Source: {ned_name}')
                else:
                    _apply_result(ned_name, ned_z)
                    status.setText(f'Found: {ned_name}   z = {ned_z:.6f}')
            except Exception as e:
                status.setText(f'NED query failed: {e}')

        def _commit():
            # Save whatever name is in the box (user may have typed without resolving)
            df_obs.loc[0, 'sourcename'] = name_edit.text().strip()
            self.source_name_button.setText(f'Source: {name_edit.text().strip()}')
            dialog.accept()

        resolve_name_btn.clicked.connect(_query_ned_by_name)
        resolve_coord_btn.clicked.connect(_query_ned_by_coords)
        ok_btn.clicked.connect(_commit)
        name_edit.returnPressed.connect(_commit)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec_()

    def on_button_click(self, data_frame, frame_id, button_name):
        """Handle button click, show a QLineEdit for editing the value"""
        global df_cont, df
        print(str(frame_id)+', '+button_name)
        if 'SNR' in button_name:
            # Create a dialog window with two input fields
            snr_window = QDialog(self)
            snr_window.setWindowTitle(f'Line Name and Parameter Constraints')
            
            # Create layout
            layout = QVBoxLayout()
            
            # Input box with placeholder text
            text_box = QLineEdit()
            # print(df)
            # print(f'frame_id = {frame_id}')
            # print(f'button_name = {button_name}')
            snr_value = df.loc[
                (df['region_ID'].astype(np.int64) == frame_id) & 
                (df['Line_ID'].astype(np.float64) == np.float64(button_name.split('~')[0])),
                'SNR'].item()
            # Convert to float safely
            try:
                snr_value = np.float64(snr_value)  # Converts valid numbers and "nan" strings to float
            except ValueError:
                snr_value = np.nan  # If conversion fails, assume NaN
                
            # Check for NaN properly
            if not np.isnan(snr_value):
                text_box.setText(str(snr_value))
            else:
                text_box.setPlaceholderText("Enter SNR threshold")
            
            layout.addWidget(text_box)    
            
            # Button layout (side-by-side buttons)
            button_layout = QHBoxLayout()
            
            # "Visualize" button
            visualize_button = QPushButton("Visualize")
            visualize_button.clicked.connect(lambda: self.on_submit(text_box, data_frame, frame_id, button_name, button_type='SNR'))
            button_layout.addWidget(visualize_button)
            
            # "Clear" button to close the window
            clear_button = QPushButton("Clear")
            # Reset to the white light image
            clear_button.clicked.connect(lambda: self.clear_snr_visualizer(frame_id, button_name))
            clear_button.clicked.connect(snr_window.close)
            button_layout.addWidget(clear_button)
            
            # Add button layout to main layout
            layout.addLayout(button_layout)
            
            # Set layout and show the window
            snr_window.setLayout(layout)
            snr_window.show()
            
        if ('_highlim' in button_name) or ('_lowlim' in button_name):
            string = str(df.loc[(np.int64(df['region_ID'])==frame_id) & 
                            (np.float64(df['Line_ID'])==np.int64(button_name.split('~')[0]))][button_name.split('~')[1]].item())
            text_box = QLineEdit()
            text_box.setText(string)  # Set the initial value from the dataframe
            text_box.setGeometry(200, 200, 100, 30)
            text_box.show()
            # # # # Connect the returnPressed signal to handle text submission
            text_box.returnPressed.connect(lambda: self.on_submit(text_box, data_frame, frame_id, button_name, button_type='param_limit'))
        
        else:
        
            if any(x in button_name for x in ['sourcename','redshift','resolvingpower']):
                if 'sourcename' in button_name:
                    self._show_source_dialog()
                else:
                    string = df_obs[button_name].item()
                    text_box = QLineEdit()
                    text_box.setText(string)
                    text_box.setGeometry(200, 200, 100, 30)
                    text_box.show()
                    text_box.returnPressed.connect(lambda: self.on_submit(text_box, data_frame, frame_id, button_name, button_type='obs_button'))
            else:
            
            
                if button_name.split('~')[0] == 'continuum':
                    string = str(data_frame.loc[np.int64(data_frame['region_ID']) == frame_id][button_name.split('~')[1]].item())
                    button_type = 'continuum'
                # elif button_name.split('~')[1] == 'Line_Name':
                #     string = str(data_frame.loc[(data_frame['region_ID'] == np.float64(frame_id)) & (np.float64(data_frame['Line_ID'])==np.float64(button_name.split(',')[0]))][button_name.split('~')[1]].item())
                else:
                    string = str(data_frame.loc[(data_frame['region_ID'] == np.float64(frame_id)) & (np.float64(data_frame['Line_ID'])==np.float64(button_name.split('~')[0]))][button_name.split('~')[1]].item())
                    button_type = 'line'
                
            if any(p in button_name for p in ['Centroid_0', 'Amp_0', 'Sigma_0']):
                param = button_name.split('~')[1]
                line_id = np.int64(button_name.split('~')[0])
                current = df.loc[(np.int64(df['region_ID']) == frame_id) &
                                 (np.int64(df['Line_ID']) == line_id), param]
                current_val = _fmt(current.item()) if len(current) else ''
                dialog = QDialog(self)
                dialog.setWindowTitle(f'Edit {param}')
                layout = QVBoxLayout(dialog)
                layout.addWidget(QLabel(f'{param}  (current: {current_val})'))
                text_box_p = QLineEdit()
                text_box_p.setText(current_val)
                text_box_p.selectAll()
                layout.addWidget(text_box_p)
                btn_row = QHBoxLayout()
                ok_btn = QPushButton('OK')
                ok_btn.setDefault(True)
                ok_btn.clicked.connect(lambda: [self.on_submit(text_box_p, data_frame, frame_id, button_name, button_type='line_param'), dialog.accept()])
                text_box_p.returnPressed.connect(lambda: [self.on_submit(text_box_p, data_frame, frame_id, button_name, button_type='line_param'), dialog.accept()])
                cancel_btn = QPushButton('Cancel')
                cancel_btn.clicked.connect(dialog.reject)
                btn_row.addWidget(ok_btn)
                btn_row.addWidget(cancel_btn)
                layout.addLayout(btn_row)
                dialog.exec_()

            if 'Rest Wavelength' in button_name:
                text_box = QLineEdit()
                # text_box.setText('')  # Set the initial value from the dataframe
                text_box.show()
                text_box.returnPressed.connect(lambda: self.on_submit(text_box, data_frame, frame_id, button_name, button_type='RestWavelength'))
    
            if 'Line_Name' in button_name:
                
                lineid = np.int64(button_name.split('~')[0])
                linename = df.loc[(df['region_ID'] == frame_id) & (np.float64(df['Line_ID'])==np.int64(button_name.split('~')[0]))]['Line_Name'].item()
                parname = button_name.split("~")[1].split("_")[0]            
    
                # Create a dialog window with two input fields
                dialog = QDialog(self)
                dialog.setWindowTitle(f'Line Name and Parameter Constraints')
    
                # Layout for the dialog
                layout = QVBoxLayout(dialog)
    
                # Create a horizontal layout for the label and checkmark
                label_layout = QHBoxLayout()
    
                # Label for Initial Guess
                label_line_name = QLabel('Line name:', dialog)
                label_layout.addWidget(label_line_name)
    
                # Create the confirmation label (green checkmark) and hide it initially
                confirm_label = QLabel('✔️', dialog)
                confirm_label.setStyleSheet("color: limegreen; font-size: 16px;")
                confirm_label.hide()
    
                # Add the checkmark and ensure it's right next to the label
                label_layout.addWidget(confirm_label)
    
                # Ensure no additional stretching pushes them apart - add a stretch to push to the right
                label_layout.addStretch(1)
    
                # Add the horizontal layout to the main vertical layout
                layout.addLayout(label_layout)
    
                # Function to show and hide the checkmark after 2 seconds
                def on_input_confirmed():
                    confirm_label.show()
                    QTimer.singleShot(2000, confirm_label.hide)
    
                # Text box for line name
                text_box = QLineEdit(dialog)
                text_box.setText(linename)  # Set the initial value from the dataframe
                text_box.setPlaceholderText('Enter new line name')
                text_box.returnPressed.connect(lambda: [self.on_submit(text_box, data_frame, frame_id, button_name, button_type='line_name'), on_input_confirmed()]) # Connect return pressed signal
    
                # Add the text box to the main vertical layout
                layout.addWidget(text_box)
    
                # Label for constraints box
                label_constraints = QLabel('Parameter constraints:', dialog)
                layout.addWidget(label_constraints)
    
                # Helper function to create a numbered QTextEdit with dynamic height
                def create_numbered_textbox(dialog, number, placeholder_text, initial_text=''):
                    box_layout = QHBoxLayout()
    
                    # Add the enumerator (e.g., 1., 2., etc.)
                    label_number = QLabel(f'{number}.', dialog)
                    box_layout.addWidget(label_number)
    
                    # Create the QTextEdit box
                    text_box_constraints = QTextEdit(dialog)
                    text_box_constraints.setPlainText(initial_text)  # Set initial text
                    text_box_constraints.setPlaceholderText(placeholder_text)
    
                    # Set dynamic height based on text metrics with padding
                    font_metrics = QFontMetrics(text_box_constraints.font())
                    text_height = font_metrics.height()
                    padding = 12  # Small padding for better appearance
                    text_box_constraints.setFixedHeight(text_height + padding)
    
                    box_layout.addWidget(text_box_constraints)
                    layout.addLayout(box_layout)
                    return text_box_constraints
    
                # Get existing constraints if available
                constraints_for_line = []
                if 'constraints' in df.columns:
                    try:
                        constraints_value = df.loc[(df['region_ID'] == frame_id) & (df['Line_Name'] == linename), 'constraints'].item()
                        if isinstance(constraints_value, str):
                            # Try to safely evaluate the string as a Python literal (list)
                            try:
                                constraints_list = ast.literal_eval(constraints_value)
                                if isinstance(constraints_list, list) and len(constraints_list) == 5:
                                    constraints_for_line = constraints_list
                                    print(f'constraints (from string): {constraints_for_line}')
                            except (ValueError, SyntaxError):
                                print(f"Warning: Could not parse constraints string: {constraints_value}")
                                constraints_for_line = ['', '', '', '', '']
                        elif isinstance(constraints_value, list) and len(constraints_value) == 5:
                            constraints_for_line = constraints_value
                            print(f'constraints (from list): {constraints_for_line}')
                        else:
                            constraints_for_line = ['', '', '', '', '']
                    except (KeyError, ValueError):
                        constraints_for_line = ['', '', '', '', '']
                else:
                    constraints_for_line = ['', '', '', '', '']
    
                # Create the constraints text boxes with enumerators and initial values
                text_box_constraints_01 = create_numbered_textbox(dialog, 1, f'e.g., amp <= amp_[line name]', constraints_for_line[0] if len(constraints_for_line) > 0 else '')
                text_box_constraints_02 = create_numbered_textbox(dialog, 2, '', constraints_for_line[1] if len(constraints_for_line) > 1 else '')
                text_box_constraints_03 = create_numbered_textbox(dialog, 3, '', constraints_for_line[2] if len(constraints_for_line) > 2 else '')
                text_box_constraints_04 = create_numbered_textbox(dialog, 4, '', constraints_for_line[3] if len(constraints_for_line) > 3 else '')
                text_box_constraints_05 = create_numbered_textbox(dialog, 5, '', constraints_for_line[4] if len(constraints_for_line) > 4 else '')
    
                # Submit button to send constraints
                submit_button = QPushButton('Submit Constraints', dialog, default=False, autoDefault=False)
                submit_button.clicked.connect(lambda: self.submit_constraints([
                    text_box_constraints_01,
                    text_box_constraints_02,
                    text_box_constraints_03,
                    text_box_constraints_04,
                    text_box_constraints_05
                ], frame_id, lineid))
                layout.addWidget(submit_button)
    
    
                # Increase the width by 10%
                current_width = dialog.width()
                current_height = dialog.height()
                new_width = int(current_width * 2)  # 10% wider
                dialog.resize(350, 100)
    
                # Connect returnPressed for text_box (normal behavior)
                text_box.returnPressed.connect(lambda: [self.on_submit(text_box, data_frame, frame_id, button_name, button_type='line_name'), on_input_confirmed()])
    
                # Show the dialog
                dialog.exec_()

    def submit_constraints(self, text_box_constraints_list, frame_id, lineid):
        global df  # Assuming 'df' is a global DataFrame
        """Handles the submission of the constraints from all text boxes and adds them to the DataFrame."""
        all_constraints_text = [text_box.toPlainText() for text_box in text_box_constraints_list]
        # Ensure all_constraints_text has exactly 5 elements
        while len(all_constraints_text) < 5:
            all_constraints_text.append('')
        all_constraints_text = all_constraints_text[:5]
    
        # Find the row to update
        df = df.reset_index(drop=True)
        linename = df.loc[ (df['region_ID'] == frame_id) & (df['Line_ID'] == lineid)]['Line_Name']
        mask = (df['region_ID'] == frame_id) & (df['Line_ID'] == lineid)
    
        if 'constraints' not in df.columns:
            df['constraints'] = [[] for _ in range(len(df))]  # Initialize with empty lists
        for index, row in df.iterrows():
            if mask[index]:
                df.at[index, 'constraints'] = all_constraints_text
            elif not df.at[index, 'constraints'] or not isinstance(df.at[index, 'constraints'], list) or len(df.at[index, 'constraints']) != 5:
                # Only initialize with blank strings if the 'constraints' column
                # doesn't exist or is not a valid 5-element list
                df.at[index, 'constraints'] = ['', '', '', '', '']
    
        print(f"Constraints submitted for region_ID: {frame_id}, Line_Name: {linename}")
        print("Constraints:", all_constraints_text)



    def on_submit(self, text_box, data_frame, frame_id, button_name, button_type):
        global df_cont, df, snrmap, snr_value
        """Submit the value from the text box, update the dataframe, and adjust related parameters."""
        
        if button_type == 'SNR':
            snr_value = np.float64(text_box.text())
            linewl = df.loc[(np.int64(df['region_ID'])==frame_id) & 
                            (np.float64(df['Line_ID'])==np.int64(button_name.split('~')[0]))]['Centroid_0']
            self.viewer_window.ax.cla()
            self.viewer_window.draw_image(FITS_DATA,cmap='gray',scale='linear',from_fits=True)
            # snrmap = self.calculate_snr_map(linewl, snr_value, search_window_factor=5, continuum_offset_factor=20, continuum_width_factor=30)
            snrmap = self.calculate_snr_map(linewl, snr_value, search_window_width=None, continuum_offset=None, continuum_width=None)
            self.update_button_value(frame_id, button_name, snr_value)
        if button_type == 'obs_button':
            new_value = np.float64(text_box.text()) if 'name' not in button_name else str(text_box.text())
            data_frame[button_name] = new_value
            self.update_button_value(frame_id, button_name, new_value)
            
        elif button_type == 'param_limit':
            if type(text_box.text()) == str:
                new_value = np.float64(text_box.text())
                # new_value = text_box.text()
            # else:
            #     new_value = np.float64(text_box.text())

            df.loc[(df['region_ID']==frame_id) & 
                            (np.int64(df['Line_ID']==np.int64(button_name.split('~')[0]))),button_name.split('~')[1]] = new_value
            self.update_button_value(frame_id, button_name, new_value)
            
        elif button_type == 'line_param':
            try:
                new_value = np.float64(text_box.text())
            except ValueError:
                print(f"Invalid value: {text_box.text()}")
                return
            param = button_name.split('~')[1]
            line_id = np.int64(button_name.split('~')[0])
            df.loc[(np.int64(df['region_ID']) == frame_id) &
                   (np.int64(df['Line_ID']) == line_id), param] = new_value
            self.update_button_value(frame_id, button_name, _fmt(new_value))
            # Redraw the component curves to reflect the new initial guess
            vw = self.viewer_window
            for line in vw.spectrum_ax.lines[:]:
                if line.get_linestyle() in ('--', 'dashed'):
                    try: line.remove()
                    except ValueError: pass
            vw.gaussian_component_lines.clear()
            for actor in df_cont.loc[df_cont["region_ID"] == frame_id, 'lineactor']:
                if actor and actor in vw.spectrum_ax.lines:
                    try: actor.remove()
                    except ValueError: pass
            vw.rebuild_plot(region_ID=frame_id, from_file=False, show_init=True, show_fit=False, x=0, y=0)

        elif button_type == 'RestWavelength':
            new_value = ast.literal_eval(text_box.text())
            df.loc[(df['region_ID']==frame_id) & 
                            (np.int64(df['Line_ID']==np.int64(button_name.split('~')[0]))),'Rest Wavelength'] = new_value
            self.update_button_value(frame_id, button_name, text_box.text())
        
        else:
            
            try:
                
                # Extract new value from the text box
                new_value = np.float64(text_box.text()) if 'Name' not in button_name else str(text_box.text())
        
                # Identify which parameter is being updated
                param = button_name.split('~')[1]
                
            except ValueError:
                print(f"Invalid input: {text_box.text()} is not valid for this column.")
        
                if button_type == 'continuum':
                    
                    # Directly update the dataframe with the new value
                    data_frame.loc[data_frame['region_ID'] == frame_id, param] = new_value
                    
                    # **Important**: Fetch updated values *after* setting the new value
                    x1 = data_frame.loc[data_frame['region_ID'] == frame_id, 'x1'].values[0]
                    x2 = data_frame.loc[data_frame['region_ID'] == frame_id, 'x2'].values[0]
                    slope = data_frame.loc[data_frame['region_ID'] == frame_id, 'Slope_0'].values[0]
                    intercept = data_frame.loc[data_frame['region_ID'] == frame_id, 'Intercept_0'].values[0]
    
        
                    # Adjust other values based on what changed
                    if param == 'x1' or param == 'x2':
                        # Ensure y1 and y2 remain fixed, update slope and intercept accordingly
                        y1 = slope * x1 + intercept
                        y2 = slope * x2 + intercept
                        slope = (y2 - y1) / (x2 - x1) if x1 != x2 else 0
                        intercept = y1 - slope * x1
            
                        # **Re-store slope & intercept in dataframe** after updating them
                        data_frame.loc[data_frame['region_ID'] == frame_id, 'Slope_0'] = slope
                        data_frame.loc[data_frame['region_ID'] == frame_id, 'Intercept_0'] = intercept
            
                    elif param == 'Slope_0':
                        # Recalculate y-values based on the new slope
                        y1 = slope * x1 + intercept
                        y2 = slope * x2 + intercept
            
                    elif param == 'Intercept_0':
                        # Recalculate y-values based on the new intercept
                        y1 = slope * x1 + intercept
                        y2 = slope * x2 + intercept
            
                    # **Ensure UI updates instantly**: Force refresh of button values
                    self.update_button_value(frame_id, 'continuum~x1', x1)
                    self.update_button_value(frame_id, 'continuum~x2', x2)
                    self.update_button_value(frame_id, 'continuum~Slope_0', slope)
                    self.update_button_value(frame_id, 'continuum~Intercept_0', intercept)
            
                    # **Redraw the updated line with correct values**
                    self.viewer_window.update_line_with_slope_intercept(x1, x2, slope, intercept)
        
            if button_type == 'line':
                df.loc[(df['region_ID'] == np.float64(frame_id)) & (df['Line_ID'] == np.float64(button_name.split('~')[0])), param] = new_value
                self.update_button_value(frame_id, button_name, new_value)
                df_cont.loc[(df_cont["region_ID"] == frame_id)]['lineactor'].item().remove()
                self.viewer_window.spectrum_canvas.draw_idle()  # Efficient redrawing
                self.viewer_window.rebuild_plot(frame_id, from_file=False,show_init=True,show_fit=False,x=0,y=0)
            
            if button_type == 'line_name':
                df.loc[(df['region_ID'] == np.float64(frame_id)) & (df['Line_ID'] == np.float64(button_name.split('~')[0])), param] = new_value
                print(f'frame id: {frame_id}, button name: {button_name}, new name: {new_value}')
                self.update_button_value(frame_id, button_name, new_value)
                df_cont.loc[(df_cont["region_ID"] == frame_id)]['lineactor'].item().remove()
                self.viewer_window.spectrum_canvas.draw_idle()  # Efficient redrawing
                self.viewer_window.rebuild_plot(frame_id, from_file=False,show_init=True,show_fit=False,x=0,y=0)
    
        # Hide the text box after submission
        if (button_type != 'line_name') & (button_type != 'SNR'):
            text_box.hide()

def load_pixmap(image_path):
    """Load QPixmap with proper error handling"""
    pixmap = QPixmap()
    if not pixmap.load(image_path):
        # Try alternative loading method
        with open(image_path, 'rb') as f:
            pixmap.loadFromData(f.read())
    if pixmap.isNull():
        raise ValueError(f"Failed to load image: {image_path}\n"
                       f"Supported formats: {', '.join(QPixmap.supportedImageFormats())}")
    return pixmap

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = Path(__file__).parent

    # Try multiple possible locations
    search_paths = [
        base_path / relative_path,
        base_path / 'Resources' / relative_path,
        Path.cwd() / relative_path
    ]
    
    for path in search_paths:
        if path.exists():
            print(f"Found resource at: {path}")
            return str(path)
    
    raise FileNotFoundError(f"Resource not found in any of: {search_paths}")
    
    
def load_stylesheet(app,filename):
    try:
        stylesheet_path = resource_path(filename)
        print(f"Loading stylesheet from: {stylesheet_path}")
        
        with open(stylesheet_path, "r") as f:
            style = f.read()
            app.setStyleSheet(style)
    except Exception as e:
        print(f"Error loading stylesheet: {e}")
        # Fallback to basic dark theme
        app.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
        """)    
    
def main():
    """Entry point for Hypercube spectral analysis tool."""
    try:
        app = QApplication(sys.argv)
        load_stylesheet(app, 'QDarkOrange_style.qss')
        
        
        print("Application starting...")
        
        # Load splash image
        try:
            image_path = resource_path("hypercube_logo.png")
            print(f"Loading image from: {image_path}")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at: {image_path}")
            
            pixmap = load_pixmap(image_path)
            pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            print("Image loaded successfully")
            
        except Exception as e:
            print(f"Image loading failed: {str(e)}")
            # Fallback to blank pixmap
            pixmap = QPixmap(400, 300)
            pixmap.fill(Qt.white)
        
        
        # Show splash screen
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
        splash.show()
        print("Splash screen shown")
        
        splash.raise_()
        splash.repaint()
        app.processEvents()
        print("Events processed")
        
        # Create main window
        print("Creating main window...")
        viewer = ViewerWindow()
        
        # Set up transition
        def show_main_window():
            print("Closing splash and showing main window")
            splash.finish(viewer)
            viewer.show()
        
        QTimer.singleShot(100, show_main_window)
        print("Timer set")
        
        print("Entering main loop")
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()