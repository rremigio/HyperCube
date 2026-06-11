<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" width="100%" srcset="https://github.com/user-attachments/assets/223adf80-8792-454e-ad99-94bbef751a5c">
    <img width="100%" alt="auto light/dark mode" src="https://github.com/user-attachments/assets/2e03646a-c6d5-444f-bcf6-bf43dfd4dd5c">
  </picture>
</div>


HyperCube is a python-based spectral fitting tool designed to make integral field spectroscopic (IFS), or hyperspectral data analysis more interactive and intuitive, while preserving automation and repeatability. The tool combines a user-friendly [PyQT5](https://github.com/PyQt5) GUI with the robust and flexible fitting capabilities of [lmfit](https://github.com/lmfit/lmfit-py), and is particularly well-suited for interactive and batch process spectral modeling of 3D spectral data.


---

## What's New in v0.3.0

- **Kinematic groups (K1–K5).** Tie the velocity *and* velocity dispersion of multiple emission lines into a single kinematic solution with one click, directly from the Line Name window. Members share the same velocity and the same km/s dispersion (widths and centroids scaled by each line's rest wavelength); the group's reference line — the one carrying the free kinematics — is surfaced in the UI.
- **Velocity dispersion in km/s.** σ is now displayed and edited in km/s throughout the GUI (parameter buttons, column headers, σ maps) and exported alongside the wavelength-space values in the CSV and FITS outputs.
- **Constraint workflow upgrades.** A syntax **?** help button and an **Auto-suggest constraints** button in the Line Name window, plus clearer "constraints saved" feedback.
- **Constraint correctness fixes (important).** Relational constraints referencing bracketed forbidden-line names (`[S II]`, `[N II]`, `[O III]`, …) were being silently dropped — now fixed. A separate bug that let amplitude constraints be lost during per-spaxel flux rescaling is also fixed, so constraints are now reliably respected in every spaxel.
- **UI fix.** Checkbox check-marks now render correctly in the dark theme.

---

## Table of Contents
1. [Installation](#installation)
   - [Source Version](#hypercube-source-version)
   - [Standalone Version](#hypercube-standalone-version)
2. [Quick Start Guide](#quick-start-guide)
3. [Interactive Usage Mode](#interactive-usage-mode)
   - [Initiating Models Interactively](#initiating-models-interactively)
   - [Adjusting Model Parameters Interactively](#adjusting-model-parameters-interactively)
   - [Per-Spaxel Fit Correction](#per-spaxel-fit-correction)
   - [Saving and Restoring Sessions](#saving-and-restoring-sessions)
   - [Relational Constraints](#relational-constraints)
4. [Stellar Kinematics with pPXF](#stellar-kinematics-with-ppxf)
5. [Pipeline Usage Mode](#pipeline-usage-mode)
   - [Initiating Models with Configuration Files](#initiating-models-with-configuration-files)
   - [Batch Processing](#batch-processing)
6. [Troubleshooting](#troubleshooting)
7. [Acknowledging HyperCube](#acknowledging-hypercube)

---



# Installation
Installation and use of this tool has been tested on MacOS and Windows, it has not yet been tested on Linux operating systems. The first step is to clone the repository to a directory on your local machine where you have read/write/execute privileges:
```
git clone https://github.com/jkader925/HyperCube.git
```
This will create a "HyperCube" directory containing the distribution. Alternatively, download source files as a .zip and unpack to desired location.

### HyperCube Source Version
The tool was designed for quick and painless installation using `conda` environment management via the included environment file `hypercube.yml`. In a terminal, from your base conda environment, navigate to the new HyperCube directory and issue the following command:

```
conda env create -f hypercube.yml
```

Conda will install all of the required packages automatically. If not using conda, you can manually install the required packages (listed in hypercube.yml) via `pip`.

> **Stellar kinematics fitting** (the [Stellar Kinematics with pPXF](#stellar-kinematics-with-ppxf) section) additionally requires the [`ppxf`](https://pypi.org/project/ppxf/) package. If it is not already in your environment, install it with `pip install ppxf`. The bundled stellar template libraries live in the `indo_us_library/` and `eMILES/` folders and ship with the distribution.

#### Updating the Source Version
 
New releases may require additional packages. To bring your existing environment up to date, first get the latest code — if you cloned the repository with git:
 
```
git pull origin main
```
 
Or download the latest zip from the [Releases](https://github.com/jkader925/HyperCube/releases) page and replace the contents of your HyperCube folder with the new files. Then update your conda environment from your base conda environment:
 
```
conda env update -f hypercube.yml --prune
```
 
The `--prune` flag removes any packages that are no longer needed. If you run into environment conflicts, a clean rebuild is the most reliable fix:
 
```
conda env remove -n HyperCube
conda env create -f hypercube.yml
```
 
Then launch as normal:
 
```
python hypercube.py
```
 
---

### Hypercube Standalone Version
The repository also comes with a `hypercube.spec` file for use with [pyinstaller](https://github.com/pyinstaller/pyinstaller), in order to package a standalone app version of HyperCube. From a Python console (conda or otherwise), install pyinstaller:

```
pip install pyinstaller
```

Next, navigate to your HyperCube directory and install the HyperCube standalone app:

```
pyinstaller hypercube.spec
```

This will generate a `dist` folder which contains hypercube.app, which can be double-clicked to open the tool. You can create a shortcut to this application from anywhere on your machine.

#### Updating the Standalone Version
 
The app does not update automatically — each new version requires a fresh build. First get the latest code by downloading the latest zip from the [Releases](https://github.com/jkader925/HyperCube/releases) page and replacing the contents of your HyperCube folder, or if you cloned with git:
 
```
git pull origin main
```
 
Then rebuild from your HyperCube directory:
 
```
pyinstaller hypercube.spec
```

This regenerates the `dist` folder with an updated `hypercube.app`. Replace your existing app with the new one from `dist/`. If you have a shortcut or dock icon pointing to the old ap, update it to point to the newly built version.

---
 
# Quick Start Guide
This guide walks you through a basic analysis of a Keck Cosmic Wave Imager (KCWI) data cube observation of the luminous infrared galaxy IRAS F23365+3604. The purpose of this guide is to familiarize you with the basic features and modes available to you when using HyperCube to fit 3D spectral data, it is not intended as a comprehensive introduction to every feature the tool offers.

From your new `hypercube` conda environment, open the tool via the following command:

```
python hypercube.py
```

This should launch the main application window. You can now load the IFS data using the `Open FITS` button on the bottom right of the application window and selecting the file `IRAS_F23365+3604.fits`. The main application (visualizer) window will now show to panels: on the left is the image viewer, which initially shows a white light image of the galaxy (from integrating the spectrum in each spectral pixel, or "spaxel"), on the right is a live spectrum viewer that updates as you move the cursor across the white light image. 

### Interacting with the Image Viewer Panel
As the cursor is moved around the image, an orange rectangle indicates the currently focused spaxel. You can lock the spaxel by pressing the `L` key. To unlock, move the cursor back to the image viewer panel and press `L` again.

### Interacting with the the Spectrum Viewer Panel
The spectrum viewer panel shows the spectrum contained in the currently-selected spaxel. You can zoom into a portion of the spectrum by clicking and dragging across the spectrum. As you do, a grey rectangular region will indicate the range that will be zoomed to when you release click. The new horizontal (spectral) range reflects the one you selected, while the new vertical (signal or flux) range is auto-scaled to show the continuum and the peaks of any lines in that spectral window. Right-click anywhere on the spectrum to bring up a `reset zoom` button which can be clicked to set the spectrum viewer window to its original range.

### Draw Continuum and Gaussians to Initialize a Model
Select and lock onto a spaxel containing a spectrum with nice, bright emission lines (in **Fig. 1**, we've locked onto position x=16, y=10), then zoom into the Hα-[N II] line complex (6940--7040 Å). With the cursor hovered over the continuum at ~6950 Å, press the `d` key to start placing your linear continuum model. One end of the line remains locked at the initial position, while the free end follows the cursor. Move to the continuum at ~7030 Å and press the `d` key once more to lock in the continuum model; this will bring up the parameter window which we can ignore for now.

<div align="center">
  <picture>
<img width="985" alt="Screenshot 2025-04-15 at 12 28 36 PM" src="https://github.com/user-attachments/assets/e0862f7a-3ea1-4185-8827-fb76939fa7d2" />
  </picture>
</div>

**Figure 1:** <em>HyperCube visualizer window showing the white light image of IRAS F23365+3604 (left) and spectrum of spaxel (x,y)=(16,10) zoomed into the Hα-[N II] region of the spectrum (right). The solid green and orange dashed lines overlaid on the spectrum represent the currently-defined model and model components, respectively.</em>

Now that a linear continuum model has been placed, we can start to place the Gaussians. With the cursor at the position of the peak of the [N II] 6548 Å line (redshifted to ~6970 Å in this case, press the `g` key to initialize a Gaussian model: horizontal mouse movement affects the Gaussian width, vertical mouse movement affects the amplitude. When you are satisfied with the Gaussian, press the `g` key again to lock it. Repeat this for the other two emission lines Hα and [N II]6584 Å. Congratulations, you have now interactively specified the initial parameter values for a spectral model composed of a continuum line and three Gaussians! 

### Final Preparations and Fitting the Cube
To inspect the parameters of the model, bring the fit parameters window to the foreground (it should already be open but hidden behind the visualizer window). Scroll down until you see the "Spectral Region 1" panel, containing all of the initial parameter guesses for your model. If you were to go back to the visualizer window and place a line+Gaussians to, say, the [S II] line doublet in the same spectrum, you would see a "Spectral Region 2" panel in the fit parameters window. Click the `Line Name` button in the first row, corresponding to the first Gaussian you placed. This will bring up the "Line Name and Parameter Constraints" window for this emission line (**Fig. 2**). Replace "Line 0" with a name of your choice and press enter -- a green checkmark will notify you the name has been accepted -- then close the window (red 'x' or `esc` key). Repeat this for the other two lines. Next, specify the rest wavelength for your emission lines (in the same units as in your spectra) by clicking the `λ_rest` buttons for each line: 6548, 6563, and 6584 Å. **Save this configuration for later use by pressing `Cmd+S` (mac) or `Ctr+S` (win).**

<div align="center">
  <picture>
<img width="785" alt="Screenshot 2025-04-15 at 1 13 48 PM" src="https://github.com/user-attachments/assets/898b9316-a4df-4436-84ad-713949a2be28" />
  </picture>
</div>

**Figure 2:** <em>The Fit Parameters window displays all model parameter information as well as pertinent information about the observation and the selected spaxel. Here, one of the line name buttons has been pressed, bringing up the Line Name and Parameter Constraints windows.</em>

For this simple example, let's leave the parameter limits at their initial values and forego specifying any relational constraints between model parameters. We can input the observation details at the top of the Fit Parameters window, in the "Observation Data" panel. The tool will attempt to scrape the source name from the FITS header, but if that fails, or if you want to change the name, click the Source Name button. For this observation, the Source Redshift is 0.064 and the Resolving Power is 4000. Now we are ready to fit the cube! Press the `Fit Cube` button in the "Spectral Fitting" panel at the top right of the window. For this example, we are using a cropped version of the full cube, containing only around 800 spaxels, so the fit will only take a few seconds to complete. 

### Inspecting the Fit
First, let's visually inspect the model fit to the spectrum by bringing back the Visualizer window. Unlock the current spaxel in the white light image by pressing `L`, then, as the cursor moves around the image the Spectrum Viewer window will show the spectrum and the best fit model (**Fig. 3**). The total model is represented with a solid red line and the individual Gaussians are each assigned a unique color. The reduced chi-square value of the fit is shown to the top-right of the Spectrum Viewer panel. 

<div align="center">
  <picture>
<img width="947" alt="Screenshot 2025-04-15 at 1 24 10 PM" src="https://github.com/user-attachments/assets/2cef7095-efaf-425b-a806-6a7d149482d8" />
  </picture>
</div>

**Figure 3:** <em>After fitting the cube, the Spectrum Viewer panel (right) shows the spectrum + best-fitting model in the spaxel currently highlighted in the Image Viewer panel (left). The total model is shown in red, model components are shown with dashed lines colored-coded according to the Fit Parameters window. The reduced Chi-square value for the fit is shown to the top-right of the Spectrum Viewer panel.</em>

To inspect the fitted values of each parameter for each model component, e.g., the continuum slope or the amplitude (flux density) of the Hα line, bring the Fit Parameters Window forward. Like the Spectrum Viewer panel, the fitted values will update in realtime to reflect the best-fit model in the currently-selected spaxel in the Image Viewer panel. Any of the parameter_fit buttons can be pressed to show the spatially-resolved map of that fitted parameter in the Image Viewer panel (**Fig. 4**). 

<div align="center">
  <picture>
<img width="645" alt="Screenshot 2025-04-15 at 1 39 13 PM" src="https://github.com/user-attachments/assets/8b95f6ed-698c-49f4-b1dc-b8eb1fa4cf06" />
  </picture>
</div>

**Figure 4:** <em>Visualizing the fitted values of your model spatially is as easy as clicking the button corresponding to the parameter.</em>

If you are not satisfied with the fit, you can specify parameter limits or parameter constraints to obtain a better result. You can also reset your initial parameter guesses by pressing the red delete button at the far-right of each row of buttons, and retry drawing the Gaussian on the spectrum. In the current version, it is recommended to restart the program, load the FITS file, open the Fit Parameters window and use `Cmd-O` (mac) or `Ctr-O` (win) to load your original configuration.

### Outputting the Fit ###
If you are satisfied with the cube fit, you can save the result to a csv table and/or a multi-extension FITS file. To do this, bring forward the Fit Parameters window and look at the Spectral Fitting panel. Here, you will find the `Save Cube Fit` and `Save Fit to FITS File` buttons, which output the fit to csv and FITS files, respectively. You can always view your fit result in HyperCube at a later time by opening the tool, loading the original FITS cube, opening the Fit Parameters window, and clicking the `Load Cube Fit` button.

# Interactive Usage Mode
One of the two main use cases for HyperCube is intuitive/dynamic spectral fitting (or data exploration), the other being automated/batch spectral fitting (described below in the [Pipeline Usage Mode](#pipeline-usage-mode) section). In interactive mode, spectral fitting more or less follows the steps outlined in the [Quick Start Guide](#quick-start-guide), i.e., we dynamically place continuum+line sets using the cursor and specify parameter values, names, limits, and constraints using the interactive GUI. *This usage mode is ideal for quick exploration of data cubes where visual feedback is critical.*

### Initiating Models Interactively

Each spectral region is built from a **continuum** plus any number of **Gaussian** emission lines drawn on top of it. Three continuum types can be drawn interactively over the spectrum:

- **Linear** (`d`) — press `d` at one end of the continuum and again at the other to set a straight line.
- **Spline** (`s`) — press `s` to drop interpolation knots (connect-the-dots); press `Enter` to finalize, `Backspace` to undo the last knot, `Esc` to cancel.
- **Polynomial** (`p`) — press `p` at the start and end of a wavelength range to fit a Chebyshev polynomial to the data in that range.

Once a continuum is placed, press `g` over a line peak to begin a Gaussian (horizontal motion sets the width, vertical motion the amplitude) and `g` again to lock it. Each continuum + line set becomes a "Spectral Region" panel in the Fit Parameters window.

### Adjusting Model Parameters Interactively

Every value in a Spectral Region panel is an editable button. Click a continuum cell (slope, intercept, knots, polynomial degree) or a line cell (amplitude *f*<sub>λ,0</sub>, observed wavelength λ<sub>obs,0</sub>, width σ<sub>0</sub>, and their min/max limits) to type a new value; the model overlay updates immediately. Line widths (σ) are shown and entered in **km/s** (velocity dispersion). Click `Line Name` to name a line, assign it to a [kinematic group](#relational-constraints), and set relational constraints, and `λ_rest` to set its rest wavelength.

### Per-Spaxel Fit Correction

After fitting the whole cube, individual spaxels can be corrected without re-fitting everything. Lock onto a spaxel (`L`) and use the per-spaxel controls in the **This Spaxel** group at the top of the Fit Parameters window:

- **Fit This Spaxel** — (re)fit the model to just the locked spaxel.
- **Clear Spaxel Fit** — remove this spaxel's fit and enter *edit mode*. The old (poor) model is greyed for reference, and a dialog lets you seed the edit from the spaxel's existing fit or from the base template. You can then re-specify the continuum type and emission-line initial guesses for **this spaxel only** — graphically (`d`/`s`/`p` *replace* the region's continuum in place; `g` snaps to the nearest existing line and updates its guess) or by editing the panel cells. While editing, the model **schema is locked** (the same number of regions/lines and the same line names as every other spaxel) so spatially-resolved line maps never develop NaN holes.
- **Cancel Edit** — discard an in-progress per-spaxel edit and restore the original fit.
- **Toggle Edited** — overlay translucent blue boxes on the image marking every spaxel that has a per-spaxel edit.

Per-spaxel edits are remembered and reloaded whenever you lock back onto that spaxel. To wipe every fit for the whole cube while keeping the model definition (so you can re-fit), use **Clear All Fits** in the cube-level controls.

### Saving and Restoring Sessions

The entire tool state — cube, model, fits, per-spaxel edits, stellar results, and display/background settings — can be saved to a `.hcsession` file with **Save Session** and restored later with **Load Session**, so you can close HyperCube and resume exactly where you left off.

### Relational Constraints

Open the **Line Name and Parameter Constraints** window (click any `Line Name` button) to tie a line's parameters to other lines in the model. Up to five constraints per line can be entered, using the syntax:

```
param  op  param_[line name]
param  op  factor * param_[line name]
```

where `param` is one of `amp`, `sigma`, `cen` (centroid), or `vel` (velocity), and `op` is one of `<=`, `<`, `>=`, `>`, `==`. For example:

- `amp <= amp_[Halpha]` — keep a component fainter than another line
- `sigma >= sigma_[Halpha]` — keep a component broader than another
- `amp <= 0.33 * amp_[Halpha]` — fixed flux-ratio bound
- `vel == vel_[nii_1]` — tie velocities (shared kinematics)

A **?** help button lists the available parameters, operators, and the lines currently in the model. The **Auto-suggest constraints** button proposes sensible constraints based on the components' initial guesses, which you can review before applying. Constraints reference lines by **name**, so forbidden-line names containing brackets (e.g. `[S II]_6716`) are fully supported.

#### Kinematic Groups (K-groups)

For multi-component fits it is often desirable for several lines to share one kinematic solution. Assign lines to the same **K-group** (K1–K5, via the checkboxes in the Line Name window) to tie their **velocity and velocity dispersion** together during fitting — every member shares the same velocity and the same km/s dispersion, with widths and centroids scaled by each line's rest wavelength. The first line of a group (in model order) is the **reference** that carries the group's free kinematics, and the window indicates which line that is. K-groups are a shortcut that writes the equivalent relational constraints for you, and they coexist non-destructively with any manual sigma constraints — a manual constraint is held inactive while the line is grouped and re-activates if you remove it from the group.

> **Note:** velocity dispersion (σ) is displayed and entered in **km/s** throughout the GUI and is included (alongside the wavelength-space values) in the CSV and FITS output.


# Stellar Kinematics with pPXF

HyperCube can model the **stellar continuum** and recover the stellar line-of-sight velocity (V) and velocity dispersion (σ) using the Penalized Pixel-Fitting method ([pPXF](https://pypi.org/project/ppxf/); Cappellari 2017). The stellar fit is integrated as a new **continuum type**: pPXF fits a combination of stellar templates convolved with the line-of-sight velocity distribution (LOSVD), and emission-line Gaussians are added on top exactly as for the linear/spline/polynomial continua.

Two template libraries ship with HyperCube (in the `indo_us_library/` and `eMILES/` folders):
- **Indo-US** — empirical stellar spectra (Valdes et al. 2004); ideal for pure kinematics.
- **eMILES** — single stellar population models with wide wavelength coverage.

### Fitting a stellar continuum to a spaxel
1. Set the **Source Redshift** and **Resolving Power** in the Observation Data panel — pPXF needs both. HyperCube auto-fills them from the FITS header when the relevant keywords are present.
2. Lock onto a spaxel (`L`) and click **Stellar Template…** in the *This Spaxel* group. The Stellar Templates window lets you choose a library (showing its wavelength coverage and resolution against your data's observed and rest-frame ranges), the fit range, the number of LOSVD moments (V, σ or V, σ, h3, h4), additive/multiplicative polynomial degrees, an initial σ guess, and whether to mask emission lines.
3. Click **Fit**. pPXF fits the spaxel, a **Stellar** spectral-region panel appears spanning the fit range with editable **V**, **σ**, and **scale** cells, and the best-fit stellar model is overplotted on the spectrum.

Editing the V, σ, or scale cells re-renders the model instantly (no re-fit); **Refit Stellar** re-runs pPXF for the current spaxel. Add emission lines with `g` on top of the stellar continuum — when you fit, the stellar baseline is held fixed and the lines are fit to the stellar-subtracted residual.

### Stellar maps across the cube
Press **Fit Cube** to fit the stellar continuum *and* the emission lines across every spaxel in one pass (or **Fit Stellar (Cube)** for kinematics only). The **V map** and **σ map** buttons in the Stellar panel then display the spatially-resolved stellar velocity and dispersion maps; the buttons themselves show the current spaxel's best-fit V and σ as you move the cursor. A **Cancel** button in the progress bar stops a long fit, keeping the spaxels already fit.

Stellar results are included when you **Save Fit** (CSV) and **Save Fit to FITS File** (as `stellar_vel`, `stellar_sigma`, … image extensions), and are fully preserved in saved sessions.


# Pipeline Usage Mode

### Initiating Models with Configuration Files

### Batch Processing

*work in progress*

# Troubleshooting

If you get the "UnboundLocalError: cannot access local variable 'piecewise_model' where it is not associated with a value" error, it means you need to add a model to the HyperCube_ModelFunctions.py file because it doesn't yet include a model for your Nregions+Nlines, e.g., it doesn't have one already for 3 continuum regions with one line each (Nregions=3,Nlines=3) -- you'd need to add it manually (following the syntax of the other models in that script).

# Acknowledging HyperCube
If you used HyperCube in your research, please consider acknowledging the use of the tool by including this text in your publications:

_This research has made use of HyperCube, the interactive analysis tool for integral field spectroscopic data, written by Justin A Kader._
