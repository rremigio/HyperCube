<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" width="100%" srcset="https://github.com/user-attachments/assets/223adf80-8792-454e-ad99-94bbef751a5c">
    <img width="100%" alt="auto light/dark mode" src="https://github.com/user-attachments/assets/2e03646a-c6d5-444f-bcf6-bf43dfd4dd5c">
  </picture>
</div>


HyperCube is a python-based spectral fitting tool designed to make integral field spectroscopic (IFS), or hyperspectral data analysis more interactive and intuitive, while preserving automation and repeatability. The tool combines a user-friendly [PyQT5](https://github.com/PyQt5) GUI with the robust and flexible fitting capabilities of [lmfit](https://github.com/lmfit/lmfit-py), and is particularly well-suited for interactive and batch process spectral modeling of 3D spectral data.


---

## Table of Contents
1. [Installation](#installation)
   - [Source Version](#hypercube-source-version)
   - [Standalone Version](#hypercube-standalone-version)
2. [Quick Start Guide](#quick-start-guide)
3. [Interactive Usage Mode](#interactive-usage-mode)
   - [Initiating Models Interactively](#initiating-models-interactively)
   - [Adjusting Model Parameters Interactively](#adjusting-model-parameters-interactively)
   - [Relational Constraints](#relational-constraints)
4. [Pipeline Usage Mode](#pipeline-usage-mode)
   - [Initiating Models with Configuration Files](#initiating-models-with-configuration-files)
   - [Batch Processing](#batch-processing)
4. [Troubleshooting](#troubleshooting)
5. [Acknowledging HyperCube](#acknowledging-hypercube)

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
 
# Updating
 
### Updating — Run from Source
 
**1. Pull the latest code**
 
If you cloned the repository with git:
```
git pull origin main
```
If you downloaded a zip, download the latest zip from the [Releases](https://github.com/jkader925/HyperCube/releases) page and replace the contents of your HyperCube folder with the new files.

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

### Adjusting Model Parameters Interactively

### Relational Constraints


# Pipeline Usage Mode

### Initiating Models with Configuration Files

### Batch Processing

*work in progress*

# Troubleshooting

If you get the "UnboundLocalError: cannot access local variable 'piecewise_model' where it is not associated with a value" error, it means you need to add a model to the HyperCube_ModelFunctions.py file because it doesn't yet include a model for your Nregions+Nlines, e.g., it doesn't have one already for 3 continuum regions with one line each (Nregions=3,Nlines=3) -- you'd need to add it manually (following the syntax of the other models in that script).

# Acknowledging HyperCube
If you used HyperCube in your research, please consider acknowledging the use of the tool by including this text in your publications:

_This research has made use of HyperCube, the interactive analysis tool for integral field spectroscopic data, written by Justin A Kader._
