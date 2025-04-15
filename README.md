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
   - [Python Version](#python-version)
   - [Standalone Version](#standalone-version)
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
Installation and use of this tool has been tested on MacOS and Windows, it has not yet been tested on Linux operating systems. The first step is to clone the repository to a directory on your local machine where you have read/write/execute privileges. 

### Python version
The tool was designed for quick and painless installation using `conda` environment management via the included environment file `hypercube.yml`. In a terminal, from your base conda environment, navigate to the new HyperCube directory and issue the following command:

```
conda env create -f hypercube.yml
```

Conda will install all of the required packages automatically. If not using conda, you can manually install the required packages (listed in hypercube.yml) via `pip`.

### Standalone version
The repository also comes with a `hypercube.spec` file for use with [pyinstaller](https://github.com/pyinstaller/pyinstaller), in order to package a standalone app version of HyperCube. From a Python console (conda or otherwise), install pyinstaller:

```
pip install pyinstaller
```

Next, navigate to your HyperCube directory and install the HyperCube standalone app:

```
pyinstaller hypercube.spec
```

This will generate a `dist` folder which contains hypercube.app, which can be double-clicked to open the tool. You can create a shortcut to this application from anywhere on your machine.

# Quick Start Guide
This guide walks you through a basic analysis of a Keck Cosmic Wave Imager (KCWI) data cube observation of the luminous infrared galaxy IRAS F23365+3604. The purpose of this guide is to familiarize you with the basic features and modes available to you when using HyperCube to fit 3D spectral data, it is not intended as a comprehensive introduction to every feature the tool offers.

From your new `hypercube` conda environment, open the tool via the following command:

```
python hypercube.py
```

This should launch the main application window. You can now load the IFS data using the `Open FITS` button on the bottom right of the application window and selecting the file `IRAS_F23365+3604.fits`. The main application (visualizer) window will now show to panels: on the left is the image viewer, which initially shows a white light image of the galaxy (from integrating the spectrum in each spectral pixel, or "spaxel"), on the right is a live spectrum viewer that updates as you move the cursor across the white light image. 

### Interacting with the Image Viewer Panel
As the cursor is moved around the image, an orange rectangle indicates the currently focused spaxel. You can lock the spaxel by pressing the `L` key. To unlock, move the cursor back to the image viewer panel and press `L` again.

### Interacting the the Spectrum Viewer Panel
The spectrum viewer panel shows the spectrum contained in the currently-selected spaxel. You can zoom into a portion of the spectrum by clicking and dragging across the spectrum. As you do, a grey rectangular region will indicate the range that will be zoomed to when you release click. The new horizontal (spectral) range reflects the one you selected, while the new vertical (signal or flux) range is auto-scaled to show the continuum and the peaks of any lines in that spectral window. Right-click anywhere on the spectrum to bring up a `reset zoom` button which can be clicked to set the spectrum viewer window to its original range.

### Draw Continuum and Gaussians to Initialize a Model
Select and lock onto a spaxel containing a spectrum with nice, bright emission lines (in **Fig. 1**, we've locked onto position x=16, y=10), then zoom into the Hα-[N II] line complex (6940--7040 Å). With the cursor hovered over the continuum at ~6950 Å, press the `d` key to start placing your linear continuum model. One end of the line remains locked at the initial position, while the free end follows the cursor. Move to the continuum at ~7030 Å and press the `d` key once more to lock in the continuum model; this will bring up the parameter window which we can ignore for now.

<img width="985" alt="Screenshot 2025-04-15 at 12 28 36 PM" src="https://github.com/user-attachments/assets/e0862f7a-3ea1-4185-8827-fb76939fa7d2" />

**Figure 1:** <em>HyperCube visualizer window showing the white light image of IRAS F23365+3604 (left) and spectrum of spaxel (x,y)=(16,10) zoomed into the Hα-[N II] region of the spectrum (right). The solid green and orange dashed lines overlaid on the spectrum represent the currently-defined model and model components, respectively.</em>

Now that a linear continuum model has been placed, we can start to place the Gaussians. With the cursor at the position of the peak of the [N II] 6548 Å line (redshifted to ~6970 Å in this case, press the `g` key to initialize a Gaussian model: horizontal mouse movement affects the Gaussian width, vertical mouse movement affects the amplitude. When you are satisfied with the Gaussian, press the `g` key again to lock it. Repeat this for the other two emission lines Hα and [N II]6584 Å. Congratulations, you have now interactively specified the initial parameter values for a spectral model composed of a continuum line and three Gaussians! 

To inspect the parameters of the model, bring the fit parameters window to the foreground (it should already be open but hidden behind the visualizer window). Scroll down until you see the "Spectral Region 1" panel, containing all of the initial parameter guesses for your model. If you were to go back to the visualizer window and place a line+Gaussians to, say, the [S II] line doublet in the same spectrum, you would see a "Spectral Region 2" panel in the fit parameters window. Click the `Line Name` button in the first row, corresponding to the first Gaussian you placed. This will bring up the "Line Name and Parameter Constraints" window for this emission line (**Fig. 2**). Replace "Line 0" with a name of your choice and press enter -- a green checkmark will notify you the name has been accepted -- then close the window. Repeat this for the other two lines. Next, specify the rest wavelength for your emission lines (in the same units as in your spectra) by clicking the `λ_rest` buttons for each line: 6548, 6563, and 6584 Å. 

<img width="785" alt="Screenshot 2025-04-15 at 1 13 48 PM" src="https://github.com/user-attachments/assets/898b9316-a4df-4436-84ad-713949a2be28" />


**Figure 2:** <em>Line Name and Parameter Constraints window for one of the emission lines. 



# Interactive Usage Mode

### Initiating Models Interactively

### Adjusting Model Parameters Interactively

### Relational Constraints


# Pipeline Usage Mode

### Initiating Models with Configuration Files

### Batch Processing

*work in progress*

# Troubleshooting

*work in progress*

# Acknowledging HyperCube
If you used HyperCube in your research, please consider acknowledging the use of the tool by including this text in your publications:

_This research has made use of HyperCube, the interactive analysis tool for integral field spectroscopic data, written by Justin A Kader._
