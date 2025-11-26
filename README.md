# XSuperlet
Easily calculate various wavelet transforms of X-ray light curves from the terminal or in your own Python scripts.
XSuperlet runs on Python 3.12 and offers several classes for wavelet analysis related tasks:
- Calculate Wavelet/Superlet/WWZ transforms and plot them in a few simple steps.
- Generate artificial light curves from combinations of sinusoids.
- Simulate light curves based on an observed light curve (Timmer & Koenig or Emmanoulopoulos methods).
- Estimate wavelet significance based on simulated light curves.


# Dependencies
- Python 3.12
- `numpy 1.x`
- `scipy`
- `matplotlib`
- `sklearn`
- `pyLag` (https://github.com/wilkinsdr/pyLag)
- `wwz` (https://github.com/skiehl/wwz)

# Optional Dependencies
- `pty` for terminal commands (on unix)

# Getting Started
Below is the simplest example showing how to load a light curve and calculate its superlet transform from the terminal
using XSuperlet.

You can start a terminal instance of XSuperlet with:
`python xsuperlet_terminal.py`

To load a light curve (and bin by 100s):
`add-lc my_light_curve.fits 100` Since this is the first light curve loaded it will be given an ID of `0`.

Plot this light curve:
`plot-lc 0`

Calculate the superlet transform with default parameters:
`slt 0`

Plot the scalogram of the superlet transform:
`scalogram 0 s`