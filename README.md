# XSuperlet
Easily calculate various wavelet transforms of X-ray light curves from the terminal or in your own Python scripts.
XSuperlet runs on Python 3.12 or higher and offers several classes for wavelet analysis related tasks:
- Calculate Wavelet/Superlet/WWZ transforms and plot them in a few simple steps.
- Generate artificial light curves from combinations of sinusoids.
- Simulate light curves based on an observed light curve (Timmer & Koenig or Emmanoulopoulos methods).
- Estimate wavelet significance based on simulated light curves.


# Dependencies
- Python 3.12 or higher
- `numpy`
- `scipy`
- `matplotlib`
- `sklearn`
- `pyLag` (https://github.com/wilkinsdr/pyLag)
- `wwz` (https://github.com/skiehl/wwz)

# Optional Dependencies
- `pty` for terminal commands (on unix)
- `setproctitle` for setting the process name

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

The `help` command will print the list of all available XSuperlet commands. `command ?` will show information about that command, including the parameters.

---
The XSuperlet program has two optional parameters and two optional flags:

`filename` Name of light curve file to load on startup.

`binsize` Size of light curve time bins in seconds, defaults to 0 (No rebinning).

`-c`, `--command` Load the given text file at start and run the commands within it, one command per line.

`-p`, `--processes` Set the number of processes to use for parallelised methods (Light curve simulation and significance estimation), defaults to 1 (Serial only).

# Config File Options
`BIN_SIZE`: Sets the default bin size in time units, light curves will be automatically rebinned by this time. `0` disables automatic rebinning (Recommended).

`FREQ_GRID_MIN`: Sets the default minimum frequency of the transform frequency grid in μHz.

`FREQ_GRID_MAX`: Sets the default maximum frequency of the transform frequency grid in μHz.

`FREQ_GRID_SIZE`: Sets the number of frequency bins in grid.

`CWT_CYCLES`: Sets the default number of wavelet cycles for the CWT.

`SLT_BASE_CYCLES`: Sets the default base wavelet cycles for the SLT.

`SLT_MIN_ORDER`: Sets the default minimum superlet order for the SLT.

`SLT_MAX_ORDER`: Sets the default maximum superlet order for the SLT.

`WWZ_FREQS`: Sets the default number of frequency bins for the WWZ.

`WWZ_TBIN_SIZE`: Sets the default size of WWZ time bins.

`PEAK_HEIGHT`: Sets the default minimum peak height for signal detection.

`MIN_PROMINENCE`: Sets the default minimum prominence level for signal detection.

`PEAK_INTERVAL_OFFSET`: Sets the default offset of peak detection with `trace-frequency`.

`AUTO_SCALE_MIN`: Sets the default minimum percentile for scalogram auto-scaling.

`AUTO_SCALE_MAX`: Sets the default maximum percentile for scalogram auto-scaling.

`[USER_SHORTCUTS]`: These lists define acceptable shortcuts for referring to light curves or specific transforms.

---
The following settings control the units displayed on plots. Note that currently the internal units used are always megaseconds and microhertz.

`FREQUENCY`: Frequency display unit. Must be set so that frequency is strictly $\geq 1$. Must use correct capitalisation (See `time_units.py`)

`TIME`: Time display unit.

`PERIOD`: Period display unit.

---
`SCALOGRAM_AUTO_SCALE`: Boolean. If True and no min/max given when plotting a scalogram, the scale will be set to be between the two percentiles set above.
If the COI has been calculated the limits are set considering valid points in the transform only.

`LOGGING`: Boolean. If True commands are logged to the current day's XSuperlet log file.

`ZERO_TIME_SERIES`: Boolean. If True the start of the light curve is forced to zero in pyLag (Not recommended).

`SUPPRESS_WARNINGS`: Boolean. If True any runtime warnings from numpy or similar will be suppressed.

`FREQ_BIN_SCALE`: String (Either log or linear). Sets the spacing of the frequency grid. Logarithmic is recommended.

`SHOW_GP_RESULT`: Boolean. If True XSuperlet will show the result of Gaussian processes after completion, allowing the user to verify them. 

`SIMULATION_TYPE`: String (Either TK or EM). Selects the type of light curve simulation used. TK is faster, EM models the PDF of the real light curve.

`WWZ_VERBOSE`: Integer. Sets the verbosity of output from `wwz.py`. 0 is none, 1 is some, 2 is all.

`GAP_FILL_COLOUR`: Sets the fill colour of light curve gaps on scalograms. Format is ("Colour", alpha).

`COI_FILL_COLOUR`: Sets the fill colour of the cone of influence on scalograms. Format is ("Colour", alpha).