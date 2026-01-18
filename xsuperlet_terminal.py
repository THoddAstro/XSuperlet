"""
Terminal control for XSuperlet superlet analysis package.

SuperletTransform.py based on 'Time-frequency super-resolution with superlets' by Moca et al., 2021 Nature Communications.

Uses wwz.py by Kiehlmann, S., Max-Moerbeck, W., & King, O. (2021), 'wwz: Weighted wavelet z-transform code'.

Uses Light curve simulation based on Timmer & Koenig (1995) and Emmanoulopoulos et al. (2013).

Usage
---------
python xsuperlet_terminal.py <FileName> <BinSize>

FileName: - Initial light curve file (Optional)

BinSize - Light curve bin size in seconds (Optional, default 100)

Options
---------
-c --command - List of commands to execute on start

-p --processes - Number of processes to use for parallelised methods (optional, default 1)

---------

Author: Thomas Hodd

Date - 25th November 2025

Version - 1.0.2
"""

# Check for command-line arguments
import argparse

# Input args
parser = argparse.ArgumentParser(description="Wavelet Analysis Package for (X-ray) Light Curves")
parser.add_argument("filename", type=str, nargs="?", help="Initial Light curve file (optional)")
parser.add_argument("binsize", type=int, nargs="?", default=0, help="Bin size in seconds (optional)")
parser.add_argument("-c", "--command", type=str, nargs="?", default=None, help="List of commands to execute on start")
parser.add_argument("-p", "--processes", type=int, nargs="?", default=1, help="Number of processes to use for WWZ/A simulation (optional, default 1)")

# Parse args
pargs = parser.parse_args()
filename = pargs.filename
binsize = pargs.binsize
command_file = pargs.command
PROCESS_COUNT = pargs.processes

# Measure loading time
import time as t

loadstart = t.time()
print("Loading...", end="", flush=True)

import os
import sys
import select

# Try to import pty, which will fail on Windows systems
try:
    import pty
except ModuleNotFoundError:
    UNIX = False
    PREFIX = "XSuperlet"
    SUFFIX = "> "
else:
    UNIX = True
    PREFIX = "[XSuperlet "
    SUFFIX = "]> "

import ast
import readline
import configparser
import pathlib
import warnings
import traceback
from datetime import datetime as dt

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gmean

from xsuperlet import *
from pylag.gaussian_process import GPLightCurve
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel as Constant

from typing import Type, Literal, Callable, Any

home_dir = os.path.dirname(os.path.abspath(__file__))

# Read in user-specified constants
config = configparser.ConfigParser()
config.read(f"{home_dir}/config.ini")

DEFAULT_BIN_SIZE = config.getint("DEFAULTS", "BIN_SIZE")
DEFAULT_MIN_FREQ = config.getfloat("DEFAULTS", "FREQ_GRID_MIN")
DEFAULT_MAX_FREQ = config.getfloat("DEFAULTS", "FREQ_GRID_MAX")
DEFAULT_FREQ_SIZE = config.getint("DEFAULTS", "FREQ_GRID_SIZE")

CWT_DEFAULT_CYCLES = config.getint("DEFAULTS", "CWT_CYCLES")
WWZ_DEFAULT_FREQS = config.getint("DEFAULTS", "WWZ_FREQS")
WWZ_DEFAULT_TBIN = config.getint("DEFAULTS", "WWZ_TBIN_SIZE")
SLT_DEFAULT_BASE_CYCLES = config.getint("DEFAULTS", "SLT_BASE_CYCLES")
SLT_DEFAULT_MIN_ORDER = config.getint("DEFAULTS", "SLT_MIN_ORDER")
SLT_DEFAULT_MAX_ORDER = config.getint("DEFAULTS", "SLT_MAX_ORDER")

PEAK_HEIGHT_DEFAULT = config.getfloat("DEFAULTS", "PEAK_HEIGHT")
MIN_PROMINENCE_DEFAULT = config.getfloat("DEFAULTS", "MIN_PROMINENCE")
PEAK_INTERVAL_OFFSET = config.getint("DEFAULTS", "PEAK_INTERVAL_OFFSET")

LC_PROXIES = ast.literal_eval(config.get("USER_SHORTCUTS", "LC_PROXIES"))
CWT_PROXIES = ast.literal_eval(config.get("USER_SHORTCUTS", "CWT_PROXIES"))
WWZ_PROXIES = ast.literal_eval(config.get("USER_SHORTCUTS", "WWZ_PROXIES"))
WWA_PROXIES = ast.literal_eval(config.get("USER_SHORTCUTS", "WWA_PROXIES"))
SLT_PROXIES = ast.literal_eval(config.get("USER_SHORTCUTS", "SLT_PROXIES"))

LOGGING = config.getboolean("SETTINGS", "LOGGING")
ZERO_TIME_SERIES = config.getboolean("SETTINGS", "ZERO_TIME_SERIES")
FREQ_BIN_SCALE = config.get("SETTINGS", "FREQ_BIN_SCALE")
DEBUG_GP = config.getboolean("SETTINGS", "SHOW_GP_RESULT")
GAP_FILL = ast.literal_eval(config.get("SETTINGS", "GAP_FILL_COLOUR"))
COI_FILL = ast.literal_eval(config.get("SETTINGS", "COI_FILL_COLOUR"))
SIM_TYPE = config.get("SETTINGS", "SIMULATION_TYPE")

# Suppress RuntimeWarnings
if config.getboolean("SETTINGS", "SUPPRESS_WARNINGS"):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print(f"\n{ERRR}RuntimeWarnings Suppressed!{ENDC}")

if PROCESS_COUNT > 1:
    print(f"{WARN}Parallel processes set to use {PROCESS_COUNT} CPUs!{ENDC}")


class Xsuperlet:
    """
    Xsuperlet instance.

    Interface for performing wavelet/superlet analysis of AGN light curves.
    For list of commands type "help"
    """
    def __init__(self):
        self.__user_lightcurves: list[WaveTransform] = []
        self.__user_lightcurve_count = 0

        self.__units = 1E+6
        if FREQ_BIN_SCALE == "log":
            self.__frequency_grid = np.logspace(np.log10(DEFAULT_MIN_FREQ * self.__units), np.log10(DEFAULT_MAX_FREQ * self.__units), DEFAULT_FREQ_SIZE)
        else:
            self.__frequency_grid = np.linspace(DEFAULT_MIN_FREQ * self.__units, DEFAULT_MAX_FREQ * self.__units, DEFAULT_FREQ_SIZE)

        # Dict of available user commands
        self.command_index = {
            "help": self._show_functions,
            "show": self.show,
            "print-traceback": self.print_traceback,
            "command": self.run_command_file,

            "frequency": self.set_freq_range,

            "add-lc": self.add_lightcurve,
            "delete-lc": self.remove_lightcurve,
            "plot-lc": self.plot_lightcurve,
            "export-lc": self.export_lightcurve,

            "rebin-lc": self.rebin_lightcurve,
            "norm-lc": self.norm_lightcurve,
            "new-comp": self.add_component,

            "cwt": self.wavelet_transform,
            "wavelet": self.wavelet_transform,

            "slt": self.superlet_transform,
            "superlet": self.superlet_transform,

            "wwz": self.wwz_transform,

            "gpslt": self.gaussian_process_slt,
            "gauss-slt": self.gaussian_process_slt,

            "coi": self.calculate_coi,
            "gaps": self.find_gaps,

            "fslice": self.vertical_slice,
            "trace-frequency": self.trace_frequency_peaks,
            "untrace-frequency": self.remove_frequency_peaks,

            "simulate-lc": self.simulate_lightcurves,
            "fit-pdf": self.pdf_fit,
            "import-lcs": self.import_sim_lightcurves,

            "significance": self.calculate_significance,
            "set-significance": self.set_significance,
            "load-significance": self.load_significance,
            "clr-significance": self.clear_significance,

            "scalogram": self.plot_scalogram,
            "save-transform": self.export_transform,
            "difference": self.difference_scalogram
        }

        self.last_traceback = None

    def __get_lc(self, identifier) -> int | bool:
        return next((i for i, lightcurve in enumerate(self.__user_lightcurves) if lightcurve.code == int(identifier)), False)

    def __get_wave_transform_object(self, lc_code) -> WaveTransform:
        return self.__user_lightcurves[self.__get_lc(lc_code)]

    # Static methods for input validation
    @staticmethod
    def __get_transform(transform_str: str) -> Literal["CWT", "WWZ", "WWA", "SLT"] | None:
        try:
            transform_str.lower()
        except AttributeError:
            print(f"{ERRR}'{transform_str}' is not a valid transform{ENDC}")
            return None
        else:
            if transform_str.lower() in CWT_PROXIES:
                return "CWT"
            elif transform_str.lower() in WWZ_PROXIES:
                return "WWZ"
            elif transform_str.lower() in WWA_PROXIES:
                return "WWA"
            elif transform_str.lower() in SLT_PROXIES:
                return "SLT"
            else:
                print(f"{ERRR}'{transform_str}' is not a valid transform{ENDC}")
                return None

    @staticmethod
    def __validate_type(object_type: Type, item: Any, par: str) -> bool:
        """
        Validates a non-default parameter.

        :param object_type: The type to validate: int, str, float, etc...
        :param item: Given parameter value
        :param par: Parameter name
        :return: True if given value can be an 'object_type', else False
        """
        try:
            object_type(item)
        except ValueError:
            print(f"{ERRR}Invalid input '{item}' for parameter <{par}>{ENDC}")
            return False
        except TypeError:
            return False

        return True

    @staticmethod
    def __validate_default_type(object_type: Type, item: Any, par: str, default) -> Any:
        """
        Validates a default parameter. Returns the default value if input value is invalid.

        :param object_type: The type to validate: int, str, float, etc...
        :param item: Given parameter value, None if default
        :param par: Parameter name
        :param default: Default parameter value
        :return: Valid given parameter value, else default parameter value
        """
        if item is None:
            print(f"{WARN}No value specified for parameter <{par}>, defaulting to {default}{ENDC}")
            return default

        if item == "":
            return default

        try:
            item = object_type(item)
        except ValueError:
            print(f"{ERRR}Invalid value '{item}' for parameter <{par}> - defaulting to {default}{ENDC}")
            return default
        else:
            return item

    @staticmethod
    def __force_list_size(input_list: list, size: int) -> list:
        """
        Forces a list to be of a given size, missing values are set to None.

        :param input_list: List to set size
        :param size: Size to set list to
        :return: Resized list
        """
        result = [None] * size
        for i in range(min(len(input_list), size)):
            result[i] = input_list[i]

        return result

    @staticmethod
    def _validate_lc(func: Callable) -> Callable | None:
        def __validation_wrapper(self, lc_code: int, *args, **kwargs) -> Callable | None:
            """
            Validates the light curve ID sent to a function before continuing (if valid)
            or issuing an error message (if invalid).

            :param self: Xsuperlet instance
            :param lc_code: Light curve identifier
            :return: Validated function or None if validation failed
            """
            # Check ID is a valid int
            if self.__validate_type(int, lc_code, "ID"):
                index = self.__get_lc(lc_code)

                # Check ID exists
                if index is not False:
                    return func(self, lc_code, *args, **kwargs)
                else:
                    print(f"{ERRR}ID '{lc_code}' does not exist{ENDC}")
                    return None
            else:
                print(f"{ERRR} '{lc_code}' is not a valid light curve ID{ENDC}")
                return None

        # Ensure the real function metadata is passed back, not the wrapper's name and docstring
        __validation_wrapper.__name__ = func.__name__
        __validation_wrapper.__doc__ = func.__doc__
        return __validation_wrapper

    @staticmethod
    def _help(func: Callable) -> Callable | None:
        def __fetch_help_wrapper(parent, *args, **kwargs) -> Callable | None:
            """
            Prints the docstring instead of calling the method when "?" is passed.

            :param parent: Root Xsuperlet instance
            :return: Either the requested method or its docstring (via print)
            """
            if len(args) > 0:
                if args[0] == "?":
                    print(f"{CYAN}Method: {func.__name__}{func.__doc__}{ENDC}")
                    return None
            return func(parent, *args, **kwargs)
        return __fetch_help_wrapper

    # User accessible functions
    def _show_functions(self):
        """
        Prints all available commands.

        :return: None
        """
        print([key for key in self.command_index])

    @_help
    def print_traceback(self) -> None:
        """
        Prints the most recent traceback.

        :return: None
        """
        print(f"\n{CYAN}XSuperlet Traceback:\n{"\n".join(self.last_traceback.split("\n")[1:])}{ENDC}\n")

    @_help
    def run_command_file(self, file_name: str) -> None:
        """
        Run a command file.

        :param file_name: Name of command file
        :return: None
        """
        with open(file_name, "r") as f:
            print(f"Loaded command file: {file_name}\n{PINK}----------{ENDC}")
            for line in f.readlines():
                print(f"{PINK}>>> {line}{ENDC}")
                # For each line split off the command and params
                comm, params = line.split()[0], line.split()[1:]
                # Execute first matching command
                self.command_index[self.validate_command(comm)[0]](*params)
            print(f"{PINK}----------{ENDC}\n")

    @_help
    def set_freq_range(self, min_f = None, max_f = None, grid_size = None) -> None:
        """
        Sets the frequency range for all future calculations.

        :param min_f: Minimum frequency
        :param max_f: Maximum frequency
        :param grid_size: Size of frequency grid in steps
        :return: None
        """
        # Validate new frequency grid
        min_f = self.__validate_default_type(float, min_f, "minimum frequency", DEFAULT_MIN_FREQ)
        max_f = self.__validate_default_type(float, max_f, "maximum frequency", DEFAULT_MAX_FREQ)
        grid_size = self.__validate_default_type(int, grid_size, "grid size", DEFAULT_FREQ_SIZE)

        # Make new frequency grid
        if FREQ_BIN_SCALE == "log":
            self.__frequency_grid = np.logspace(np.log10(min_f * self.__units), np.log10(max_f * self.__units), grid_size)
        else:
            self.__frequency_grid = np.linspace(min_f * self.__units, max_f * self.__units, grid_size)

        # Update light curves
        for lc in self.__user_lightcurves:
            lc.frequencies = self.__frequency_grid
            lc.update_lc_attributes()

        print(f"{WARN}Frequency grid changed! Transforms must be recalculated to use new grid!{ENDC}\n")

    @_help
    def add_lightcurve(self, length_file: str | int = 200, samples_rbin = None) -> None:
        """
        Adds a new light curve.

        :param length_file: Length of light curve in ks, or file name
        :param samples_rbin: Number of samples, or size of time bins in s (optional)
        :return: None
        """
        # Validates samples/bin size
        samples_rbin = self.__validate_default_type(int, samples_rbin, "samples/bin size", DEFAULT_BIN_SIZE)

        # Check for either an integer of file name
        try:
            int(length_file)
        except ValueError:
            # Load light curve from file
            lightcurve = LightCurve(file=length_file, rebin=samples_rbin, zerot=ZERO_TIME_SERIES)

            if lightcurve.load_state:
                # Add WaveTransform object to list
                self.__user_lightcurves.append(WaveTransform(self.__user_lightcurve_count, lightcurve, self.__frequency_grid, length_file, self.__units))
        else:
            # Add empty light curve
            lightcurve = LightCurve(int(length_file), int(samples_rbin))
            self.__user_lightcurves.append(WaveTransform(self.__user_lightcurve_count, lightcurve, self.__frequency_grid, "SIMULATED", self.__units))
        finally:
            self.__user_lightcurve_count += 1

    @_help
    @_validate_lc
    def remove_lightcurve(self, lc_code: int) -> None:
        """
        Removes a light curve.

        :param lc_code: Identifier of light curve to remove
        :return: None
        """
        self.__user_lightcurves.pop(self.__get_lc(lc_code))

    @_help
    @_validate_lc
    def export_lightcurve(self, lc_code: int, file_name: str) -> None:
        """
        Exports a light curve to file.

        :param lc_code: Identifier of light curve to export
        :param file_name: Name of export file (Should be .fits or .lc)
        :return: None
        """
        # Get WT object
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)

        # Save the light curve to disk
        wt.lc.save(file_name)

    @_help
    def show(self, item: str) -> None:
        """
        Prints information about the current instances of a given object type.

        :param item: Object type, either light curve, cwt, or slt
        :return: None
        """
        # Light curves
        if item.lower() in LC_PROXIES:
            print("\nCurrent Light Curves:")
            for lc in self.__user_lightcurves:
                print(lc)

        # CWTs
        elif item.lower() in CWT_PROXIES:
            print("\nCurrent Wavelet Transforms:")
            for lc in self.__user_lightcurves:
                print(lc.print_wavelet())

        # SLTs
        elif item.lower() in SLT_PROXIES:
            print("\nCurrent Superlet Transforms:")
            for lc in self.__user_lightcurves:
                print(lc.print_superlet())

        # WWZs
        elif item.lower() in WWZ_PROXIES:
            print("\nCurrent WWZ Transforms:")
            for lc in self.__user_lightcurves:
                print(lc.print_wwz())

        else:
            print(f"{ERRR}Invalid value '{item}' for parameter <item>, possible values: 'lc', 'cwt', 'slt', 'wwz'{ENDC}")

    @_help
    @_validate_lc
    def plot_lightcurve(self, lc_code: int) -> None:
        """
        Plots a light curve.

        :param lc_code: Light curve identifier
        :return: None
        """
        # Get WT object
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)

        # Plot light curve
        wt.plot_signal()

    @_help
    @_validate_lc
    def rebin_lightcurve(self, lc_code: int, rbin: int, gaps: int = 0) -> None:
        """
        Rebins a light curve.

        :param lc_code: Light curve identifier
        :param rbin: Bin size in seconds
        :param gaps: If 1, will bin sections individually, else gaps will be ignored
        :return: None
        """
        # Check bin size is a valid int
        if self.__validate_type(int, rbin, "bin_size"):
            # Get WT object
            wt: WaveTransform = self.__get_wave_transform_object(lc_code)

            # Rebin and update light curve
            wt.lc.rebin(int(rbin), gaps)
            wt.update_lc_attributes()

    @_help
    @_validate_lc
    def norm_lightcurve(self, lc_code: int) -> None:
        """
        Normalises a light curve.

        :param lc_code: Light curve identifier
        :return: None
        """
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)
        wt.lc.normalise()
        wt.update_lc_attributes()

    @_help
    @_validate_lc
    def add_component(self, lc_code: int, comp: str, *args) -> None:
        """
        Adds a component to a light curve.

        :param lc_code: Light curve identifier
        :param comp: Component string: sin/poi/red/gap/fill
        :param args: Parameters for component
        :return: None
        """
        # Get WT and LightCurve objects
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)
        lc: LightCurve = wt.lc

        # Add sinusoid
        if comp.lower() == "sin":
            if not (1 <= len(args) <= 4):
                print(f"{ERRR}Invalid parameters for `Add Sinusoid` component\nAdd Sinusoid: <frequency> <amplitude> <start> <stop>{ENDC}")
            else:
                lc.add_sinusoid(*args)
                if lc.filename != "SIMULATED":
                    lc.filename = f"{lc.filename}_MODIFIED"

        # Add Poisson noise
        elif comp.lower() == "poi" or comp.lower() == "poisson":
            if len(args) != 0:
                print(f"{WARN}Extra parameters for `Add Poisson` component ignored{ENDC}")
            lc.add_poisson_noise()
            if lc.filename != "SIMULATED":
                lc.filename = f"{lc.filename}_MODIFIED"

        # Add Red noise
        elif comp.lower() == "red" or comp.lower() == "rnoise":
            if len(args) > 1:
                print(f"{WARN}Extra parameters for `Add Poisson` component ignored{ENDC}")
            elif len(args) == 0:
                print(f"{WARN}No value specified for parameter <level>, defaulting to 1.0{ENDC}")
                args = [1.0]

            if self.__validate_type(float, args[0], "level"):
                lc.add_red_noise(args[0])
                if lc.filename != "SIMULATED":
                    lc.filename = f"{lc.filename}_MODIFIED"

        # Add gaps
        elif comp.lower() == "gap" or comp.lower() == "gaps":
            if len(args) != 2:
                print(f"{ERRR}Invalid parameters for `Add Gaps` component\nAdd Gaps: <data_length> <gap_length>{ENDC}")
            elif self.__validate_type(int, args[0], "data_length") and self.__validate_type(int, args[1], "gap_length"):
                lc.add_gaps(True, int(args[0]), int(args[1]))
                if lc.filename != "SIMULATED":
                    lc.filename = f"{lc.filename}_MODIFIED"

                # Update the light curve since its attributes have changed
                wt.update_lc_attributes()

        # Fill gaps
        elif comp.lower() == "fill" or comp.lower() == "gap-fill":
            if len(args) > 1:
                print(f"{ERRR}Invalid parameters for `Fill Gaps` component\nFill Gaps: <value>{ENDC}")
                return
            elif len(args) == 1:
                if self.__validate_type(float, args[0], "value"):
                    lc.fill_gaps(float(args[0]))
                elif args[0] == "mean":
                    lc.fill_gaps("mean")
                    if lc.filename != "SIMULATED":
                        lc.filename = f"{lc.filename}_MODIFIED"
            else:
                print(f"{WARN}No value specified for parameter <value>, defaulting to mean rate{ENDC}")
                lc.fill_gaps()
                if lc.filename != "SIMULATED":
                    lc.filename = f"{lc.filename}_MODIFIED"

            # Update the light curve since its attributes have changed
            wt.update_lc_attributes()

        # Linear Interpolate
        elif comp.lower() == "lin" or comp.lower() == "linear":
            print(f"Using linear interpolation to fill gaps in light curve {lc_code} ({wt.filename})...")
            lc.linear_interpolate()
            if lc.filename != "SIMULATED":
                lc.filename = f"{lc.filename}_MODIFIED"

            # Update the light curve since its attributes have changed
            wt.update_lc_attributes()

        # User entered non-existent component
        else:
            print(f"{ERRR} {comp} is not a valid component{ENDC}")

    @_help
    @_validate_lc
    def vertical_slice(self, lc_code: int, transform: str, time: float, peak_height: float = None, prominence: float = None) -> None:
        """
        Plots a slice of the specified scalogram at a given time.

        :param lc_code: Light curve identifier
        :param transform: Transform to take slice of
        :param time: Time to take slice at in ks
        :param peak_height: Height of peaks to search for
        :param prominence: Minimum prominence required for a peak to be detected
        :return: None
        """
        # Get WT object and selected transform
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)
        transform = self.__get_transform(transform)

        peak_height = self.__validate_default_type(float, peak_height, "peak_height", PEAK_HEIGHT_DEFAULT)
        prominence = self.__validate_default_type(float, prominence, "min_prominence", MIN_PROMINENCE_DEFAULT)

        if self.__validate_type(float, time, "time"):
            wt.plot_slice(transform, float(time) * 1E-3, peak_height, prominence, COI_FILL)

    @_help
    @_validate_lc
    def trace_frequency_peaks(self, lc_code: int, transform: str, f_target: float, interval: int,
                              peak_height: float = None, prominence: float = None, offset: int = None) -> None:
        """
        Calculates the uncertainty of the frequency peak closest to the target frequency at each interval.
        These frequency uncertainties will be plot on the next scalogram.

        :param lc_code: Light curve identifier
        :param transform: Transform to use
        :param f_target: Approximate frequency of target peak in μHz
        :param interval: Number of ks between each sample
        :param peak_height: Height of peaks to search for
        :param prominence: Minimum prominence required for a peak to be detected
        :param offset: Offset from zero of time interval
        :return:
        """
        # Get WT object and selected transform
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)
        transform = self.__get_transform(transform)

        peak_height = self.__validate_default_type(float, peak_height, "peak_height", PEAK_HEIGHT_DEFAULT)
        prominence = self.__validate_default_type(float, prominence, "min_prominence", MIN_PROMINENCE_DEFAULT)
        offset = self.__validate_default_type(int, offset, "offset", PEAK_INTERVAL_OFFSET)

        if self.__validate_type(float, f_target, "frequency_target") and self.__validate_type(int, interval, "interval"):
            wt.trace_frequency(transform, float(f_target), int(float(interval) * 1E+3), peak_height, prominence, offset)

    @_help
    @_validate_lc
    def remove_frequency_peaks(self, lc_code: int) -> None:
        """
        Deletes all frequency tracers from the light curve.
        They will no longer appear on scalograms.

        :param lc_code: Light curve identifier
        :return: None
        """
        self.__get_wave_transform_object(lc_code).clear_frequency_tracers()

    @_help
    @_validate_lc
    def wavelet_transform(self, lc_code: int, cycles: int = None, *args) -> None:
        """
        Calculates the continuous wavelet transform of the specified light curve.

        :param lc_code: Light curve identifier
        :param cycles: Number of wavelet cycles
        :return: None
        """
        # Get WT object and number of CWT cycles
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)
        cycles = self.__validate_default_type(int, cycles, "cycles", CWT_DEFAULT_CYCLES)

        # Do CWT
        print(f"Calculating wavelet transform for light curve {lc_code} ({wt.filename})...")
        wt.calculate_wavelet_transform(int(cycles), *args)

    @_help
    @_validate_lc
    def wwz_transform(self, lc_code: int, f_bins: int, t_bin: int) -> None:
        """
        Calculates the weighted wavelet Z-transform of the specified light curve.

        :param lc_code: Light curve identifier
        :param f_bins: Number of frequency bins
        :param t_bin: Time bin size in time unit (default ks)
        :return: None
        """
        # Get WT object and WWZ parameters
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)
        f_bins = self.__validate_default_type(int, f_bins, "frequency_bins", WWZ_DEFAULT_FREQS)
        t_bin = self.__validate_default_type(int, t_bin, "time_bin_size", WWZ_DEFAULT_TBIN)

        # Do WWZ
        print(f"Calculating weighted wavelet Z-transform for light curve {lc_code} ({wt.filename})...")
        wt.calculate_wwz_transform([f_bins, t_bin])

    @_help
    @_validate_lc
    def superlet_transform(self, lc_code: int, base_cycles: int = None, min_ord: int = None, max_ord: int = None, *args) -> None:
        """
        Calculates the superlet transform of the specified light curve.

        :param lc_code: Light curve identifier
        :param base_cycles: Base number of cycles
        :param min_ord: Minimum order
        :param max_ord: Maximum order
        :param args: Additional options for superlet transform, possible flags:
                    -w Shows the superlets and signal together
                    -s Shows the scalogram
        :return: None
        """
        # Get WT object and SLT parameters
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)
        base_cycles = self.__validate_default_type(int, base_cycles, "base_cycles", SLT_DEFAULT_BASE_CYCLES)
        min_ord = self.__validate_default_type(int, min_ord, "minimum_order", SLT_DEFAULT_MIN_ORDER)
        max_ord = self.__validate_default_type(int, max_ord, "maximum_order", SLT_DEFAULT_MAX_ORDER)

        # Do SLT
        print(f"Calculating superlet transform for light curve {lc_code} ({wt.filename})...")
        wt.calculate_superlet_transform([base_cycles, min_ord, max_ord], *args)

    @_help
    @_validate_lc
    def gaussian_process_slt(self, lc_code: int, *gp_slt_params) -> None:
        """
        Calculates the superlet transform of the specified light curve, using Gaussian processes to model over gaps.

        :param lc_code: Light curve identifier
        :param gp_slt_params: Gaussian process and Superlet transform parameters
        :return: None
        """
        # Get SLT parameters
        gp_slt_params = list(gp_slt_params)
        gp_slt_params = self.__force_list_size(gp_slt_params, 4)

        gp_slt_params[0] = self.__validate_default_type(int, gp_slt_params[0], "base_cycles", SLT_DEFAULT_BASE_CYCLES)
        gp_slt_params[1] = self.__validate_default_type(int, gp_slt_params[1], "minimum_order", SLT_DEFAULT_MIN_ORDER)
        gp_slt_params[2] = self.__validate_default_type(int, gp_slt_params[2], "maximum_order", SLT_DEFAULT_MAX_ORDER)

        while True:
            samples = input("Number of Gaussian process samples: ")
            if self.__validate_type(int, samples, "samples"):
                break

        # Get the WaveTransform object and update SLT parameters
        transform_obj: WaveTransform = self.__get_wave_transform_object(lc_code)
        transform_obj.slt_params = gp_slt_params

        # Get the LightCurve object and rebin
        lcurve: LightCurve = transform_obj.lc

        # Fit light curve with Gaussian process
        start = t.time()
        print(f"Fitting light curve {lc_code} ({transform_obj.filename}) with Gaussian Processes... ", end="", flush=True)
        gpmodel = GPLightCurve(lc=lcurve,
                               kernel=Constant(1.0, (1e-3, 1e3))
                                      * RationalQuadratic(1000, 0.01, (1e-10, 1e10), (1e-10, 1e10)))
        print(f"({round(t.time() - start, 2)}s)")

        # Take GP samples
        start = t.time()
        print(f"Taking {samples} samples... ", end="", flush=True)
        gpsamples = [gpmodel.sample() for _ in range(int(samples))]
        pred = gpmodel.predict()
        print(f"({round(t.time() - start, 2)}s)")

        # View GP result
        if DEBUG_GP:
            plt.figure(figsize=(16, 5), label=f"Gaussian Process Debug Plot: {transform_obj.filename}")
            for sample in gpsamples:
                plt.plot(sample.time, sample.rate, color="dodgerblue", label="GP Sample", linewidth=0, marker="+")
            plt.plot(lcurve.time, lcurve.rate, color="grey", label="Light Curve")
            plt.plot(pred.time, pred.rate, color="purple", label="GP Model")
            plt.fill_between(pred.time, pred.rate - pred.error, pred.rate + pred.error, color="purple", alpha=0.2, label="Error region")
            print("Plotting GP")
            plt.show(block=True)
            input("Press Return to continue ")

        # Calculate SLT for each sample
        start = t.time()
        print(f"Calculating superlet transforms for Gaussian process samples...", end="\r", flush=True)
        gp_slts = []
        for i, gps in enumerate(gpsamples):
            # Append the SLT of each sample to a list
            slt = SuperletTransform(gps.time, gps.rate, transform_obj.sample_rate, transform_obj.frequencies, c_1=transform_obj.slt_params[0],
                                    orders=[transform_obj.slt_params[1],transform_obj.slt_params[2]]).calculate_transform()
            gp_slts.append(slt)
            print(f"Calculating superlet transforms for Gaussian process samples {i+1}/{samples}", end="\r", flush=True)

        # Calculate the mean SLT
        mean_slt = gmean(np.stack(gp_slts), axis=0)
        transform_obj.slt = mean_slt
        transform_obj.slt_gp_samples = samples
        print(f"Calculating superlet transforms for Gaussian process samples {samples}/{samples}\n"
              f"{GREN}GP/SLT Successful ({round(t.time() - start, 2)}s){ENDC}")

    @_help
    @_validate_lc
    def simulate_lightcurves(self, lc_code: int, sim_curves: int = None, name: str = None) -> None:
        """
        Simulates a number of light curves based on the PSD of a given light curve.

        :param lc_code: Light curve identifier
        :param sim_curves: Number of light curves to simulate
        :param name: File name of the simulated light curves
        :return: None
        """
        # Get WT object and validate parameters
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)
        sim_curves = self.__validate_default_type(int, sim_curves, "number_to_simulate", 10)
        name = self.__validate_default_type(str, name, "name", "SIM")

        # Simulate light curves
        wt.simulate_light_curves(sim_curves, name, SIM_TYPE, PROCESS_COUNT)

    @_help
    @_validate_lc
    def pdf_fit(self, lc_code: int, kde_bandwidth: float = None) -> None:
        """
        Plots the fitted PDF of the given light curve

        :param lc_code:
        :param kde_bandwidth:
        :return:
        """
        # Get WT object and validate parameters
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)
        if kde_bandwidth is None:
            print(f"{WARN}No value specified for parameter <kde_bandwidth>, value not changed (Current value: {wt.sampler.kde_width}){ENDC}")
        else:
            try:
                kde_bandwidth = float(kde_bandwidth)
            except TypeError:
                print(f"{ERRR}Invalid value '{kde_bandwidth}' for parameter <kde_bandwidth> - value not changed (Current value: {wt.sampler.kde_width}){ENDC}")

        if kde_bandwidth is not None:
            wt.sampler.kde_width = kde_bandwidth
        wt.sampler.fit_pdf(debug_plot=True)

    @_help
    @_validate_lc
    def import_sim_lightcurves(self, lc_code: int, directory: str) -> None:
        """
        Import previously simulated light curves for a given light curve.
        Note that it is possible to import light curves simulated from a different light curve than specified by lc_code,
        this will lead to incorrect significance estimation.

        :param lc_code: Light curve identifier
        :param directory: Directory containing simulated light curves
        :return: None
        """
        # Get WT object
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)

        # Import light curves
        wt.sim_curves = SimLightCurves("IMPORTED").read_lcs(directory)

    @_help
    @_validate_lc
    def calculate_significance(self, lc_code: int, transform: str = None, level: float = None, file_name: str = None) -> None:
        """
        Calculates the significance for the given light curve and transform, and adds it to the list of significance contours.

        :param lc_code: Light curve identifier
        :param transform: Transform to calculate significance for
        :param level: Significance level
        :param file_name: Name of output file containing the significance map at the given level for this transform.
        :return: None
        """
        # Validate transform
        transform = self.__get_transform(transform)
        if transform is None:
            return

        # Validate significance parameter
        level = self.__validate_default_type(float, level, "significance_level", 90)

        # Get WT object
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)

        # Check simulated light curves exist
        if wt.sim_curves is None:
            print(f"{ERRR}ID '{lc_code}' has no simulated light curves attributed to it{ENDC}")
            return

        # Estimate the significance
        wt.estimate_significance(transform, level, file_name, PROCESS_COUNT)

    @_help
    @_validate_lc
    def set_significance(self, lc_code: int, transform: str = None, significance: float = None, file_name: str = None) -> None:
        """
        Sets the significance level of the given transform and adds it to the list of significance contours.
        You must calculate the significance with the `significance` function first.

        :param lc_code: Light curve identifier
        :param transform: Transform to set significance for
        :param significance: Significance level as an Nth percentile
        :param file_name: Name of output file containing the significance map at the given level for this transform.
        :return: None
        """
        # Validate the transform
        transform = self.__get_transform(transform)
        if transform is None:
            return

        # Get WT object and significance
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)
        significance = self.__validate_default_type(float, significance, "significance_level", 90)

        # Set the significance
        wt.set_significance(transform, significance, file_name)
        print(f"Light curve ID {wt.code} significance contours for {transform} set to {significance}% level")

    @_help
    @_validate_lc
    def load_significance(self, lc_code: int, transform: str = None, file_name: str = None) -> None:
        """
        Loads a significance map for the given transform and adds it to the list of significance contours.
        Note that it is possible to load the significance of a different light curve/transform than specified,
        this will lead to incorrect significance estimation.

        :param lc_code: Light curve identifier
        :param transform: Transform to set significance for
        :param file_name: Name of input file containing the significance map at the given level for this transform.
        :return: None
        """
        # Validate the transform
        transform = self.__get_transform(transform)
        if transform is None:
            return

        # Get WT object
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)

        # Load the significance
        wt.load_significance(transform, file_name)
        print(f"Light curve ID {wt.code} significance contours for {transform} set from file {file_name}")

    @_help
    @_validate_lc
    def clear_significance(self, lc_code: int, transform: str = None) -> None:
        """
        Clears all current significance contours for a given light curve and transform.

        :param lc_code: Light curve identifier
        :param transform: Transform to clear significance for
        :return: None
        """
        # Validate the transform
        transform = self.__get_transform(transform)
        if transform is None:
            return

        # Get WT object
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)

        # Clear the significance
        wt.clr_significance(transform)
        print(f"Light curve ID {wt.code} significance contours for {transform} cleared")

    @_help
    @_validate_lc
    def calculate_coi(self, lc_code: int, *args) -> None:
        """
        Calculates the COI for the given light curve.
        Remove the COI by passing `-r` as a parameter.

        :param lc_code: Light curve identifier
        :return: None
        """
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)
        wt.calculate_coi(args)

    @_help
    @_validate_lc
    def find_gaps(self, lc_code: int, *args) -> None:
        """
        Finds the gaps in the specified light curve.
        Remove the gap highlights by passing `-r` as a parameter.

        :param lc_code: Light curve identifier
        :return: None
        """
        # Get WT object
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)

        # Find the gaps
        wt.find_gaps(args)

    @_help
    @_validate_lc
    def plot_scalogram(self, lc_code: int, transform: str = None, v_max = None, v_min = None, mask_max = None, mask_min = None) -> None:
        """
        Plots a scalogram for the specified transform and light curve.
        If COI/gaps has been called, these regions will be shaded in.
        Both `v_min` and `v_max` must be given to set the colour scale.

        :param lc_code: Light curve identifier
        :param transform: Indicates which transform to plot
        :param v_max: Maximum colour value
        :param v_min: Minimum colour value
        :param mask_max: Mask upper frequency
        :param mask_min: Mask lower frequency
        :return: None
        """
        # Setup mask region(s)
        if mask_min == "0" and mask_max == "0":
            mask = None
        elif self.__validate_type(float, mask_min, "mask_min") and self.__validate_type(float, mask_max, "mask_max"):
            mask = [float(mask_min) * self.__units, float(mask_max) * self.__units]
            print(f"Masking from {mask_min}Hz to {mask_max}Hz")
        else:
            mask = None

        # Validate transform
        try:
            transform.lower()
        except AttributeError:
            print(f"{ERRR}'{transform}' is not a valid transform{ENDC}")
            return

        # Set min/max colour scaling
        if v_min is not None and v_max is not None:
            v_min = self.__validate_default_type(float, v_min, "v_min", None)
            v_max = self.__validate_default_type(float, v_max, "v_max", None)

        # Get WT object
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)

        # Plot CWT
        if transform.lower() in CWT_PROXIES:
            wt.plot_transform("CWT", f"CWT - c={wt.cwt_cycles}", mask, v_min, v_max, gap_fill=GAP_FILL, coi_fill=COI_FILL)
        # Plot WWZ
        elif transform.lower() in WWZ_PROXIES:
            wt.plot_transform("WWZ", f"WWZ - fbin={wt.wwz_params[0]}, tbin={wt.wwz_params[1]}", mask, v_min, v_max, gap_fill=GAP_FILL, coi_fill=COI_FILL)
        # Plot WWA
        elif transform.lower() in WWA_PROXIES:
            wt.plot_transform("WWA", f"WWA - fbin={wt.wwz_params[0]}, tbin={wt.wwz_params[1]}", mask, v_min, v_max, gap_fill=GAP_FILL, coi_fill=COI_FILL)
        # Plot SLT
        elif transform.lower() in SLT_PROXIES:
            wt.plot_transform("SLT", f"SLT - c1={wt.slt_params[0]}, o:{wt.slt_params[1]}-{wt.slt_params[2]}", mask, v_min, v_max, gap_fill=GAP_FILL, coi_fill=COI_FILL)
        else:
            print(f"{ERRR}Invalid value '{transform}' for parameter <transform>, possible values: 'cwt', 'wwz', 'wwa', 'slt'{ENDC}")

    @_help
    @_validate_lc
    def export_transform(self, lc_code: int, transform: str = None) -> None:
        """
        Exports transform data to file.

        :param lc_code: Light curve identifier
        :param transform: Transform to export
        :return: None
        """
        # Get WT object
        wt: WaveTransform = self.__get_wave_transform_object(lc_code)

        # Get transform data
        if transform.lower() in CWT_PROXIES:
            transform = "CWT"
        elif transform.lower() in WWZ_PROXIES:
            transform = "WWZ"
        elif transform.lower() in WWA_PROXIES:
            transform = "WWA"
        elif transform.lower() in SLT_PROXIES:
            transform = "SLT"
        else:
            print(f"{ERRR}Invalid value '{transform}' for parameter <transform>, possible values: 'cwt', 'wwz', 'wwa', 'slt'{ENDC}")
            return
        data = wt.get_transform_data(Literal["CWT", "WWZ", "WWA", "SLT"](transform))

        file_name = wt.filename.split(".")[0]

        # Save data
        if data[0] is not None:
            np.save(f"{transform}_{file_name}", data[0])
        if data[1] is not None:
            np.save(f"{transform}_{file_name}_sig", data[1])

        # Save times, frequencies, COI
        np.save(f"{transform}_{file_name}_times", wt.times)
        np.save(f"{transform}_{file_name}_freqs", wt.frequencies)
        if wt._coi is not None:
            np.save(f"{transform}_{file_name}_coi", wt._coi)
            np.save(f"{transform}_{file_name}_coi_f", wt._coi_freq)

    @_help
    @_validate_lc
    def difference_scalogram(self, lc_code: int, lc2_code: int, transform: str = None) -> None:
        """
        Plot the difference scalogram of two transforms.

        :param lc_code:
        :param lc2_code:
        :param transform:
        :return:
        """
        # Validate second light curve
        if self.__validate_type(int, lc2_code, "ID"):
            index = self.__get_lc(lc2_code)
            # Check ID exists
            if index is False:
                print(f"{ERRR}ID '{lc_code}' does not exist{ENDC}")
                return
        else:
            print(f"{ERRR} '{lc_code}' is not a valid light curve ID{ENDC}")
            return

        lc1: WaveTransform = self.__get_wave_transform_object(lc_code)
        lc2: WaveTransform = self.__get_wave_transform_object(lc2_code)

        # Validate transform
        try:
            transform.lower()
        except AttributeError:
            print(f"{ERRR}'{transform}' is not a valid transform{ENDC}")
            return

        if transform.lower() in CWT_PROXIES:
            t1 = lc1.get_transform_data("CWT")[0]
            t2 = lc2.get_transform_data("CWT")[0]
        elif transform.lower() in WWZ_PROXIES:
            t1 = lc1.get_transform_data("WWZ")[0]
            t2 = lc2.get_transform_data("WWZ")[0]
        elif transform.lower() in WWA_PROXIES:
            t1 = lc1.get_transform_data("WWA")[0]
            t2 = lc2.get_transform_data("WWA")[0]
        elif transform.lower() in SLT_PROXIES:
            t1 = lc1.get_transform_data("SLT")[0]
            t2 = lc2.get_transform_data("SLT")[0]
        else:
            print(f"{ERRR}Invalid value '{transform}' for parameter <transform>, possible values: 'cwt', 'wwz', 'wwa', 'slt'{ENDC}")
            return

        # Normalise transforms
        t1_norm = (t1 - t1.min()) / (t1.max() - t1.min())
        t2_norm = (t2 - t2.min()) / (t2.max() - t2.min())

        lc1.plot_transform(t1_norm - t2_norm, f"Difference SLT - c1={lc1.slt_params[0]}, o:{lc1.slt_params[1]}-{lc1.slt_params[2]}",
                           gap_fill=GAP_FILL, coi_fill=COI_FILL)

    # Command validation and line completion functions (Not user functions)
    def validate_command(self, com_str: str) -> list:
        """
        Validates command input and returns list of matching commands.

        :param com_str: User input string
        :return: List of matching commands
        """
        matches = [c for c in self.command_index if c.startswith(com_str)]
        return matches

    def line_completion(self, text, state) -> str | list | None:
        """
        Will automatically complete user input.

        :param text: Current word being input
        :param state:
        :return:
        """
        # Split the current line to check for command or file name completion
        parts = readline.get_line_buffer().split()

        # Complete command
        if len(parts) == 1:
            matching_cmds = self.validate_command(text)
            if state < len(matching_cmds):
                return str(matching_cmds[state]) + " "
            return None

        # Complete file name
        else:
            file_matches = [f for f in os.listdir(".") if f.startswith(text)]
            if state < len(file_matches):
                return file_matches[state] + " "
            return None


def spawn_interactive_shell() -> tuple[int, int]:
    """
    Creates an interactive shell with pty.

    :return: pid and fd of shell
    """
    shell = os.environ.get("SHELL", "/bin/sh")

    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(shell, [shell, '-i'])
    else:
        return pid, fd


# Main loop
if __name__ == "__main__":
    # Create shell if possible
    if UNIX:
        shell_pid, shell_fd = spawn_interactive_shell()
    else:
        shell_pid, shell_fd = None, None

    # Print program header
    print(f"({round(t.time() - loadstart, 2)}s)")
    print(f"\n{BLUE}"
          "========================\n"
          "XSuperlet         v1.0.2\n"
          "========================\n"
          f"{ENDC}")

    # Initialise instance
    root = Xsuperlet()

    # Set up tab completion
    readline.set_completer(root.line_completion)
    readline.parse_and_bind("tab: complete")

    # Setup initial light curve
    if filename is not None:
        print(f"{PINK}>>> add-lc {filename} {binsize}{ENDC}")
        root.add_lightcurve(filename, binsize)

    # Run initial commands
    if command_file is not None:
        root.run_command_file(command_file)

    # Issue hint to start
    if filename is None and command_file is None:
        print("Type a command to start\n")

    # Run the program: Get user input and run commands
    loc = ""
    while True:
        if UNIX:
            loc = os.getcwd().split("/")[-1]
        raw_input = input(f"{PREFIX}{loc}{SUFFIX}").strip()

        # Exit
        if raw_input.lower() == 'exit' or raw_input.lower() == 'quit':
            if input(f"{WARN}Are you sure you want to exit? [y/n]{ENDC}\n{PREFIX}{loc}{SUFFIX}").lower() == "y":
                print(f"{BLUE}Goodbye, have a nice day...{ENDC}\n")
                break

        # No input
        elif raw_input.lower() == "":
            pass

        # Process command
        else:
            # Split input into command and parameters
            user_input = raw_input.split()
            com = user_input[0]
            param = user_input[1:]

            # Find matching commands
            matching_commands = root.validate_command(com)

            if len(matching_commands) == 1:
                # Log command
                if LOGGING:
                    log_dir = pathlib.Path(__file__).parent
                    try:
                        with open(f"{log_dir}/logs/XSuperlet_log_{dt.now().strftime("%Y-%m-%d")}.log", "a") as log:
                            log.write(" ".join(user_input) + "\n")
                    except FileNotFoundError:
                        os.mkdir(f"{log_dir}/logs")
                        with open(f"{log_dir}/logs/XSuperlet_log_{dt.now().strftime("%Y-%m-%d")}.log", "a") as log:
                            log.write(" ".join(user_input) + "\n")

                # Unique match -> call function/method
                command = matching_commands[0]
                try:
                    root.command_index[command](*param)
                except TypeError as e:
                    print(f"{ERRR}XSuperlet runtime exception:\n{e}{ENDC}")
                    root.last_traceback = traceback.format_exc()
                except Exception as e:
                    print(f"{ERRR}XSuperlet runtime exception:\n{e}")
                    user_input = input(f"\n{WARN}We apologise for this issue. Do you wish to attempt to proceed regardless? (y/n)\n-> ")[0].lower()
                    if user_input != "y":
                        raise e
                    else:
                        print(f"{ENDC}")
                        root.last_traceback = traceback.format_exc()

            # Multiple command matches
            elif len(matching_commands) > 1:
                # Not unique -> prompt for clarification
                print(f"?\nTry: {', '.join(matching_commands)}")

            # Command does not exist, send it to the shell if possible
            # TODO: Tidy this up
            elif UNIX:
                # cd command: Move directories
                if com == "cd":
                    target = param[0] if param else os.path.expanduser("~")
                    try:
                        os.chdir(os.path.expanduser(target))
                    except FileNotFoundError as e:
                        print(f"{ERRR}Cannot cd to {param[0]}: No such directory{ENDC}")
                        continue

                # Pass command to shell
                os.write(shell_fd, (raw_input + "\n").encode())

                # Print output minus command (j=0 to j=1)
                j = 0
                while True:
                    rlist, _, _ = select.select([shell_fd, sys.stdin], [], [], 0.1)

                    if shell_fd in rlist and j > 1:
                        # Read and output non-whitespace chunks
                        chunk = os.read(shell_fd, 16384)
                        if not chunk.strip() == b"":
                            os.write(sys.stdout.fileno(), chunk)

                    elif shell_fd not in rlist:
                        # If nothing else in fd, finish and return to XSuperlet
                        break

                    else:
                        # Read chunk but do nothing
                        chunk = os.read(shell_fd, 16384)
                    j += 1

                # Flush final shell output
                sys.stdout.flush()
                os.write(sys.stdout.fileno(), b"\r\033[K")
