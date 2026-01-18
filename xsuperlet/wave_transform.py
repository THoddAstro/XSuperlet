"""
Wavelet/Superlet transform calculation/plotting for AGN timing analysis.

SuperletTransform.py based on 'Time-frequency super-resolution with superlets' by Moca et al., 2021 Nature Communications.

Uses wwz.py by Kiehlmann, S., Max-Moerbeck, W., & King, O. (2021), 'wwz: Weighted wavelet z-transform code'.

Uses Light curve simulation based on Timmer & Koenig (1995) and Emmanoulopoulos et al. (2013).

Author: Thomas Hodd

Date - 18th January 2026

Version - 1.2
"""
# System packages
import os
import sys
import time as t
from datetime import datetime as dt
import secrets

# Science packages
import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from scipy.signal import peak_widths, find_peaks

# Local packages
from .xlight_curves import LightCurve
from .superlet_transform import SuperletTransform
from .sim_light_curve import LightCurveSampler

# Typing and text format
from typing import Literal
from .text_format import *
import contextlib

# pyLag & WWZ
try:
    import pylag as pylg
except ModuleNotFoundError:
    print(f"{ERRR}pyLag not found!\nInstall it with `pip install pylag` to continue!")
    exit()

try:
    from wwz import WWZ
except ModuleNotFoundError:
    print(f"{ERRR}\n\nwwz.py not found!\n WWZ transform will not be supported!{ENDC}\n")

# Multiprocessing packages
import copy as cp
import concurrent.futures as cf


def tick_format(x: int | float, _) -> str:
    """
    Determines the format of axes numbers.

    :param x: Value
    :param _: Matplotlib parameter
    :return: String of axis tick
    """
    if x == 0:
        return "0"
    elif x < 1:
        return f"{x:.2f}"
    else:
        return f"{int(x)}"


def closest_index(array: list | np.ndarray, value: int | float) -> int:
    """
    Returns the index of the value in a list closest to the given value.
    This is a horrible one line function based on:
    https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/

    :param array: List to search through
    :param value: Value to search for
    :return: Index of nearest value in list
    """
    return array.tolist().index(array[min(range(len(array)), key=lambda i: abs(array[i] - value))])


def remap(array1: list | np.ndarray, array2: list | np.ndarray) -> np.ndarray:
    """
    Remaps one array onto the space of another.

    :param array1: Array to remap
    :param array2: Array to remap array1 into
    :return: Remapped array1
    """
    t_grid_original = np.linspace(np.min(array1), np.max(array1), len(array1))
    t_grid_new = np.linspace(np.min(array2), np.max(array2), len(array2))
    return np.interp(t_grid_new, t_grid_original, array1)


def renormalise_lc(parent: LightCurve, child: LightCurve) -> LightCurve:
    """
    Renormalises child light curve to its parent.

    :param parent: Parent light curve
    :param child: Child light curve
    :return: Renormalised child light curve
    """
    std = np.std(parent.rate)
    mean = np.mean(parent.rate)
    c_mean = np.mean(child.rate)
    c_std = np.std(child.rate)

    child.rate = [(r - c_mean) / c_std for r in child.rate]
    child.rate = [(r * std) + mean for r in child.rate]
    return child


def divide_list(lst: list, n: int) -> list:
    """
    Divides a list into n roughly equal parts.

    :param lst: List to divide
    :param n: Number of parts to divide into
    :return: List containing n parts of original list
    """
    chunk_size = len(lst) // n
    remainder = len(lst) % n
    split_lists = []
    start = 0
    for i in range(n):
        end = start + chunk_size + (1 if i < remainder else 0)
        split_lists.append(lst[start:end])
        start = end
    return split_lists


def move_up(n: int):
    """
    Move up in stdout.
    """
    if n > 0:
        sys.stdout.write(f"\x1b[{n}A")

def move_down(n: int):
    """
    Move down in stdout.
    """
    if n > 0:
        sys.stdout.write(f"\x1b[{n}B")
def clear_line():
    """
    Clear current line in stdout.
    """
    sys.stdout.write(f"\x1b[2K")


class WaveTransform:
    """
    Class for calculating wavelet and superlet transforms. Transforms and their parameters are stored in each object.
    This class includes methods for plotting the CWT and SLT as well as the input time series.
    A WaveTransform object is created from a signal and array of frequencies. WaveTransform objects can be created from the terminal by instances of Xsuperlet.
    WaveTransform objects may also be created manually in Python code.

    Parameters
    ==========
    code: str | int
        Name of the signal.
    lightcurve: LightCurve
        Input light curve
    frequencies: np.ndarray
        Array of frequencies of interest
    filename: str
        Name of light curve FITS file
    unit: float
        Units of time - 1E+0 = s
    sim_pdf_use_kde: bool
        If True, light curve simulation will use kernel density estimation to fit the light curve PDF
        If False, light curve simulation will use the user defined statistical model (Gamma + Lognorm by default)

    Attributes
    ==========
    code: str | int
        Name of the signal
    lc: LightCurve
        Input light curve, the "real" observational data
    signal: np.ndarray
        Array containing the binned count rate of the light curve
    times: np.ndarray
        Array containing the centres of the light curve time bins
    filename: str
        Name of light curve FITS file, or name of faked light curve
    frequencies: np.ndarray
        Array of frequencies of interest
    timespan: float
        The total length of the light curve in units of time as defined upon initialisation (s by default)
    sample_rate: float
        The sampling rate of the light curve in units of time as defined upon initialisation (uHz by default)
    sampler: LightCurveSampler
        LightCurveSampler object that may be used to generate simulated light curves based on the "real" light curve
    sim_curves: list
        List of simulated light curves
    sim_transforms: list
        List of the transforms calculated from the simulated light curves
    X: np.ndarray
        Array containing any calculated transforms - X may be any of "cwt", "slt", "wwz", "wwa"
    X_significance: np.ndarray
        Array containing the significance of any calculated transforms - X may be any of "cwt", "slt", "wwz", "wwa"
    """
    def __init__(self, code: str | int, lightcurve: LightCurve, frequencies: np.ndarray,
                 filename: str, unit: float = 1E+0, sim_pdf_use_kde: bool = True) -> None:
        # Unique identifier (set by Xsuperlet), or name (set by user)
        self.code = code

        # Light curve information
        self.lc = lightcurve
        self.signal = self.lc.rate
        self.times = self.lc.time * (1 / unit)
        self.filename = filename

        # Frequencies and units
        # TODO: Tidy up units
        self.frequencies = frequencies
        self.__unit = unit
        self.__f_unit = 1E+6
        self.__f_units = {1E-3: "kHz",
                          1E+0: "Hz",
                          1E+3: "mHz",
                          1E+6: "μHz",
                          1E+9: "nHz"}
        self.__t_unit = 1E-3
        self.__t_units = {1E-9: "Gs",
                          1E-6: "Ms",
                          1E-3: "ks",
                          1E+0: "s",
                          1E+3: "ms"}

        # Cone of Influence
        self._coi = None
        self._coi_freq = None
        self._poisson_lim = None

        # Gaps
        self.__gaps = None
        self.__gap_frequency = None

        # Light curve simulation and significance estimation
        self.sampler_use_kde = sim_pdf_use_kde
        self.sampler = LightCurveSampler(self.get_pylag_lc(), kde=self.sampler_use_kde)
        self.sim_curves = None
        self.sim_transforms = None

        # Period diagnostic lines
        self.__period_lines = None

        # Frequency Peaks
        self.__f_peaks = []
        self.__f_peaks_time = []
        self.__f_peaks_error = []

        # CWT
        self.cwt_cycles = 16
        self.cwt = None
        self.cwt_significance = []

        # WWZ
        self.wwz_params = [1000, 1000]
        self.wwz = None
        self.wwz_significance = []
        self.wwa = None
        self.wwa_significance = []
        self.wwz_freq_grid = None
        self.wwz_time_grid = None

        # SLT
        self.slt_params = [3, 1, 30]
        self.slt_gp_samples = "No"
        self.slt = None
        self.slt_significance = []

        # Calculate timespan and sample rate
        self.timespan = int((self.times[-1] - self.times[0]) * self.__unit)
        self.sample_rate = int((1 / int((self.times[1] - self.times[0]) * self.__unit)) * self.__unit)

        # Print summary
        print(f"\n{GREN}Successfully loaded: {self.filename}\n"
              f"ID {self.code}, {self.timespan}s @ {self.sample_rate}μHz, {len(self.times)} samples\n{ENDC}")

        # Calculate the scalogram limits
        self.__limits = (float(self.times[0] * self.__unit * self.__t_unit), float(self.times[-1] * self.__unit * self.__t_unit),
                         float(self.frequencies[0] * (1 / self.__unit) * self.__f_unit), float(self.frequencies[-1] * (1 / self.__unit) * self.__f_unit))

    def __str__(self) -> str:
        return f"ID: {self.code:<10} Length: {self.timespan:<10} Samples: {len(self.signal):<10}{"Filename: " + self.filename:>70}"

    def print_wavelet(self) -> str:
        """
        Returns str of wavelet transform.
        :return:
        """
        if self.cwt is not None:
            return f"ID: {self.code:<10} CWT cycles: {self.cwt_cycles:<10}{"Filename: " + self.filename:>86}"
        else:
            return f"ID: {self.code:<10} CWT cycles: {"N/A":<10}{"Filename: " + self.filename:>86}"

    def print_superlet(self) -> str:
        """
        Returns str of wavelet transform.
        :return:
        """
        if self.slt is not None:
            return (f"ID: {self.code:<10} SLT base cycle: {self.slt_params[0]:<10}"
                    f" SLT order: ({self.slt_params[1]:<3}, {self.slt_params[2]:<3})      GP: {self.slt_gp_samples:<5}{"Filename: " + self.filename:>45}")
        else:
            return (f"ID: {self.code:<10} SLT base cycle: {"N/A":<10}"
                    f" SLT order: ({"N/A":<3}, {"N/A":<3})      GP: {"N/A":<5}{"Filename: " + self.filename:>45}")

    def print_wwz(self) -> str:
        """
        Returns str of weighted wavelet z-transform.
        :return:
        """
        if self.wwz is not None:
            return (f"ID: {self.code:<10} Frequency bins: {self.wwz_params[0]:<6}"
                    f" Time bin size: {self.wwz_params[1]:<6}{"Filename: " + self.filename:>64}")
        else:
            return (f"ID: {self.code:<10} Frequency bins: {"N/A":<6}"
                    f" Time bin size: {"N/A":<6}{"Filename: " + self.filename:>64}")

    def update_lc_attributes(self) -> None:
        """
        Updates attributes related to the light curve.
        Should be called whenever the light curve is modified (This is done automatically by Xsuperlet)

        :return: None
        """
        # Update rate and time arrays
        self.signal = self.lc.rate
        self.times = self.lc.time * (1 / self.__unit)

        # Update timespan and sample rate
        self.timespan = int((self.times[-1] - self.times[0]) * self.__unit)
        self.sample_rate = int((1 / int((self.times[1] - self.times[0]) * self.__unit)) * self.__unit)

        # Calculate the scalogram limits
        self.__limits = (float(self.times[0] * self.__unit * self.__t_unit), float(self.times[-1] * self.__unit * self.__t_unit),
                         float(self.frequencies[0] * (1 / self.__unit) * self.__f_unit), float(self.frequencies[-1] * (1 / self.__unit) * self.__f_unit))

        # Recreate the sampler
        self.sampler = LightCurveSampler(self.get_pylag_lc(), kde=self.sampler_use_kde)

        print(f"{GREN}Light curve ID {self.code} updated\n"
              f"{self.timespan}s @ {self.sample_rate}μHz, {len(self.times)} samples\n{ENDC}")

    def get_transform_data(self, transform: Literal["CWT", "WWZ", "WWA", "SLT"] | np.ndarray) -> tuple:
        """
        Fetches the data for the given transform, if it exists.

        :param transform: 3-letter code of the transform: "CWT", "WWZ", "WWA" or "SLT"
        :return: Transform data
        """
        if transform == "CWT":
            transform_data = self.cwt
            transform_sig = self.cwt_significance
        elif transform == "WWZ":
            transform_data = self.wwz
            transform_sig = self.wwz_significance
        elif transform == "WWA":
            transform_data = self.wwa
            transform_sig = self.wwa_significance
        else:
            transform_data = self.slt
            transform_sig = self.slt_significance

        # If transform doesn't exist, return nothing
        if transform_data is None:
            print(f"{ERRR}{transform} transform has not been calculated{ENDC}")
            return None, None

        # Get the absolute value of the CWT/SLT
        if transform == "CWT" or transform == "SLT":
            transform_data = np.abs(transform_data)

        return transform_data, transform_sig

    def get_pylag_lc(self):
        """
        Returns the light curve as a pyLag LightCurve object.

        :return: LightCurve object
        """
        return pylg.LightCurve(t=self.lc.time, r=self.lc.rate, e=self.lc.error)

    def calculate_wavelet_transform(self, cycles: int = None, *args) -> None:
        """
        (Re)calculates the continuous wavelet transform of the signal.
        If `cycles` is given, these will overwrite the previous number of cycles.

        :param cycles: Change the number of cycles in the wavelet
        :return: None
        """
        start = t.time()

        # Determine if a new number of cycles has been given
        if cycles is not None:
            self.cwt_cycles = cycles

        # Detect flags
        show_superlet, show_scalogram = False, False
        if "-w" in args:
            show_superlet = True
        if "-s" in args:
            show_scalogram = True

        # Calculate the continuous wavelet transform
        cwt_object = SuperletTransform(self.times, self.signal, self.sample_rate, self.frequencies, self.cwt_cycles, [1])
        cwt_object.calculate_transform(show_scalogram, show_superlet, *args)
        self.cwt = cwt_object.transform

        end = t.time()
        print(f"{GREN}CWT calculated in {end - start:.2f} seconds{ENDC}")

    def calculate_wwz_transform(self, wwz_params: list[int | float] = None) -> None:
        """
        (Re)calculates the weighted wavelet Z-transform of the signal.
        :return:
        """
        start = t.time()

        # Determine if new parameters have been given
        if wwz_params is not None:
            self.wwz_params = wwz_params

        # Initialise WWZ object
        wwz = WWZ(self.times * self.__unit, self.signal)

        # Create time and frequency grids
        self.wwz_freq_grid = wwz.get_freq(self.__unit / self.frequencies.max(), self.__unit / self.frequencies.min(), wwz_params[0])
        self.wwz_time_grid = wwz.get_tau(self.times.min(), self.times.max(), wwz_params[1])

        wwz.set_freq(self.wwz_freq_grid)
        wwz.set_tau(self.wwz_time_grid)

        # Calculate the WWZ transform
        wwz.transform(verbose=2)

        # Save WWZ and WWA
        self.wwz = wwz.wwz.transpose()
        self.wwa = wwz.wwa.transpose()

        end = t.time()
        print(f"{GREN}WWZ calculated in {end - start:.2f} seconds{ENDC}")

    def calculate_superlet_transform(self, slt_params: list[int | str] = None, *args) -> None:
        """
        (Re)calculates the superlet transform of the signal.
        If `slt_params` is given, these will overwrite the previous parameters.

        :param slt_params: Change the superlet transform base cycles, minimum order, maximum order
        :return: None
        """
        start = t.time()

        # Determine if a new number of cycles has been given
        if slt_params is not None:
            self.slt_params = slt_params

        # Detect flags
        show_superlet, show_scalogram = False, False
        if "-w" in args:
            show_superlet = True
        if "-s" in args:
            show_scalogram = True

        # Calculate the superlet transform
        slt_object = SuperletTransform(self.times, self.signal, self.sample_rate, self.frequencies, self.slt_params[0], [self.slt_params[1], self.slt_params[2]])
        slt_object.calculate_transform(show_scalogram, show_superlet, *args)
        self.slt = slt_object.transform

        # Normalise scalogram power to mean count rate
        # self.slt /= np.mean(self.lc.rate)

        self.slt_gp_samples = "No"

        end = t.time()
        print(f"{GREN}SLT calculated in {end - start:.2f} seconds{ENDC}")

    def calculate_coi(self, rm: str | tuple | bool = " ") -> None:
        """
        (Re)calculates the cone of influence (COI).
        Remove the COI by passing `-r` as a parameter.

        :param rm: Destroys the COI so that it will no longer be plot
        :return: None
        """
        # Destroy the COI
        if rm is True or "-r" in rm:
            self._coi = None
            self._coi_freq = None
            print(f"{WARN}COI Destroyed for ID {self.code}{ENDC}")
            return

        # Determine the midpoint
        midpoint = ((self.times[-1]) / 2) * self.__unit

        # Calculate the COI
        c_f = 6 / (2 * pi)
        coi = sqrt(2) * c_f / (self.frequencies * (1 / self.__unit))
        mask = coi <= midpoint

        # Add both halves together
        self._coi = np.concatenate((np.flip(coi[mask]), 2 * midpoint - coi[mask]))
        self._coi_freq = np.concatenate((np.flip(self.frequencies[mask]), self.frequencies[mask]))

        # Add points at the limits
        self._coi = np.append(self._coi, self.times[-1] * self.__unit)
        self._coi_freq = np.append(self._coi_freq, self.frequencies[-1])

        self._coi = np.insert(self._coi, 0, self.times[0] * self.__unit)
        self._coi_freq = np.insert(self._coi_freq, 0, self.frequencies[-1])

        self._poisson_lim = self.sampler.get_poisson_limit()

    def calculate_period_length(self, freq: float, location: float) -> None:
        """
        Calculates the length of a period at a certain location in the scalogram.

        :param freq: Frequency of interest in Hz
        :param location: Location of interest in ks
        :return: None
        """
        # Calculate the period given the frequency
        period = (1 / freq) * 1E-3

        # Create line the same length as the period, at f
        line_f = np.array([freq, freq])
        line_t = np.array([location - period / 2, location + period / 2])

        # Save the line
        if self.__period_lines is None:
            self.__period_lines = [[line_f, line_t]]
        else:
            self.__period_lines.append([line_f, line_t])

    def simulate_light_curves(self, number=1, name="SIM", sim_type: Literal["TK", "EM"] = "TK", cpus=1) -> None:
        """
        Simulates a given number of light curves, fit to the PSD of self.

        :param number: Number of light curves to simulate
        :param name: Name of simulated light curve files
        :param sim_type: Use either T&K or Emmanoulopoulos methods
        :param cpus: Number of CPUs to use for multiprocessing
        :return: None
        """
        start = t.time()

        # Initialise list of simulated light curves
        self.sim_curves = []

        # Create directory
        dir_name = name + "-" + dt.now().strftime("%Y_%m_%d_%H_%M")
        os.mkdir(f"{dir_name}")

        print(f"{WARN}{cpus} processes in use{ENDC}")
        for i in range(cpus):
            print(f"[{i}] Simulating light curve 1/{number // cpus}\n")
        sys.stdout.flush()

        # Generate simulated light curves
        with cf.ProcessPoolExecutor(cpus) as exe:
            futures = {exe.submit(self._parallel_sim_handler, cp.deepcopy(self), i, number, sim_type, dir_name, name, cpus) for i in range(cpus)}

        # Wait for all futures to finish
        print(f"{WARN}Waiting for parallel processes to finish...{ENDC}")
        cf.wait(futures)

        # Combine each future's results into the list of simulated transforms
        for r in futures:
            self.sim_curves.extend(r.result())

        end = t.time()
        print(f"\n{GREN}Light curve simulation completed in {end - start:.2f} seconds{ENDC}\n")

    @staticmethod
    def _parallel_sim_handler(clone, proc_num: int, n, sim_type: Literal["TK", "EM"], dir_name, name, cpus) -> list:
        # Set seed to a truly random value to avoid duplicating other processes
        seed = secrets.randbits(64) ^ (os.getpid() << 16) ^ int(t.time() * 1e6)
        seed &= 0xFFFFFFFF
        np.random.seed(seed)

        n_lcs = n // cpus

        local_sims = []
        for i in range(n_lcs):

            if sim_type == "TK":
                sim_lc = clone.sampler.tk_sample()
            elif sim_type == "EM":
                sim_lc = clone.sampler.em_sample()
            else:
                raise ValueError(f"Unknown simulation type: '{sim_type}', must be 'TK' or 'EM'")

            # Match points to observed times (Including gaps)
            if clone.__gaps is not None:
                sim_lc.rate = [np.interp(clone.lc.time, sim_lc.time, sim_lc.rate)][0]
                sim_lc.time = clone.lc.time

            # Add to list and save to disk
            sim_lc = pylg.LightCurve(t=sim_lc.time, r=sim_lc.rate, e=sim_lc.error)
            sim_lc.write_fits(f"{dir_name}/SIM_{name}_{proc_num}_{i}.lc")

            local_sims.append(sim_lc)

            # Print update
            lines_up = cpus - proc_num
            move_up(lines_up)
            clear_line()
            sys.stdout.write(f"[{proc_num}] Simulating light curve {i + 2}/{n_lcs}\n")
            move_down(lines_up)
            sys.stdout.flush()

        # Finished
        lines_up = cpus - proc_num
        move_up(lines_up)
        clear_line()
        sys.stdout.write(f"[{proc_num}] Finished\n")
        move_down(lines_up)
        sys.stdout.flush()

        return local_sims

    def estimate_significance(self, transform: Literal["CWT", "WWZ", "WWA", "SLT"] | None, n = 95, filename: str = None, cpus = 1) -> None:
        """
        Estimates the significance of a given wavelet transform by calculating the transform for all simulated light curves.
        This method can be allowed to run transform calculation in parallel by setting `cpus` to a number greater than 1.

        :param transform: Transform to estimate significance of
        :param n: Nth percentile for significance
        :param filename: Name of output file containing the significance map at the given level for this transform.
        :param cpus: Number of CPUs to use for multiprocessing
        :return: None
        """
        start = t.time()

        # Initialise list of transforms
        self.sim_transforms = []

        # Split light curves between CPU
        lc_lists = divide_list(self.sim_curves, cpus)

        # Run each set of transform calculations on a separate process
        with cf.ProcessPoolExecutor(cpus) as exe:
            futures = {exe.submit(self._parallel_transform_handler, cp.deepcopy(self), i, lc_lists[i], transform, cpus) for i in range(cpus)}

        # Wait for all futures to finish
        print(f"{WARN}Waiting for parallel processes to finish...{ENDC}")
        cf.wait(futures)

        # Combine each future's results into the list of simulated transforms
        for r in futures:
            self.sim_transforms.extend(r.result())

        end = t.time()
        print(f"\n{GREN}Transforms completed in {end - start:.2f} seconds{ENDC}\n")

        # Set the significance contours based on the statistics of these transforms
        print(f"Generating significance contours for {transform}...")
        self.set_significance(transform, n, filename)
        print(f"\n{GREN}Significance estimation completed!{ENDC}\n")

    @staticmethod
    def _parallel_transform_handler(clone, proc_num: int, lc_list: list, transform: Literal["CWT", "WWZ", "WWA", "SLT"] | None, cpus = 1) -> list:
        # Confirm number of parallel processes
        if proc_num == 0:
            print(f"{WARN}{cpus} processes in use{ENDC}")

        # Initialise this process' own list of completed transforms
        local_transforms = []

        # Stagger the processes if calculating WWZ/A to reduce simultaneous requests for additional threads
        if transform == "WWZ" or transform == "WWA":
            t.sleep(5 * proc_num)

        # For each light curve, calculate the transform and append it to the list
        for i, lc in enumerate(lc_list):
            if proc_num == 0:
                # Track progress
                print(f"\r[{cpus} CPUs] Calculating {transform} transform {(i + 1) * cpus}/{len(clone.sim_curves)}...", end="", flush=True)

            # CWT
            if transform == "CWT":
                cwt = SuperletTransform(lc.time, lc.rate, clone.sample_rate, clone.frequencies, clone.cwt_cycles,
                                        [1]).calculate_transform()
                local_transforms.append(cwt)

            # SLT
            elif transform == "SLT":
                slt = SuperletTransform(lc.time, lc.rate, clone.sample_rate, clone.frequencies, clone.slt_params[0],
                                        [clone.slt_params[1], clone.slt_params[2]]).calculate_transform()
                local_transforms.append(slt)

            # WWZ/A
            elif transform == "WWZ" or transform == "WWA":
                with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
                    # Shut these guys up
                    wwz = WWZ(lc.time, lc.rate)
                    freq = wwz.get_freq(clone.__unit / clone.frequencies.max(), clone.__unit / clone.frequencies.min(), clone.wwz_params[0])
                    tau = wwz.get_tau(lc.time.min(), lc.time.max(), clone.wwz_params[1])

                wwz.set_freq(freq, verbose=False)
                wwz.set_tau(tau, verbose=False)
                wwz.transform(verbose=0)

                if transform == "WWZ":
                    local_transforms.append(wwz.wwz.transpose())
                else:
                    local_transforms.append(wwz.wwa.transpose())

        return local_transforms

    def set_significance(self, transform: Literal["CWT", "WWZ", "WWA", "SLT"] | None, sig: float = 95, filename: str = None) -> None:
        """
        Sets the significance level of contours.

        :param transform: Transform to set significance level for
        :param sig: Significance level
        :param filename: Name of output file containing the significance map at the given level for this transform.
        :return: None
        """
        # Stack transforms
        stacked_scalograms = np.stack(self.sim_transforms, axis=0)

        # Calculate the nth percentile for each point
        nth_percentiles = np.percentile(stacked_scalograms, sig, axis=0)

        # Save significance map to file
        if filename is not None:
            np.save(f"{filename}_{transform}_{round(sig)}", nth_percentiles)

        # Determine where the real transform is significant (Above Nth percentile)
        self.__add_significance(transform, nth_percentiles)

    def load_significance(self, transform: Literal["CWT", "WWZ", "WWA", "SLT"] | None, filename: str = None) -> None:
        """
        Loads a significance map from file.

        :param transform: Transform to set significance level for
        :param filename: Name of input file containing the significance map for this transform.
        :return: None
        """
        # Load significance map from file
        try:
            sig_map = np.load(filename)
        except FileNotFoundError:
            print(f"{ERRR}Significance map '{filename}' could not be found!")
            return
        else:
            # Determine where the real transform is significant
            self.__add_significance(transform, sig_map)

    def __add_significance(self, transform: Literal["CWT", "WWZ", "WWA", "SLT"] | None, sig_map: np.ndarray) -> None:
        """
        Appends a new significance map to the list.

        :param transform: Transform to set significance level for
        :param sig_map: Array of significance levels
        :return: None
        """
        if transform == "CWT":
            self.cwt_significance.append(np.abs(self.cwt) > sig_map)
        elif transform == "WWZ":
            self.wwz_significance.append(np.abs(self.wwz) > sig_map)
        elif transform == "WWA":
            self.wwa_significance.append(np.abs(self.wwa) > sig_map)
        elif transform == "SLT":
            self.slt_significance.append(np.abs(self.slt) > sig_map)

    def clr_significance(self, transform: Literal["CWT", "WWZ", "WWA", "SLT"] | None) -> None:
        """
        Clears all current significance contours for the given transform.

        :param transform: Transform to clear significance for
        :return: None
        """
        if transform == "CWT":
            self.cwt_significance = []
        elif transform == "WWZ":
            self.wwz_significance = []
        elif transform == "WWA":
            self.wwa_significance = []
        elif transform == "SLT":
            self.slt_significance = []

    def find_gaps(self, rm: str | tuple | bool = False) -> None:
        """
        Finds gaps in the light curve which will be masked out when plotting the scalogram.

        :return: None
        """
        # Ignore the gap masks
        if rm is True or "-r" in rm:
            self.__gaps = None
            print(f"{WARN}Gaps ignored for ID {self.code}{ENDC}")
            return

        # Initialise list of gaps
        self.__gaps = []
        gap_times = []

        # Split the light curve on gaps
        lc = pylg.LightCurve(t=self.lc.time, r=self.lc.rate)
        lc_sections = lc.split_on_gaps()

        # Locate the bounds of gaps and add them and length to arrays
        for i in range(len(lc_sections) - 1):
            self.__gaps.append([lc_sections[i].time[-1] * self.__t_unit, lc_sections[i + 1].time[0] * self.__t_unit])
            gap_times.append((lc_sections[i + 1].time[0] - lc_sections[i].time[0]) * self.__t_unit)

        print(f"Found {len(self.__gaps)} gaps, length: {np.mean(gap_times)}{self.__t_units[self.__t_unit]}")

        # If no gaps found, destroy list
        if len(self.__gaps) == 0:
            self.__gaps = None
        else:
            # Calculate the mean gap length
            self.__gap_frequency = (1 / float(np.mean(gap_times))) * self.__t_unit * self.__f_unit

    def plot_signal(self) -> None:
        """
        Plots the input light curve of the transform.

        :return: None
        """
        plt.figure(0, (16, 5), label="XSuperlet - Light Curve")
        plt.scatter(self.times * self.__t_unit * self.__unit, self.signal, marker="+", color="k")

        plt.xlabel(f"Time ({self.__t_units[self.__t_unit]})")
        plt.ylabel("Count Rate (Counts / s)")
        plt.tick_params(axis="both", direction="in", top=True, right=True)
        plt.title(self.filename, fontweight="bold")
        plt.tight_layout()
        plt.show(block=False)

    def plot_transform(self, transform: Literal["CWT", "WWZ", "WWA", "SLT"] | np.ndarray, title = "Unnamed Transform", mask_region = None,
                       v_min = None, v_max = None, norm: Literal["linear", "log", "symlog", "logit"] = "linear", cmap: str = None,
                       gap_fill: tuple=("k", 0.3), coi_fill: tuple=("k", 0.5), figure = None, axis: Axes = None) -> QuadMesh | None:
        """
        Plots either the latest CWT, WWZ, WWA, or SLT as a scalogram.
        Will also include the COI and any gaps if they exist.

        :param transform: Indicates which transform to plot
        :param title: Title of the plot
        :param mask_region: Region of the transform to be ignored
        :param v_min: Minimum colour value
        :param v_max: Maximum colour value
        :param norm: Linear/Logarithmic normalisation on colour scale
        :param cmap: Colour map to use, if none "jet" is used
        :param gap_fill: Colour and alpha value of gaps
        :param coi_fill: Colour and alpha value of cone of influence
        :param figure: Matplotlib figure to use for subplots
        :param axis: Matplotlib axis to use for subplots
        :return: None
        """
        # Setup pcolormesh edges
        if transform in ["WWZ", "WWA"]:
            times = self.wwz_time_grid
            freqs = self.wwz_freq_grid
        else:
            times = self.times * self.__unit
            freqs = self.frequencies * (1 / self.__unit)

        # Time edges
        d_t = np.diff(times * (1 / self.__unit)).mean()
        time_edges = np.concatenate([
            [times[0] * self.__t_unit - d_t / 2],
            times[:-1] * self.__t_unit + d_t / 2,
            [times[-1] * self.__t_unit + d_t / 2]])

        # Frequency edges, either log or linear space
        if round(freqs[1] - freqs[0], 5) != round(freqs[-1] - freqs[-2], 5):
            freq_edges = np.geomspace(freqs[0], freqs[-1], len(freqs) + 1)
        else:
            d_f = np.diff(freqs).mean()
            freq_edges = np.concatenate([[(freqs[0] * self.__f_unit) - d_f / 2],
                                         (freqs[:-1] * self.__f_unit) + d_f / 2,
                                         [(freqs[-1] * self.__f_unit) + d_f / 2]])

        if axis is None:
            # Initialise figure
            fig, ax = plt.subplots(figsize=(10, 7), label="XSuperlet - Scalogram")
        else:
            fig = figure
            ax = axis

        # Select the transform to plot
        if type(transform) is np.ndarray:
            transform_data = transform
            transform_sig = None
            if cmap is None:
                cmap="twilight_shifted"
        else:
            transform_data, transform_sig = self.get_transform_data(transform)
            if cmap is None:
                cmap="jet"

        # If transform doesn't exist, abort
        if transform_data is None:
            return None

        # if transform in ["WWZ", "WWA"]:
        #     temp_arr = np.zeros((transform_data.shape[0], transform_data.shape[1] + 1))
        #     temp_arr[:, :transform_data.shape[1]] = transform_data
        #     transform_data = temp_arr

        # Ignore masked region
        if mask_region is not None:
            mask = np.ones(transform_data.shape, dtype=bool)
            # Convert from user entered value to index of frequency list
            minimum = closest_index(freqs, mask_region[0])
            maximum = closest_index(freqs, mask_region[1])
            print(f"Mask indices: {minimum}, {maximum}")
            # Set the region to ignore to False
            mask[minimum:maximum, :] = False

            # Apply the mask to the data
            transform_data = np.ma.masked_array(transform_data, mask=~mask)

        # Plot the transform
        plot = ax.pcolormesh(*np.meshgrid(time_edges, freq_edges), transform_data, shading="flat", cmap=cmap, vmin=v_min, vmax=v_max, norm=norm)

        if axis is None:
            fig.colorbar(plot, label="Power (Arbitrary Units)", orientation="horizontal", ax=ax, aspect=40, shrink=0.8, pad=0.1)
            plt.subplots_adjust(top=0.95, bottom=0.001)

        # Plot gaps, if found
        if self.__gaps is not None:
            for gap in self.__gaps:
                ax.fill_betweenx([freqs[0] * self.__f_unit, freqs[-1] * self.__f_unit],
                                 gap[0], gap[1], color=gap_fill[0], alpha=gap_fill[1])
                plt.plot([self.lc.time[0] * self.__t_unit, self.lc.time[-1] * self.__t_unit],
                         [self.__gap_frequency, self.__gap_frequency], color="w", linestyle="dashed", linewidth=1)

        # Plot the significance, if determined
        if transform_sig is not None:
            axc = ax.twinx()
            contours = ["-", "--", ":", "-.", "-", "--", ":", "-."]
            for i, sig_map in enumerate(transform_sig):
                axc.contour(sig_map, colors="w", linewidths=1.4, linestyles=contours[i], extent=self.__limits)
            axc.set_yticks([])

        # Plot the COI, if calculated
        if self._coi is not None:
            ax_coi = ax.twinx()
            ax_coi.plot(self._coi * self.__t_unit, self._coi_freq * (1 / self.__unit) * self.__f_unit, color=coi_fill[0], linestyle="dashed", linewidth=1)
            ax_coi.fill_between(self._coi * self.__t_unit, self._coi_freq * (1 / self.__unit) * self.__f_unit,
                                (self.frequencies * (1 / self.__unit) * self.__f_unit).min(), color=coi_fill[0], alpha=coi_fill[1])
            ax_coi.set_yscale("log")
            ax_coi.set_xlim(self.__limits[0], self.__limits[1])
            ax_coi.set_ylim(self.__limits[2], self.__limits[3])
            ax_coi.set_yticks([])
            ax_coi.set_yticks([], minor=True)

        # Plot any period diagnostic lines
        if self.__period_lines is not None:
            for line in self.__period_lines:
                ax.plot(line[1] * 1E+3, line[0], color="magenta", linewidth=3)

        # Plot Frequency peak tracers
        ax.errorbar(self.__f_peaks_time, self.__f_peaks, yerr=self.__f_peaks_error, xerr=0, fmt="ws", markersize=6, capsize=4,)

        # TODO: Plot the Poisson noise limit

        ax.set_yscale("log")
        ax.set_xlim(self.__limits[0], self.__limits[1])
        ax.set_ylim(self.__limits[2], self.__limits[3])

        if axis is None:
            ax.set_xlabel(f"Time ({self.__t_units[self.__t_unit]})")
        ax.set_ylabel(f"Frequency ({self.__f_units[self.__f_unit]})")

        # Deal with tick formatting
        for minmaj in [True, False]:
            y_min, y_max = ax.get_ylim()
            visible_ticks = [tick for tick in ax.get_yticks(minor=minmaj) if y_min <= tick <= y_max]
            formatted_ticks = [int(tick) for tick in visible_ticks]
            ax.set_yticks(formatted_ticks, [str(tick) for tick in formatted_ticks], minor=minmaj)

        # Create minute axis
        axmin = ax.twinx()
        axmin.set_ylabel("Period (min)")

        axmin.set_ylim(1 / (60 * self.__limits[2] * (1 / self.__f_unit)), 1 / (60 * self.__limits[3] * (1 / self.__f_unit)))
        axmin.set_yscale("log")

        axmin.yaxis.set_major_formatter(ticker.FuncFormatter(tick_format))
        axmin.yaxis.set_minor_formatter(ticker.FuncFormatter(tick_format))

        # Create title
        ax.set_title(f"{self.filename}: {title}", fontweight="bold")
        if type(transform) is str:
            try:
                plt.savefig(f"XSuperletPlots/{transform}{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}.png", dpi=300)
            except FileNotFoundError:
                os.mkdir("XSuperletPlots/")
                plt.savefig(f"XSuperletPlots/{transform}{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}.png", dpi=300)
        else:
            try:
                plt.savefig(f"XSuperletPlots/Diff{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}.png", dpi=300)
            except FileNotFoundError:
                os.mkdir("XSuperletPlots/")
                plt.savefig(f"XSuperletPlots/Diff{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}.png", dpi=300)

        # Tight layout and unblocking for new plots
        if axis is None:
            plt.tight_layout()
            plt.show(block=False)
            return None
        else:
            return plot

    def plot_slice(self, transform: Literal["CWT", "WWZ", "WWA", "SLT"] | None, time: float, peak_height: float, prominence: float,
                   coi_fill: tuple=("k", 0.5)) -> None:
        """
        Plots a slice of the given transform at the given time.

        :param transform: Transform to plot
        :param time: Time to plot slice at
        :param peak_height: Height of peaks to search for
        :param prominence: Minimum prominence required for a peak to be detected
        :param coi_fill: Colour and alpha value of cone of influence
        :return: None
        """
        # Select the transform to plot
        transform_data = self.get_transform_data(transform)[0]

        # If transform doesn't exist, abort
        if transform_data is None:
            return

        # Get slice
        time_index = closest_index(self.times, time)
        power = transform_data.transpose()[time_index]
        freqs = self.frequencies

        # Initialise figure
        _, ax = plt.subplots(figsize=(10, 7))

        # Plot slice
        ax.plot(freqs, power, color="r", label=f"{transform} Power")
        ax.set_xscale("log")

        # Set axes labels
        ax.set_xlabel(f"Frequency ({self.__f_units[self.__f_unit]})")
        ax.set_ylabel("Power (Arbitrary Units)")

        # Set title
        if transform == "CWT":
            ax.set_title(f"Wavelet Power @ {int(time * 1E+3)}{self.__t_units[self.__t_unit]}")
        elif transform == "SLT":
            ax.set_title(f"Superlet Power @ {int(time * 1E+3)}{self.__t_units[self.__t_unit]}")
        else:
            ax.set_title(f"{transform} Power @ {int(time * 1E+3)}{self.__t_units[self.__t_unit]}")

        # Set axes limits
        ax.set_ylim(0.0, max(power) * 1.05)
        ax.set_xlim(freqs[0], self.frequencies[-1])

        # Find peaks
        # TODO: Duplicated code fragment
        peaks, _ = find_peaks(power, height=peak_height, prominence=prominence)

        # Determine FWHM of peaks
        props = peak_widths(power, peaks, rel_height=0.5)
        left_ips = props[2]
        right_ips = props[3]

        # Convert FWHM in bins to frequency units
        fwhms = []
        for l, r in zip(left_ips, right_ips):
            # Interpolate frequencies for the exact (fractional) bin edges
            if l < 0: l = 0
            if r > len(self.frequencies) - 1: r = len(self.frequencies) - 1
            f_left = np.interp(l, np.arange(len(self.frequencies)), self.frequencies)
            f_right = np.interp(r, np.arange(len(self.frequencies)), self.frequencies)
            fwhms.append(abs(f_right - f_left))

        # Print FWHM
        for peak, fwhm in zip(peaks, fwhms):
            print(f"Peak at: {round(freqs[peak], 0):>6} +/- {f"{float(f"{fwhm/2:.1g}"):g}":<4} {self.__f_units[self.__f_unit]}")

        # Plot FWHM
        ax.errorbar(freqs[peaks], power[peaks], yerr=0, xerr=(np.array(fwhms)/2), fmt="ks", markersize=6, capsize=4, label="Peak Frequencies")

        # Plot COI
        if self._coi is not None:
            coi_index = closest_index(self._coi, time * 1E+6)
            coi_freq = self._coi_freq[coi_index]
            ax.axvline(coi_freq, color="k", linestyle="dashed", linewidth=2)
            ax.fill_betweenx(plt.ylim(), self.frequencies[0], coi_freq,
                             color=coi_fill[0], alpha=coi_fill[1], zorder=99, label="COI")

        # Deal with tick formatting
        x_min, x_max = ax.get_xlim()
        visible_ticks = [tick for tick in ax.get_xticks() if x_min <= tick <= x_max]
        formatted_ticks = [int(tick) for tick in visible_ticks]
        plt.xticks(formatted_ticks, [str(tick) for tick in formatted_ticks])

        plt.legend()
        plt.show(block=False)

    def trace_frequency(self, transform: Literal["CWT", "WWZ", "WWA", "SLT"] | None, f_target: float, interval: int,
                        peak_height: float, prominence: float, offset: float) -> None:
        """
        Calculates the uncertainty of the frequency peak closest to the target frequency at each interval.
        These frequency uncertainties will be plot on the next scalogram.

        :param transform: Transform to use
        :param f_target: Approximate frequency of target peak in μHz
        :param interval: Number of seconds between each sample
        :param peak_height: Height of peaks to search for
        :param prominence: Minimum prominence required for a peak to be detected
        :param offset: Offset from zero of time interval
        :return: None
        """
        # Select the transform to plot
        transform_data = self.get_transform_data(transform)[0]

        # If transform doesn't exist, abort
        if transform_data is None:
            return

        for t in self.times:
            if round((t * self.__unit) + offset, -1) % interval == 0:
                # Get slice
                time_index = closest_index(self.times, t)
                power = transform_data.transpose()[time_index]
                freqs = self.frequencies

                # Find peaks
                # TODO: Duplicated code fragment
                peaks, _ = find_peaks(power, height=peak_height, prominence=prominence)

                # Determine FWHM
                props = peak_widths(power, peaks, rel_height=0.5)
                left_ips = props[2]
                right_ips = props[3]

                # Convert FWHM in bins to frequency units
                fwhms = []
                for l, r in zip(left_ips, right_ips):
                    # Interpolate frequencies for the exact (fractional) bin edges
                    if l < 0: l = 0
                    if r > len(self.frequencies) - 1: r = len(self.frequencies) - 1
                    f_left = np.interp(l, np.arange(len(self.frequencies)), self.frequencies)
                    f_right = np.interp(r, np.arange(len(self.frequencies)), self.frequencies)
                    fwhms.append(abs(f_right - f_left))

                # Determine which peak is closest to target
                peak_fs, distance_to_target = [], []
                for peak, _ in zip(peaks, fwhms):
                    distance_to_target.append(abs(f_target - freqs[peak]))
                    peak_fs.append(freqs[peak])

                # Append the closest peak to lists if one is found
                try:
                    target_index = distance_to_target.index(min(distance_to_target))
                except ValueError:
                    print(f"{WARN}{"["+str(round(t * 1E+3, 3)) + f" {self.__t_units[self.__t_unit]}]":>12} No peak found for target frequency!")
                else:
                    print(f"{"["+str(int(round(t * 1E+3, 3))) + f" {self.__t_units[self.__t_unit]}]":>12} Target found at:"
                          f"{f"{float(f"{round(peak_fs[target_index], 0)}"):g}":>5}"
                          f" +/- {f"{float(f"{fwhms[target_index] / 2:.1g}"):g}":<4} {self.__f_units[self.__f_unit]}")

                    self.__f_peaks.append(peak_fs[target_index])
                    self.__f_peaks_time.append(t * self.__unit * self.__t_unit)
                    self.__f_peaks_error.append(fwhms[target_index] / 2)

    def clear_frequency_tracers(self) -> None:
        """
        Deletes all frequency tracers from the object.
        They will no longer appear on scalograms.

        :return: None
        """
        self.__f_peaks = []
        self.__f_peaks_time = []
        self.__f_peaks_error = []
