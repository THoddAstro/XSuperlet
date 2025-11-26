"""
Contains 2 classes used by XSuperlet:

LightCurve: Stores real/simulated light curves for CWT or SLT calculations.
SimLightCurves: Stores/saves/loads arrays of simulated light curves for significance estimation.

Author: Thomas Hodd

Date - 24th August 2025

Version - 1.0
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pylag as pylg
from scipy.interpolate import interp1d
from datetime import datetime as dt

from .text_format import *


class LightCurve:
    """
    Class real or generated light curves for performing the Wavelet and Superlet transforms on.
    Each light curve is given a length in ks and number of samples. Various components may then be added to the light curve.
    Light curves may also overwrite themselves with real data.
    """
    def __init__(self,  length: int = 100, samples: int = 1000, file: str = None, rebin: int = 0, zerot: bool = False):
        """
        Creates a new, empty light curve.

        :param length: Length of the light curve in ks
        :param samples: Number of samples (time points) in the light curve
        :param file: File name of FITS file to load (optional)
        :param rebin: Bin size for FITS file
        :param zerot: Zero time
        """
        if file is not None:
            self.load_state = self.load(file, rebin, zerot)
        else:
            self.length = length
            self.samples = samples
            self.filename = "SIMULATED"

            # Calculate dt for the requested length and samples
            self.dt = int(length * 1E+3 / samples)

            self.time = np.linspace(self.dt, length * 1E+3, samples)
            self.rate = np.zeros_like(self.time)
            self.error = np.zeros_like(self.time)

    def load(self, filename: str, rebin: int = 0, zero: bool = True) -> bool:
        """
        Loads a light curve from a file with pylag. This will overwrite any previous data in the light curve.
        The time attribute of the light curve will be automatically zeroed at the start of the time series.

        :param filename: File to load (.fits or .lc)
        :param rebin: Bin size in seconds, if zero the light curve will not be rebinned
        :param zero: If true the light curve time series will be zeroed at the start.
        :return: True if load successful, else False
        """
        try:
            file_lc = pylg.LightCurve(filename)
        except AssertionError:
            print(f"{ERRR}File '{filename}' does not exist{ENDC}")
            return False

        # Rebin light curve
        if rebin != 0:
            file_lc = file_lc.rebin(rebin)
            self.dt = rebin

        # Ensure time is in seconds since start of time series
        if zero:
            initt = file_lc.time[0]
            for i in range(0, len(file_lc.time)):
                file_lc.time[i] = (file_lc.time[i] - initt)

        # Save rate and time
        self.__update(file_lc)

        # Recalculate other quantities
        self.length = file_lc.time[-1] - file_lc.time[0]
        self.samples = len(file_lc.rate)

        self.filename = filename

        return True

    def save(self, filename: str) -> None:
        """
        Saves the light curve to a file.

        :param filename: Name of file to save to
        :return: None
        """
        export_lc = pylg.LightCurve(t = self.time, r = self.rate, e=self.error)
        export_lc.write_fits(filename)

    def duplicate(self, repeat=2) -> None:
        """
        Duplicates the light curve so that it is twice or more longer.

        :param repeat: Number of times to duplicate the light curve
        """
        self.length *= repeat
        self.samples *= repeat

        for _ in range(0, repeat):
            self.rate = np.concatenate((self.rate, self.rate))
            self.time = np.concatenate((self.time, self.time+self.time[-1]))

    def plot(self) -> None:
        """
        Plots the light curve.

        :return: None
        """
        plt.figure(0, (16, 5), label="Light Curve")
        plt.plot(self.time * 1E-3, self.rate)

        plt.xlabel("Time (ks)")
        plt.ylabel("Counts /s")
        plt.show()

    def add_sinusoid(self, *args) -> None:
        """
        Adds a sinusoidal signal to the light curve.

        :param args: Input parameters:
        - freq (float): Frequency of the sinusoid (required).
        - amp (float, optional): Amplitude of the sinusoid (default is 1.0).
        - start (float, optional): Start time of the sinusoid in ks (default is None).
        - end (float, optional): End time of the sinusoid in ks (default is None).
        :return: None
        """
        # Validate parameters
        try:
            freq = float(args[0])
        except ValueError:
            print(f"{ERRR} '{args[0]}' is not a valid <frequency> parameter{ENDC}")
            return

        try:
            amp = float(args[1]) if len(args) > 1 else 1.0
        except ValueError:
            print(f"{ERRR} '{args[1]}' is not a valid <amplitude> parameter{ENDC}")
            return
        try:
            start = float(args[2]) if len(args) > 2 else None
        except ValueError:
            print(f"{ERRR} '{args[2]}' is not a valid <start> parameter{ENDC}")
            return

        try:
            end = float(args[3]) if len(args) > 3 else None
        except ValueError:
            print(f"{ERRR} '{args[3]}' is not a valid <stop> parameter{ENDC}")
            return

        # Set full length if none specified
        if start is None:
            start = 0
        if end is None:
            end = self.length

        # Find the indices of the start and stop times
        start = np.searchsorted(self.time, start * 1E+3)
        end = np.searchsorted(self.time, end * 1E+3)

        # Add the sinusoid component
        self.rate[start:end] += amp * np.sin(2 * np.pi * freq * self.time[start:end])

        # Ensure counts >= 0
        self.rate += np.abs(min(self.rate))

    def add_poisson_noise(self) -> None:
        """
        Adds Poisson noise to the light curve.

        :return: None
        """
        self.rate += np.random.poisson(self.rate)

    def add_red_noise(self, level: float = 1.0) -> None:
        """
        Adds red noise to the light curve.

        :param level: Red noise level
        :return:
        """
        self.rate += level * np.cumsum(np.random.normal(0, 1, len(self.rate)))

    def add_gaps(self, delete_gaps: bool, data_length: int = 200, gap_length: int = 100) -> None:
        """
        Adds periodic gaps to a light curve.

        :param delete_gaps: If True, gaps are deleted not zeroed.
        :param data_length: Number of data points before gap
        :param gap_length: Number of data points in gap
        :return: None
        """
        counter = 1
        for i, _ in enumerate(self.rate):
            if counter > data_length:
                if delete_gaps:
                    # Set to -infinity for identification and removal later
                    self.rate[i] = -np.inf
                    self.time[i] = -np.inf
                    self.error[i] = -np.inf
                else:
                    self.rate[i] = 0
                    self.error[i] = 0
            if counter == data_length + gap_length:
                counter = 0
            counter += 1

        # Remove gaps (-infinity values)
        self.rate = np.array([r for r in self.rate if r != -np.inf])
        self.time = np.array([t for t in self.time if t != -np.inf])
        self.error = np.array([e for e in self.error if e != -np.inf])

        print(f"Gaps added: {len(self.time)} data points remain")

    def bin_over_gaps(self, bin_size: int, zero_time: bool = True) -> bool:
        """
        Removes gaps and bins each section separately before recombining the light curve.

        :param bin_size: Bin size in seconds
        :param zero_time: If True, zeros the time
        :return: True if rebinning successful, else False
        """
        lc = pylg.LightCurve(t = self.time, r = self.rate)

        # Split the light curve
        lc_sections = lc.split_on_gaps()
        print(f"Split into {len(lc_sections)} sections")
        for i, lc_section in enumerate(lc_sections):
            try:
                lc_sections[i] = lc_section.rebin(bin_size)
            except AttributeError:
                print(f"{ERRR}Binning Error: Light curve section {i} has zero length! Rebinning failed{ENDC}")
                lc_sections[i] = lc_sections[i]

        # Recombine the rebinned sections
        recombined_lc: pylg.LightCurve = pylg.LightCurve()
        for lc_section in lc_sections:
            recombined_lc = recombined_lc.concatenate(lc_section)

        # Zero the time axis
        if zero_time:
            initt = recombined_lc.time[0]
            for i in range(0, len(recombined_lc.time)):
                recombined_lc.time[i] = (recombined_lc.time[i] - initt)

        self.__update(recombined_lc)

        return True

    def fill_gaps(self, val: float | str = None) -> None:
        """
        Fills the light curve gaps with a set value.

        :param val: Values to fill gaps with, if None the mean value is used
        :return: None
        """
        if val == "mean":
            val = np.mean(self.rate)

        for j in np.where(self.rate == 0.0)[0]:
            self.rate[j] = val

    def rebin(self, bin_size: int, gaps: int = 0) -> None:
        """
        Rebins the light curve by the given bin size.

        :param bin_size: Bin size in seconds
        :param gaps: If 1, will bin sections individually
        :return: None
        """
        if gaps == 1 or gaps == "1":
            self.bin_over_gaps(bin_size)
        else:
            lc = pylg.LightCurve(t=self.time, r=self.rate)
            lc = lc.rebin(bin_size)

            self.__update(lc)
        self.dt = bin_size

    def normalise(self) -> None:
        """
        Normalises the light curve.

        :return: None
        """
        self.rate = (self.rate - self.rate.min()) / (self.rate.max() - self.rate.min())

    def linear_interpolate(self) -> None:
        """
        Linearly interpolates over gaps in the light curve.

        :return: None
        """
        # Split light curve
        lc = pylg.LightCurve(t=self.time, r=self.rate)

        lc_sections = lc.split_on_gaps()
        print(f"Split into {len(lc_sections)} sections")
        for i in range(len(lc_sections) - 1):
            # Determine the 3 points either side, they will be used for interpolation
            x1 = np.median([lc_sections[i].time[-1], lc_sections[i].time[-2], lc_sections[i].time[-3]])
            x2 = np.median([lc_sections[i+1].time[0], lc_sections[i+1].time[1], lc_sections[i+1].time[2]])
            y1 = np.median([lc_sections[i].rate[-1], lc_sections[i].rate[-2], lc_sections[i].rate[-3]])
            y2 = np.median([lc_sections[i+1].rate[0], lc_sections[i+1].rate[1], lc_sections[i+1].rate[2]])

            # Create Scipy interpolation object
            interp = interp1d([x1, x2], [y1, y2], "linear")

            # Calculate the width of the gap and the number required to fill it
            width = x2 - x1
            dt = lc_sections[i].time[1] - lc_sections[i].time[0]
            bins = int(width / dt)

            print(f"Gap {i + 1}/{len(lc_sections) - 1} - Width: {width}, Bins: {bins}, Bin size: {dt}")

            # Interpolate each bin over the gap and add it to the end of this section
            for b in range(bins):
                lc_sections[i].rate = np.append(lc_sections[i].rate, interp(x1 + dt * b))
                lc_sections[i].time = np.append(lc_sections[i].time, x1 + dt * b)
                lc_sections[i].error = np.append(lc_sections[i].error, 1)

        # Recombine all sections to get the light curve with interpolated gaps
        recombined_lc: pylg.LightCurve = pylg.LightCurve()
        for lc_section in lc_sections:
            recombined_lc = recombined_lc.concatenate(lc_section)

        # Update self
        self.__update(recombined_lc)

    def __update(self, lc: pylg.LightCurve) -> None:
        """
        Updates the light curve from a given Pylag LightCurve object.

        :param lc: Pylag LightCurve object
        :return: None
        """
        self.rate = lc.rate
        self.time = lc.time
        self.error = lc.error


class SimLightCurves:
    """
    Allows for the export/import of simulated light curves.
    Intended to speed up significance estimation by using cached simulated light curves instead of simulating light curves each time.

    Parameters
    ==========
    name : str
        Name for the simulated light curves. Usually a good idea to indicate which real data they are simulated from
    lcs: list
        List of simulated light curves

    Attributes
    ==========
    name : str
        Name for the simulated light curves. Usually a good idea to indicate which real data they are simulated from
    lcs: list
        List of simulated light curves

    Methods
    =======
    write_lcs():
        Writes each light curve in the list to a new directory determined by name and time/date.
    read_lcs(self, directory) -> list[pylg.LightCurve]:
        Reads a new list of simulated light curves from the given directory and returns them.
    """
    def __init__(self, name: str, lcs: list[pylg.LightCurve] = None) -> None:
        self.name = name
        self.lcs = lcs

    def write_lcs(self) -> None:
        """
        Writes light curves to new directory.

        :return: None
        """
        dir_name = self.name + "-" + dt.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.mkdir(dir_name)
        for i, lc in enumerate(self.lcs):
            lc.write_fits(f"{dir_name}/SIM_{self.name}_{i}.lc")

    def read_lcs(self, directory) -> list[pylg.LightCurve]:
        """
        Reads simulated light curves from directory.
        This will overwrite existing light curves.

        :param directory: Directory to read
        :return: List of light curves
        """
        print(f"Importing simulated light curves from: {directory}")
        self.lcs = []
        for lc in os.listdir(directory):
            self.lcs.append(pylg.LightCurve(f"{directory}/{lc}"))

        print(f"{GREN}Successfully imported {len(self.lcs)} light curves{ENDC}")
        return self.lcs
