"""
Wavelet (CWT) / Superlet (FASLT) transform calculation for AGN timing analysis.
Note that units used here are Ms and μHz to ensure an integer sample rate over long timescales (100s of ks).

Based on 'Time-frequency super-resolution with superlets' by Moca et al., 2021 Nature Communications
and superlet.py by Harald Bârzan and Richard Eugen Ardelean

This code takes an OOP to wavelet analysis, each wavelet is its own object, belonging to the class of MorletWavelets.
In this context, a superlet is an array of MorletWavelets with the same central frequency.

Likewise, a wavelet/superlet transform is a SuperletTransform object. CWTs are SuperletTransform objects with an order of 1,
e.g. a superlet with a single Morlet wavelet with c_1 cycles.

Author: Thomas Hodd
"""
import numpy as np
from numpy import pi
from matplotlib import pyplot as plt
from typing import Literal
from scipy.signal import fftconvolve

COLOURS = ["dodgerblue", "orangered", "forestgreen", "deeppink", "darkturquoise", "orange",
           "darkorchid", "lawngreen", "mediumblue", "violet", "black", "grey", "peru"]

# Wavelet window constants as defined in Moca 2021.
SD_SPREAD = 6       # Number of standard deviations wavelet spans
SD_FACTOR = 2.5     # Number of standard deviations for the support window of the Morlet

class MorletWavelet:
    """
    Stores a Morlet wavelet and its parameters.
    Includes a method for visualising the wavelet.
    """
    def __init__(self, f_0: float, cycles: int, f_s: float) -> None:
        """
        Creates a Morlet wavelet.

        :param f_0: Central frequency
        :param cycles: Number of cycles
        :param f_s: Sampling frequency
        """
        self.f_0 = f_0
        self.cycles = cycles
        self.f_s = f_s

        self.morlet = None
        self.time_window = None

        self.compute_wavelet()

    def __wavelet_size(self) -> int:
        # Calculate the standard deviation
        st_dev = (self.cycles / 2) * (1 / np.abs(self.f_0)) / SD_FACTOR

        # Return wavelet size
        return int(2 * np.floor(np.round(st_dev * self.f_s * SD_SPREAD) / 2) + 1)

    def compute_wavelet(self) -> None:
        """
        (Re)Computes the Morlet wavelet with the current parameters.
        To change the parameters simply reassign them before calling this method.

        :return: None
        """
        # Wavelet size and half of the wavelet size
        ws = self.__wavelet_size()
        hs = int(np.floor(ws / 2))

        # Given the size of the wavelet, we can now define the time window
        self.time_window = (np.array(range(ws), dtype=np.float64) - hs) / self.f_s

        # Create gaussian window
        t = (np.array(range(ws), dtype=np.float64) - hs) * (SD_SPREAD / 2) / hs
        gaussian_window = np.exp(-(t ** 2) / 2)

        # Sinusoid component (Eq. 2.37)
        sinusoid = np.exp(2j * pi * self.f_0 * self.time_window)

        # The Morlet is the sinusoid combined with the gaussian window
        # Normalisation is dealt with by dividing the gaussian window by its sum
        self.morlet = sinusoid * gaussian_window / gaussian_window.sum()

    def plot(self) -> None:
        """
        Plots the Morlet wavelet.

        :return: None
        """
        plt.figure(figsize=(8, 6))
        plt.plot(self.time_window, np.real(self.morlet), label="Real Component")
        plt.plot(self.time_window, np.imag(self.morlet), label="Imaginary Component")
        plt.xlabel("Time (s)")
        plt.ylabel(r"$\psi(t)$")
        plt.legend()
        plt.show()

    def renormalised(self, arr) -> list:
        """
        Returns a renormalised Morlet wavelet.

        :param arr: Array to normalise to
        :return: Renormalised Morlet wavelet
        """
        std = np.std(arr)
        mean = np.mean(arr)
        c_mean = np.mean(self.morlet)
        c_std = np.std(self.morlet)

        new_morlet = [(r - c_mean) / c_std for r in self.morlet]
        new_morlet = [(r * std) + mean for r in new_morlet]
        return new_morlet


class SuperletTransform:
    """
    Stores a superlet transform and its associated parameters.
    Includes several methods for visualising the transform.
    """
    def __init__(self, times, signal: list | np.ndarray, f_s: int, freqs: list | np.ndarray, c_1: int, orders: list[int],
                 slt_type: Literal["add", "mult"] = "mult") -> None:
        """
        Creates a SuperletTransform object and calculates the transform.

        :param signal: Signal to calculate the transform for
        :param f_s: Sampling frequency of signal
        :param freqs: List of frequencies to calculate the transform at
        :param c_1: Initial number of cycles
        :param orders: List of orders, set to [1] for CWT, set to [min, max] for SLT
        :param slt_type: Use either additive or multiplicative superlets. Defaults to multiplicative, as per Moca 2021
        """
        self.times = times
        self.signal = signal
        self.f_s = f_s
        self.freqs = freqs
        self.c_1 = c_1

        self.signal_length = len(signal)

        self.transform = None
        self.scalogram = None

        self.__fig, self.__ax = None, None

        # Determine orders
        if len(orders) == 1:
            orders = [orders[0], orders[0]]
        elif len(orders) == 2:
            orders = orders
        else:
            raise ValueError("'order' must be of length 2: [min, max]")
        self.orders = np.linspace(orders[0], orders[1], len(freqs))

        # Create the list of superlets, each superlet is a list of MorletWavelet objects.
        self.superlets = [[] for _ in range(len(freqs))]
        for i, fc in enumerate(freqs):
            num_wavelets = int(np.ceil(self.orders[i]))
            for k in range(num_wavelets):
                if slt_type == "add":
                    self.superlets[i].append(MorletWavelet(fc, (k + 1) + c_1, f_s))
                elif slt_type == "mult":
                    self.superlets[i].append(MorletWavelet(fc, (k + 1) * c_1, f_s))

        # Calculate the scalogram limits
        self.__limits = (float(self.times[0]), float(self.times[-1]),
                         float(self.freqs[0]), float(self.freqs[-1]))

    def calculate_transform(self, show_scalogram=False, show_wavelet=False, *args) -> np.ndarray:
        """
        Calculates the transform for the current data and parameters.
        This method is based on that used by Harald Bârzan and Richard Eugen Ardelean in superlet.py.

        :return: None
        """
        # Initialise result array, this collects each transform
        result = np.zeros(self.signal_length)

        # Ensure self.transform is zeroed out in case a transform was previously calculated
        self.transform = np.zeros((len(self.freqs) , self.signal_length))

        # For each frequency
        for i in range(len(self.freqs)):
            result.fill(1)

            # Determine the number of wavelets at this frequency
            num_wavelets = int(np.floor(self.orders[i]))
            geometric_exp = 1 / num_wavelets

            # For each wavelet, determine the FFT with the signal
            fft_convolves = []
            for k in range(num_wavelets):
                fft_convolve = fftconvolve(self.signal, self.superlets[i][k].morlet, "same")
                if show_wavelet:
                    fft_convolves.append(fft_convolve)
                result *= 2 * np.abs(fft_convolve) ** 2

            # Fractional wavelet (FASLT)
            if self.orders[i] - int(self.orders[i]) != 0 and len(self.superlets[i]) == num_wavelets + 1:
                exponent = self.orders[i] - num_wavelets
                geometric_exp = 1 / (num_wavelets + exponent)

                fft_convolve = fftconvolve(self.signal, self.superlets[i][num_wavelets].morlet, "same")
                result *= (2 * np.abs(fft_convolve) ** 2) ** exponent

            if show_wavelet:
                _, ax = plt.subplots(2, 1, figsize=(8, 6), num=42, label=f"Wavelet/Signal Response")

                ax[0].scatter(self.times, self.signal, marker="+", color="k", alpha=0.4)
                ax[1].plot(self.times, result ** geometric_exp, color="k")

                for k in range(num_wavelets):
                    ax[0].plot(self.superlets[i][k].time_window + self.times[-1] / 2, self.superlets[i][k].renormalised(self.signal),
                             color=COLOURS[k], label=str(self.superlets[i][k].cycles), zorder=100 - k)
                    ax[1].plot(self.times, np.abs(fft_convolves[k]), color=COLOURS[k], label=str(self.superlets[i][k].cycles), zorder=100 - k, alpha=0.2)

                ax[0].set_title(f"{round(self.freqs[i], 1)} μHz")
                ax[1].set_xlabel(f"Time (Ms)")
                ax[0].set_ylabel("Count Rate (Counts / s)")
                ax[1].set_ylabel("Power (Arbitrary Units)")
                ax[0].tick_params(axis="both", direction="in", top=True, right=True)
                ax[1].tick_params(axis="both", direction="in", top=True, right=True)
                ax[0].legend(title="Cycles")

                ax[0].set_xlim(self.__limits[0], self.__limits[1])
                ax[1].set_xlim(self.__limits[0], self.__limits[1])

                if len(args) > 1:
                    ax[1].set_ylim(0, float(args[1]))

                plt.show(block=False)
                plt.pause(0.01)
                plt.clf()

            # Calculate the geometric mean of the transform at this frequency
            self.transform[i, :] += result ** geometric_exp

            if show_scalogram and i % 10 == 0:
                self.plot_scalogram()

        return self.transform

    def plot_signal(self) -> None:
        """
        Plots the signal data.

        :return: None
        """
        t_grid = np.linspace(0, self.signal_length / self.f_s, self.signal_length)
        plt.figure(figsize=(8, 6))
        plt.plot(t_grid, self.signal)
        plt.xlabel("Time (Ms)")
        plt.ylabel("Rate (counts/s)")
        plt.show()

    def plot_superlet(self) -> None:
        """
        Plots all the wavelets that make up the longest superlet.

        :return: None
        """
        plt.figure(figsize=(8, 6))

        for wave in self.superlets[-1]:
            plt.plot(wave.time_window, np.real(wave.morlet), label=wave.cycles)

        plt.legend(title="Cycles")
        plt.xlabel("Time (Ms)")
        plt.ylabel(r"$\psi(t)$")
        plt.show()

    def plot_scalogram(self) -> None:
        """
        Plots the scalogram of the transform.

        :return: None
        """
        if self.scalogram is None:
            # Define pcolormesh edges
            d_t = np.diff(self.times).mean()
            time_edges = np.concatenate([[self.times[0]  - d_t / 2], self.times[:-1] + d_t / 2, [self.times[-1] + d_t / 2]])
            freq_edges = np.geomspace(self.freqs[0], self.freqs[-1], len(self.freqs) + 1)

            # Create figure and scalogram
            self.__fig, self.__ax = plt.subplots(figsize=(10, 7), label="Wavelet Scalogram")
            self.scalogram = self.__ax.pcolormesh(*np.meshgrid(time_edges, freq_edges), self.transform, cmap="jet", shading="flat")

            # Axes limits and labels
            self.__ax.set_xlim(self.__limits[0], self.__limits[1])
            self.__ax.set_ylim(self.__limits[2], self.__limits[3])
            self.__ax.set_yscale("log")
            plt.xlabel("Time (Ms)")
            plt.ylabel("Frequency (μHz)")

            # Colour bar
            self.__fig.colorbar(self.scalogram, label="Power (Arbitrary Units)", orientation="horizontal", ax=self.__ax, aspect=40, shrink=0.8, pad=0.1)
            plt.subplots_adjust(top=0.95, bottom=0.001)
            plt.show(block=False)

            # Deal with tick formatting
            y_min, y_max = self.__ax.get_ylim()
            visible_ticks = [tick for tick in self.__ax.get_yticks() if y_min <= tick <= y_max]
            formatted_ticks = [int(tick) for tick in visible_ticks]
            self.__ax.set_yticks(formatted_ticks, [str(tick) for tick in formatted_ticks])

        else:
            # Update the data of the existing scalogram
            self.scalogram.set_array(self.transform)
            self.__ax.draw_artist(self.scalogram)
            self.__fig.canvas.flush_events()
            plt.pause(1E-3)
