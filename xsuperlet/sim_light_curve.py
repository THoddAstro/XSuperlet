"""
Light curve simulation based on: https://github.com/lena-lin/emmanoulopoulos

Following the methods of Timmer & Koenig (1995) and Emmanoulopoulos et al. (2013)

Author: Thomas Hodd

Date - 24th August 2025

Version - 1.0
"""
import numpy as np
import matplotlib.pyplot as plt
from pylag import LightCurve, Periodogram
from scipy.optimize import curve_fit
from scipy.stats import norm, randint, gamma, lognorm
from scipy.integrate import simpson
from sklearn.neighbors import KernelDensity


class LightCurveSampler:
    """
    LightCurveSampler objects can be used to generate simulated light curves with a periodogram (PSD) and probability density function (PDF)
    matching that of the given light curve. Samples may be taken using either the Timmer & Koenig (1995) method (Fast, reliable, Gaussian PDF only)
    or the Emmanoulopoulos et al. (2013) method (Slow, requires PDF model).

    Parameters
    ==========
    real_lc : LightCurve
        The 'real' light curve that you want to simulate similar light curves for. Must be a pyLag LightCurve object.
    kde: bool
        If True uses KDE to fit PDF instead of a parametric gamma/lognormal model

    Attributes
    ==========
    real_lc : LightCurve
        The 'real' light curve given at initialisation.
    real_psd : Periodogram
        Periodogram of ``real_lc``, used for PSD fitting, calculated during initialisation.
    tbin : float
        Size of time bins in seconds. Binning is assumed to be evenly spaced, and is measured based on the first two bins.
    nbins : int
        Number of time bins in the light curve.
    length : float
        Total duration of the light curve in seconds.
    psd_params : list[float]
        Best-fit parameters [A, alpha_low, alpha_low_high, f_bend, c] for the PSD model.
    pdf_params : list[float]
        Best-fit parameters [shape_ln, scale_ln, a_gamma, scale_gamma, weight] for the PDF model.

    Methods
    =======
    tk_sample(include_poisson: bool, debug_plot: bool) -> LightCurve:
        Returns a single simulated light curve sampled using the Timmer & Koenig method.
    em_sample(tk_lc: LightCurve, tol: float, include_poisson: bool, debug_plot: bool) -> LightCurve:
        Returns a single simulated light curve sampled using the Emmanoulopoulos method.
    """
    def __init__(self, real_lc: LightCurve, kde: bool = False) -> None:
        self.real_lc = real_lc
        self.real_psd = Periodogram(self.real_lc)

        # Determine bin size, assuming even binning
        self.tbin = real_lc.time[1] - real_lc.time[0]
        self.nbins = len(real_lc.time)
        self.length = real_lc.time[-1] - real_lc.time[0]

        # Constants
        self.red_noise = 1
        self.aliasing_tbin = 1
        self.n_rnoise_bins = self.red_noise * self.aliasing_tbin * self.nbins

        # A, alpha_low, alpha_high, f_bend, c
        self.psd_params = [0, 0, 0, 0, 0]

        # shape_ln, scale_ln, a_gamma, scale_gamma, weight
        self.pdf_params = [0, 0, 0, 0, 0]

        # KDE PDF
        self.kde = kde
        self.kde_width = 0.4
        self.kde_pdf = None

        # Fit and save PSD and PDF models
        self.__fit_psd(debug_plot=False)
        self.fit_pdf(debug_plot=False)
        self.model_psd = self.__power_spectral_density(self.real_psd.freq, *self.psd_params)
        self.model_pdf = self.__pdf_model(self.real_lc.rate, *self.pdf_params)

    @staticmethod
    def __power_spectral_density(f, amp, alpha_low, alpha_high, f_bend, c) -> float | np.ndarray:
        """
        Flexible PSD model using a smoothly bending power law plus poisson noise level.
        See Emmanoulopoulos et al. (2013), eq. 2.

        :param f: Frequency
        :param amp: Normalisation
        :param alpha_low: Slope below
        :param alpha_high: Slope above
        :param f_bend: Bend frequency
        :param c: Poisson noise level
        :return: Power Spectral Density (PSD) at f
        """
        numerator = amp * f ** (-alpha_low)
        denominator = 1 + (f / f_bend) ** (alpha_high - alpha_low)
        return numerator / denominator + c

    def __log_power_spectral_density(self, *args) -> float | np.ndarray:
        """
        Log-space PSD model.
        :param args: PSD model parameters
        :return: log(PSD) at f
        """
        return np.log(self.__power_spectral_density(*args))

    def __fit_psd(self, freqs: np.ndarray = None, psd: np.ndarray = None, debug_plot: bool = False) -> None | list:
        """
        Fits the PSD of the light curve. If both `freqs` and `psd` are None, they are assumed to be those of the real light curve.

        :param freqs: Frequencies array
        :param psd: PSD values array
        :param debug_plot: Shows a plot for debugging and prints fit parameters
        :return: None (for real lc) or list of fit params (for other PSDs)
        """
        if freqs is None and psd is None:
            freqs = self.real_psd.freq
            psd = self.real_psd.periodogram
            set_self = True
        else:
            set_self = False

        # Mask out near-zero frequencies
        mask = freqs > 1E-10
        freqs = freqs[mask]
        psd = psd[mask]
        log_psd = np.log(psd)

        # Initial guess for the parameters
        # A, alpha_low, alpha_high, f_bend, c
        initial_guess = [1e-3, 1, 2.5, 1e-5, 0.1]
        bounds = (0, [1E+9, 1E+9, 1E+9, 1000, 1000])

        # Fit the model to the data
        pars, covs = curve_fit(self.__log_power_spectral_density, freqs, log_psd, p0=initial_guess, bounds=bounds, maxfev=int(1E+6))

        if debug_plot:
            # Print the best-fit parameters
            print(f"Best-fit parameters:")
            print(f"A: {pars[0]:.4f}")
            print(f"alpha_low: {pars[1]:.4f}")
            print(f"alpha_high: {pars[2]:.4f}")
            print(f"f_bend: {pars[3]:.4e}")
            print(f"c: {pars[4]:.4f}")

            # Plotting the original PSD, initial guess, and the fitted model
            plt.figure(figsize=(12, 6))
            plt.plot(freqs, psd, label="Computed PSD", c="b", alpha=0.3)
            plt.plot(freqs, self.__power_spectral_density(freqs, *initial_guess), label="Initial Guess", c="g")
            plt.plot(freqs, self.__power_spectral_density(freqs, *pars), label="Model Fit", c="r")

            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power (/Hz)")
            plt.xscale("log")
            plt.yscale("log")

            plt.legend()
            plt.show(block=False)

        # Return or save fit parameters
        if set_self:
            self.psd_params = pars
            return None
        else:
            return pars

    def get_poisson_limit(self) -> float:
        """
        Refits the PSD to the light curve to determine the Poisson noise limit.

        :Return: Poisson noise limit
        """
        self.__fit_psd()
        return self.psd_params[3]

    def __calculate_dft_psd(self, mu, dft):
        """
        Calculates the PSD of the given DFT.

        :param mu: Mean rate of light curve
        :param dft: Light curve DFT
        :return: frequencies (f_j) and PSD values (P_j)
        """
        # TODO: Check this f_j and j_max
        j_max = len(dft)
        f_j = np.arange(1, j_max + 1, 1) / (self.length * self.tbin)
        psd = ((2 * self.tbin) / (mu ** 2 * self.nbins)) * (np.abs(dft) ** 2)
        return f_j, psd


    @staticmethod
    def __pdf_model(x, s_ln, scale_ln, a_gamma, sc_gamma, w = 1) -> float | np.ndarray:
        """
        Flexible PDF model containing gamma and/or lognormal components.
        See Emmanoulopoulos et al. (2013), eq. 3.

        :param x: Model x value
        :param s_ln: Lognormal shape parameter
        :param scale_ln: Lognormal scale parameter
        :param a_gamma: Gamma shape parameter
        :param sc_gamma: Gamma centre parameter
        :param w: Gamma/lognormal weighting. w = 1 is a pure gamma model
        :return: Probability density at x / Cumulative distribution up to x
        """
        if w == 0:
            return lognorm.pdf(x, s=s_ln, loc=0, scale=scale_ln)
        elif w == 1:
            return gamma.pdf(x, a=a_gamma, scale=sc_gamma / a_gamma)
        else:
            return w * gamma.pdf(x, a=a_gamma, scale=sc_gamma / a_gamma) + (1 - w) * lognorm.pdf(x, s=s_ln, loc=0, scale=scale_ln)

    def fit_pdf(self, debug_plot=False) -> None:
        """
        Fits the PDF of the light curve.
        May use kernel density estimation to fit the PDF non-parametrically.

        :param debug_plot: Shows a plot for debugging and prints fit parameters
        :return: None
        """
        rate = self.real_lc.rate

        if self.kde:
            self.kde_pdf = KernelDensity(kernel="gaussian", bandwidth=self.kde_width)
            self.kde_pdf.fit(rate.reshape(-1, 1))
            if debug_plot:
                plt.hist(rate, bins=30, color="k", alpha=0.5, density=True)

                x_grid = np.linspace(rate.min() - 1, rate.max() + 1, 1000)
                pdf = np.exp(self.kde_pdf.score_samples(x_grid.reshape(-1, 1)))

                plt.plot(x_grid, pdf, c="fuchsia", label="KDE PDF Model")
                plt.legend()
                plt.show(block=False)
            return

        counts, bin_edges = np.histogram(rate, bins=50, range=(rate.min(), rate.max()), density=True)
        bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Initial guess for the parameters
        # shape_ln, scale_ln, a_gamma, scale_gamma, weight
        # initial_guess = [np.percentile(rate, 50), 1, np.percentile(rate, 50), 10, 1]
        # initial_guess = [12, 0.3, np.percentile(rate, 40), 5, 1]
        initial_guess = [0.5, np.mean(rate), 4, 17, 0.3]
        bounds = ([0, 0, 0.1, 17, 0], [np.inf, np.inf, np.inf, 17.1, 1])

        # Fit the model to the data
        pars, covs = curve_fit(self.__pdf_model, bin_centres, counts, p0=initial_guess, bounds=bounds, maxfev=int(1E+6), method="trf")
        self.pdf_params = pars

        if debug_plot:
            # Print the best-fit parameters
            print(f"Best-fit parameters:")
            print(f"shape_ln: {self.pdf_params[0]:.4f}")
            print(f"scale_ln: {self.pdf_params[1]:.4f}")
            print(f"a_gamma: {self.pdf_params[2]:.4f}")
            print(f"scale_gamma: {self.pdf_params[3]:.4f}")
            print(f"weight: {self.pdf_params[4]:.4f}")

            x_fit = np.linspace(min(rate), max(rate), 100)

            y_guess0 = self.__pdf_model(x_fit, *initial_guess[:-1], w=0)
            y_guess1 = self.__pdf_model(x_fit, *initial_guess[:-1], w=1)

            plt.plot(x_fit, y_guess0, c="magenta", label="Lognorm [0]")
            plt.plot(x_fit, y_guess1, c="cyan", label="Gamma [1]")
            plt.plot(x_fit, self.__pdf_model(x_fit, *initial_guess), c="lime", label="Model")

            plt.legend()
            plt.show()

            plt.hist(rate, bins=bin_edges, color="k", alpha=0.5, density=True)

            y_guess = self.__pdf_model(x_fit, *initial_guess)
            y_guess /= simpson(x=x_fit, y=y_guess)
            y_fit = self.__pdf_model(x_fit, *pars)
            y_fit /= simpson(x=x_fit, y=y_fit)
            y_rate = counts / simpson(x=bin_centres, y=counts)

            plt.plot(bin_centres, y_rate, c="b", label="Histogram")
            plt.plot(x_fit, y_guess, c="g", label="Initial Guess")
            plt.plot(x_fit, y_fit, c="r", label="PDF Model")
            plt.legend()
            plt.show(block=False)

    def __sample_pdf(self, n_samples: int) -> np.ndarray:
        """
        Sample the modelled PDF based on the fit parameters.

        :param n_samples: Number of samples to generate (Usually the same as the real light curve)
        :return: Array of sample values
        """
        if self.kde:
            samples = self.kde_pdf.sample(n_samples).ravel()
            samples = np.where(samples < 0, 0, samples)
            return samples

        shape_ln, scale_ln, a_gamma, scale_gamma, weight = self.pdf_params
        gamma_size = min(np.random.poisson(weight * n_samples), n_samples)
        ln_size = n_samples - gamma_size

        return np.append(gamma.rvs(a=a_gamma, scale=scale_gamma / a_gamma, size=gamma_size), lognorm.rvs(s=shape_ln, scale=scale_ln, size=ln_size))

    def tk_sample(self, include_poisson=True, debug_plot=False) -> LightCurve:
        """
        Draws a simulated light curve by the method of Timmer & Koenig (1995).

        :param include_poisson: Include Poisson noise component in PSD. Set to False in Emmanoulopoulos sampling
        :param debug_plot: Shows a plot for debugging
        :return: Simulated light curve
        """
        # Unpack PSD parameters
        amp, slope_low, slope_high, bend_f, p_noise = self.psd_params
        if not include_poisson:
            p_noise = 0

        # Determine j_max for light curve
        j_max = self.n_rnoise_bins // 2

        # Draw real and imaginary components from normal distributions
        re = norm.rvs(loc=0, scale=1, size=j_max)
        im = norm.rvs(loc=0, scale=1, size=j_max - 1)

        # Deal with even/odd number of bins. Even has im = 0, odd has im != 0
        if self.n_rnoise_bins % 2 == 0:
            imag = np.append(im, 0)
        else:
            imag = np.append(im, norm.rvs(loc=0, scale=1, size=1))

        # Calculate PSD model, with/without a poisson noise component
        f_j = np.arange(1, j_max + 1, 1) / (self.n_rnoise_bins * self.tbin / self.aliasing_tbin)
        psd = self.__power_spectral_density(f_j, amp, slope_low, slope_high, bend_f, p_noise)

        # Add sqrt(1/2 * PSD) * complex number to list
        complex_numbers = np.append(0, np.sqrt(0.5 * psd) * (re + imag * 1j))

        # Obtain lightcurve of sample by inverse FFT
        ifft = np.fft.irfft(complex_numbers, self.n_rnoise_bins)

        # Cut sim lightcurve down to length N
        if self.red_noise > 1:
            rand = randint(self.nbins * self.aliasing_tbin - 1, self.nbins * self.aliasing_tbin * (self.red_noise - 1)).rvs()
            sampled_lc = ifft[rand: rand + self.nbins * self.aliasing_tbin]
        else:
            sampled_lc = ifft

        if self.aliasing_tbin != 1:
            sampled_lc = sampled_lc[::self.aliasing_tbin]

        # Renormalise/shift to real lightcurve rates
        sampled_lc = (sampled_lc - np.mean(sampled_lc)) / np.std(sampled_lc) * np.std(self.real_lc.rate) + np.mean(self.real_lc.rate)

        # Convert to pylag.Lightcurve
        sampled_lc = LightCurve(t=self.real_lc.time, r=sampled_lc)

        if debug_plot:
            plt.figure(figsize=(12, 6))
            plt.plot(self.real_lc.time, self.real_lc.rate, label="Real Light Curve", c="b")
            plt.plot(sampled_lc.time, sampled_lc.rate, label="Sim Light Curve", c="r")

            plt.xlabel("Time (s)")
            plt.ylabel("Rate (counts/s)")

            plt.legend()
            plt.show()

        return sampled_lc

    def em_sample(self, tk_lc: LightCurve = None, tol: float = 1E-4, include_poisson = True, debug_plot = False) -> LightCurve:
        """
        Draws a simulated light curve by the method of Emmanoulopoulos et al. (2013).

        :param tk_lc: Timmer & Koenig light curve, if None, one will be simulated with the default parameters
        :param tol: Tolerance for convergence test
        :param include_poisson: Include the new Poisson noise in the final simulated light curve
        :param debug_plot: Shows a plot for debugging
        :return: Simulated light curve
        """
        ## Step 1 ##
        # If not given, generate the TK lightcurve (x_norm(t))
        if tk_lc is None:
            tk_lc = self.tk_sample(include_poisson=False)

        # Calculate the DFT, amplitude, and PSD for x_norm (eqs. A1, A2, A6)
        dft_norm = np.fft.rfft(tk_lc.rate)
        a_norm = np.abs(dft_norm) / self.nbins
        f_j, psd_norm = self.__calculate_dft_psd(np.mean(tk_lc.rate), dft_norm)
        psd_pars = self.__fit_psd(freqs=f_j, psd=psd_norm)

        ## Step 2 ##
        # Draw N random numbers from PDF model
        x_sim = self.__sample_pdf(self.nbins)

        cycle_num = 1
        while cycle_num != -1:
            # Calculate the DFT, and phase (argument) for x_sim (eqs. A1, A3)
            dft_sim = np.fft.rfft(x_sim)
            phase_sim = np.angle(dft_sim)

            ## Step 3: Spectral Adjustment ##
            # Replace the amplitudes a_sim with the amplitudes a_norm to obtain the adjusted DFT
            dft_adj = a_norm * self.nbins * np.exp(1j * phase_sim)
            x_adj = np.fft.irfft(dft_adj, self.nbins)

            ## Step 4: Amplitude Adjustment ##
            # The new light curve is created by ordering x_sim based on the values of x_adj
            x_sim_sorted = np.argsort(x_sim)
            x_adj_sorted = np.argsort(x_adj)
            x_adj[x_adj_sorted] = x_sim[x_sim_sorted]

            ## Step 5: Convergence Test ##
            # The adjusted light curve will be the next x_sim, unless convergence is reached
            x_sim = x_adj

            # Calculate the new DFT and PSD
            dft_sim = np.fft.rfft(x_sim)
            f_j, psd_sim = self.__calculate_dft_psd(np.mean(x_sim), dft_sim)
            psd_sim_pars = self.__fit_psd(freqs=f_j, psd=psd_sim)

            # Test for convergence
            if not any(abs(s - t) > tol for s, t in zip(psd_sim_pars, psd_pars)):
                # print(f"EM light curve simulation complete in {cycle_num} cycles")
                cycle_num = -2
            elif cycle_num == 100:
                # print(f"EM light curve simulation stopped at {cycle_num} cycles")
                cycle_num = -2

            psd_pars = psd_sim_pars
            cycle_num += 1

        ## Step 6: Add Poisson Noise ##
        # Add Poisson noise back into the light curve (Section 2.2)
        if include_poisson:
            x_sim = np.random.poisson(x_sim * self.tbin) / self.tbin

        if debug_plot:
            plt.figure(figsize=(12, 6))
            plt.plot(self.real_lc.time, self.real_lc.rate, label="Real Light Curve", c="b")
            plt.plot(self.real_lc.time, x_sim, label="Sim Light Curve", c="r")

            plt.xlabel("Time (s)")
            plt.ylabel("Rate (counts/s)")

            plt.legend()
            plt.show()

        sampled_lc = LightCurve(t=self.real_lc.time, r=x_sim)
        return sampled_lc
