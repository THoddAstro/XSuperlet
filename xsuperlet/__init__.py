"""
XSuperlet - Wavelet/superlet analysis for X-ray light curves

Author: Thomas Hodd

Date - 24th August 2025

Version - 1.0
"""
# Light curve and simulated light curve objects
from .xlight_curves import LightCurve, SimLightCurves

# Transform handling/plotting
from .wave_transform import WaveTransform

# Low-level wavelet transform calculation
from .superlet_transform import SuperletTransform

# Light curve simulation
from .sim_light_curve import LightCurveSampler

# Terminal text formatting
from .text_format import *
