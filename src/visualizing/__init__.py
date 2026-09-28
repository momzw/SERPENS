"""Reusable visualization data and optional local web interface.

Importing this package does not require Dash or start a server.

DISCLAIMER: This directory provides visualization tools partially coded by a coding agent (GPT-6-ASTRA).
(Especially the web interface).

"""

from .data import DisplayOptions, PlotRequest, VisualizationDataProvider

__all__ = ['DisplayOptions', 'PlotRequest', 'VisualizationDataProvider']