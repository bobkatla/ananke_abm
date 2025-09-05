"""
Ananke ABM - Agent-Based Model for synthetic population data and activity predictions.

This package provides tools and models for connecting synthetic population data
with activity predictions for agent-based modeling.
"""

__version__ = "0.1.0"
__author__ = "Bob La"
__email__ = "duc.la@monash.edu"

# Import main modules for easy access
from . import models
from . import data_generator

__all__ = [
    "models",
    "utils", 
    "data_generator",
] 