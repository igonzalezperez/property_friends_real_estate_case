"""
Custom API endpoint exceptions.
"""


class PredictException(Exception):
    """Exception for prediction-related errors."""


class ModelLoadException(Exception):
    """Exception for model loading errors."""
