# neuroevolution/server/errors.py

class InvalidMediatorIdError(Exception):
    """Raised when the mediator_id is invalid"""
    pass

class MissingFieldsError(Exception):
    """Raised when required fields are missing"""
    pass

class MissingGenomeError(Exception):
    """Raised when a new genome cannot be fetched"""
    pass

class ExperimentError(Exception):
    """Base class for exceptions in this module."""
    pass

class ConfigurationError(ExperimentError):
    """Raised for errors in the experiment configuration."""
    pass

class EvolutionError(ExperimentError):
    """Raised when evolution-related errors occur."""
    pass

class PopulationError(ExperimentError):
    """Raised for issues related to the population."""
    pass