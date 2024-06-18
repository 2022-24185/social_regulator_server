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