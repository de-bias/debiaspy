class DebiasPyError(Exception):
    """Base exception for debiaspy."""


class DebiasSchemaError(DebiasPyError, ValueError):
    """Raised when tabular inputs do not match the debiasR contract."""


class DebiasParityError(DebiasPyError):
    """Raised when an audit comparison cannot be evaluated."""
