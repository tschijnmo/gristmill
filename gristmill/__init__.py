"""Gristmill base module.

Public names are going to be imported here.
"""

from .generate import (
    BasePrinter, ImperativeCodePrinter, CCodePrinter, FortranPrinter,
    EinsumPrinter
)
from .optimize import optimize, verify_eval_seq, Strategy
from .utils import get_flop_cost

__version__ = '0.3.0'

__all__ = [
    'Strategy',
    'optimize',
    'verify_eval_seq',
    'get_flop_cost',
    'BasePrinter',
    'ImperativeCodePrinter',
    'CCodePrinter',
    'FortranPrinter',
    'EinsumPrinter'
]
