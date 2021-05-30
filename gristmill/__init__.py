"""Gristmill base module.

Public names are going to be imported here.
"""

from .generate import (
    BasePrinter, NaiveCodePrinter, C89CodePrinter, FortranPrinter,
    EinsumPrinter, mangle_base
)
from .optimize import optimize, verify_eval_seq, ContrStrat, RepeatedTermsStrat
from .utils import get_flop_cost

__version__ = '0.8.0dev0'

__all__ = [
    'ContrStrat',
    'RepeatedTermsStrat',
    'optimize',
    'verify_eval_seq',
    'get_flop_cost',
    'BasePrinter',
    'mangle_base',
    'NaiveCodePrinter',
    'C89CodePrinter',
    'FortranPrinter',
    'EinsumPrinter'
]
