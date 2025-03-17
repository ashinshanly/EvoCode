"""
EvoCode: Evolutionary Code Generation Library
"""

__version__ = "0.1.0"

from .core import EvoCode
from .engine import EvolutionEngine
from .fitness import FitnessEvaluator
from .mutation import CodeMutation, ASTMutationVisitor 