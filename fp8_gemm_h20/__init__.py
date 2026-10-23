"""Block-scaled FP8 GEMM for H20, with explicit model profiles."""
from .dispatch import matmul
from .config import available_profiles, select, supports

__all__ = ["matmul", "available_profiles", "select", "supports"]
