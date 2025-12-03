"""Fast query modules for optimized knowledge graph querying"""

from .fast_querier import FastKGQuerier
from .index_builder import IndexBuilder

__all__ = ['FastKGQuerier', 'IndexBuilder']