"""
LaTeX Multi-File Gathering, Hierarchy Parsing, and Statistics Module.
Developed for Milestone 2 - NMKHDL Nhom 11.
"""

from .gatherer import MultiFileGatherer
from .extractor import HierarchyExtractor
from .dedup import ElementDeduplicator
from .benchmark import ParserBenchmark
from .pipeline import ParsingPipeline

__all__ = [
    "MultiFileGatherer",
    "HierarchyExtractor",
    "ElementDeduplicator",
    "ParserBenchmark",
    "ParsingPipeline",
]
