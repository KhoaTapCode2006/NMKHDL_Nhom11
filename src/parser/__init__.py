"""
Package xử lý gom file LaTeX, dựng cây phân cấp (Hierarchy) và thống kê cho Milestone 2.
Phát triển bởi Nhóm 11 - Nhập môn Khoa học Dữ liệu.
"""

from .gatherer import MultiFileGatherer
from .extractor import HierarchyExtractor
from .hierarchy_builder import HierarchyBuilder
from .dedup import ElementDeduplicator
from .benchmark import ParserBenchmark
from .pipeline import ParsingPipeline

__all__ = [
    "MultiFileGatherer",
    "HierarchyExtractor",
    "HierarchyBuilder",
    "ElementDeduplicator",
    "ParserBenchmark",
    "ParsingPipeline",
]
