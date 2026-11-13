"""
Package xử lý gom file LaTeX, dựng cây phân cấp (Hierarchy) và thống kê cho Milestone 2.
Phát triển bởi Nhóm 11 - Nhập môn Khoa học Dữ liệu.
"""

from .gatherer import MultiFileGatherer
from .extractor import HierarchyExtractor
from .hierarchy_builder import HierarchyBuilder
from .dedup import ElementDeduplicator
from .benchmark import ParserBenchmark

__all__ = [
    "MultiFileGatherer",
    "HierarchyExtractor",
    "HierarchyBuilder",
    "ElementDeduplicator",
    "ParserBenchmark",
    "ParsingPipeline",
]


def __getattr__(name: str):
    if name == "ParsingPipeline":
        from .pipeline import ParsingPipeline

        return ParsingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
