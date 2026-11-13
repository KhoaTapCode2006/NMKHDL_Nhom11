"""Backward-compatible entry point for the unified parsing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from parser.pipeline import main as _pipeline_main
from parser.pipeline import parse_args as _pipeline_parse_args


def parse_args(argv: Optional[List[str]] = None):
    return _pipeline_parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    return _pipeline_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
