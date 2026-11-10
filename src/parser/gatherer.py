"""
Multi-file gathering for LaTeX publications.
Identifies root TeX file, recursively resolves \\input and \\include,
builds compilation path, and manages redundant file cleanup.
"""

from __future__ import annotations

import os
import re
import json
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Set, Tuple


class MultiFileGatherer:
    """Discovers root TeX documents and recursively inlines included files."""

    # Matches LaTeX comments (% not preceded by an odd number of backslashes)
    COMMENT_PATTERN = re.compile(r"(?<!\\)(?:\\\\)*%.*$", re.MULTILINE)

    # Matches \input{...} and \include{...} (supports spaces and optional quotes)
    INPUT_INCLUDE_PATTERN = re.compile(
        r"\\(?:input|include)\s*\{[\"']?([^}\"'\n]+)[\"']?\}"
    )

    STANDARD_ROOT_NAMES = {
        "main.tex",
        "paper.tex",
        "final.tex",
        "ms.tex",
        "article.tex",
        "draft.tex",
    }

    IGNORE_KEYWORDS = {
        "elsdoc",
        "response",
        "rebuttal",
        "reply",
        "author_response",
        "referee",
        "cover_letter",
    }

    @classmethod
    def strip_comments(cls, text: str) -> str:
        """Strip LaTeX comments from text while preserving escaped %."""
        return cls.COMMENT_PATTERN.sub("", text)

    @classmethod
    def find_root_file(
        cls, version_dir: Path, metadata: Optional[dict] = None
    ) -> Optional[Path]:
        """
        Locates the root .tex file containing \\documentclass or \\documentstyle.
        Uses heuristics and metadata title matching in case of multiple candidates.
        """
        tex_files = list(version_dir.glob("**/*.tex"))
        if not tex_files:
            return None

        candidates = []
        for tex_path in tex_files:
            try:
                content = tex_path.read_text(encoding="utf-8", errors="ignore")
                clean = cls.strip_comments(content)
                if re.search(r"\\(?:documentclass|documentstyle)\b", clean):
                    candidates.append((tex_path, content, clean))
            except Exception:
                continue

        if not candidates:
            # Fallback: check for \begin{document} if no \documentclass was explicitly found
            for tex_path in tex_files:
                try:
                    content = tex_path.read_text(encoding="utf-8", errors="ignore")
                    clean = cls.strip_comments(content)
                    if "\\begin{document}" in clean:
                        candidates.append((tex_path, content, clean))
                except Exception:
                    continue

        if not candidates:
            # Fallback: if only 1 tex file exists, use it
            if len(tex_files) == 1:
                return tex_files[0]
            return None

        if len(candidates) == 1:
            return candidates[0][0]

        # Filter out obvious auxiliary or template documents
        filtered_candidates = []
        for path, raw, clean in candidates:
            lower_name = path.name.lower()
            lower_parts = [p.lower() for p in path.parts]
            if any(ign in lower_name or ign in lower_parts for ign in cls.IGNORE_KEYWORDS):
                continue
            filtered_candidates.append((path, raw, clean))

        if not filtered_candidates:
            filtered_candidates = candidates

        if len(filtered_candidates) == 1:
            return filtered_candidates[0][0]

        # Score candidates
        title_keywords = []
        if metadata and "title" in metadata and metadata["title"]:
            # Extract distinctive alphanumeric words (>3 chars) from paper title
            title_keywords = [
                w.lower()
                for w in re.findall(r"[A-Za-z0-9]{4,}", metadata["title"])
                if w.lower() not in {"with", "from", "that", "this", "using"}
            ]

        best_candidate = None
        best_score = -1

        for path, raw, clean in filtered_candidates:
            score = 0
            # Has \begin{document}
            if "\\begin{document}" in clean:
                score += 50

            # Matches standard root names
            if path.name.lower() in cls.STANDARD_ROOT_NAMES:
                score += 30

            # Title keyword matching
            clean_lower = clean.lower()
            if title_keywords:
                matches = sum(1 for kw in title_keywords if kw in clean_lower)
                score += matches * 15

            # Prefer shallower files (closer to version root)
            depth = len(path.relative_to(version_dir).parts)
            score += max(0, 10 - depth * 2)

            # Prefer larger file size
            try:
                size_kb = path.stat().st_size / 1024
                score += min(10, size_kb / 5)
            except Exception:
                pass

            if score > best_score:
                best_score = score
                best_candidate = path

        return best_candidate

    @classmethod
    def resolve_include_path(
        cls, base_dir: Path, root_dir: Path, target_str: str
    ) -> Optional[Path]:
        """
        Resolves an include/input target string to an existing file path.
        Checks both relative to current file's directory and relative to root_dir.
        """
        # Normalize slashes and clean up
        normalized = target_str.strip().replace("\\", "/").strip('"\'')
        if not normalized:
            return None

        # Build candidate paths
        candidate_rel_paths = [normalized]
        if not normalized.lower().endswith(".tex"):
            candidate_rel_paths.append(f"{normalized}.tex")

        search_bases = [base_dir, root_dir]
        for base in search_bases:
            for rel in candidate_rel_paths:
                candidate = (base / rel).resolve()
                if candidate.is_file():
                    return candidate

        # Case-insensitive search fallback
        for base in search_bases:
            for rel in candidate_rel_paths:
                parts = rel.split("/")
                curr = base
                matched = True
                for part in parts:
                    if not curr.is_dir():
                        matched = False
                        break
                    found_child = None
                    part_lower = part.lower()
                    for child in curr.iterdir():
                        if child.name.lower() == part_lower:
                            found_child = child
                            break
                    if found_child:
                        curr = found_child
                    else:
                        matched = False
                        break
                if matched and curr.is_file():
                    return curr.resolve()

        return None

    @classmethod
    def gather(
        cls,
        version_dir: Path,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Performs full multi-file gathering for a single version directory.
        Returns:
            {
                "success": bool,
                "root_file": Optional[Path],
                "compilation_files": List[Path],
                "orphan_files": List[Path],
                "inlined_content": str,
                "error": Optional[str]
            }
        """
        version_dir = Path(version_dir).resolve()
        all_tex_files = sorted(list(version_dir.glob("**/*.tex")))

        if not all_tex_files:
            return {
                "success": False,
                "root_file": None,
                "compilation_files": [],
                "orphan_files": [],
                "inlined_content": "",
                "error": "No .tex files found in version directory (empty version)",
            }

        root_file = cls.find_root_file(version_dir, metadata=metadata)
        if not root_file:
            return {
                "success": False,
                "root_file": None,
                "compilation_files": [],
                "orphan_files": [f.resolve() for f in all_tex_files],
                "inlined_content": "",
                "error": "Failed to identify root .tex file",
            }

        visited_paths: Set[Path] = set()
        compilation_files: List[Path] = []

        def inline_recursive(current_file: Path) -> str:
            current_file = current_file.resolve()
            if current_file in visited_paths:
                return f"\n% [Cycle detected: {current_file.name}]\n"

            visited_paths.add(current_file)
            compilation_files.append(current_file)

            try:
                content = current_file.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                return f"\n% [Read error in {current_file.name}: {e}]\n"

            curr_dir = current_file.parent

            # We process line-by-line to properly honor comments
            output_lines = []
            for line in content.splitlines(keepends=True):
                # Identify comments on this line
                clean_line = cls.strip_comments(line)
                matches = list(cls.INPUT_INCLUDE_PATTERN.finditer(clean_line))

                if not matches:
                    output_lines.append(line)
                    continue

                # If there are \input or \include commands in the non-comment part
                last_idx = 0
                line_parts = []
                for match in matches:
                    start_char = match.start()
                    end_char = match.end()
                    target_str = match.group(1).strip()

                    # Keep text before match
                    line_parts.append(clean_line[last_idx:start_char])
                    last_idx = end_char

                    resolved_path = cls.resolve_include_path(
                        curr_dir, version_dir, target_str
                    )
                    if resolved_path:
                        inlined_child = inline_recursive(resolved_path)
                        line_parts.append(
                            f"\n% --- BEGIN INPUT: {resolved_path.name} ---\n"
                            f"{inlined_child}\n"
                            f"% --- END INPUT: {resolved_path.name} ---\n"
                        )
                    else:
                        # Could not resolve file: keep original line comment
                        line_parts.append(
                            f"\n% [Unresolved include: {target_str}]\n"
                        )

                line_parts.append(clean_line[last_idx:])
                output_lines.append("".join(line_parts) + "\n")

            return "".join(output_lines)

        inlined_content = inline_recursive(root_file)

        compilation_set = {f.resolve() for f in compilation_files}
        orphan_files = [
            f.resolve() for f in all_tex_files if f.resolve() not in compilation_set
        ]

        return {
            "success": True,
            "root_file": root_file,
            "compilation_files": compilation_files,
            "orphan_files": orphan_files,
            "inlined_content": inlined_content,
            "error": None,
        }

    @classmethod
    def clean_redundant_files(
        cls,
        version_dir: Path,
        compilation_files: List[Path],
        orphan_files: List[Path],
        keep_root: bool = False,
    ) -> List[Path]:
        """
        Deletes redundant .tex files (both orphan files and included files that
        have already been gathered/inlined). If keep_root is False, all .tex files
        in the version directory are removed.
        Returns the list of deleted file paths.
        """
        deleted_files: List[Path] = []
        files_to_delete = set(orphan_files)

        if not keep_root:
            files_to_delete.update(compilation_files)
        else:
            # Delete all included compilation files except root
            if compilation_files:
                files_to_delete.update(compilation_files[1:])

        for file_path in sorted(files_to_delete):
            try:
                if file_path.is_file():
                    file_path.unlink()
                    deleted_files.append(file_path)
            except Exception:
                pass

        # Clean empty subdirectories in version_dir
        for root, dirs, files in os.walk(version_dir, topdown=False):
            for d in dirs:
                dir_path = Path(root) / d
                try:
                    if dir_path.is_dir() and not any(dir_path.iterdir()):
                        dir_path.rmdir()
                except Exception:
                    pass

        return deleted_files
