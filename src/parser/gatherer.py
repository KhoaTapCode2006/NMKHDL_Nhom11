"""
Multi-file gathering for LaTeX publications.
Identifies root TeX file, recursively resolves \\input and \\include,
honors \\includeonly, avoids cycle misdetection via call stack tracking,
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

    # Matches \include{...}, \input{...}, and unbraced \input filename (TeX primitive syntax)
    # Group 1 & 2: \include{target}
    # Group 3 & 4: \input{target}
    # Group 5 & 6: \input target (unbraced filename up to space, %, newline or command)
    INPUT_INCLUDE_PATTERN = re.compile(
        r"\\(?:"
        r"(include)\s*\{[\"']?([^}\"'\n]+)[\"']?\}|"
        r"(input)\s*\{[\"']?([^}\"'\n]+)[\"']?\}|"
        r"(input)\s+[\"']?([a-zA-Z0-9_\-\./\\]+)[\"']?"
        r")"
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
    def parse_includeonly(cls, preamble_text: str) -> Optional[Set[str]]:
        """
        Parses \\includeonly{...} from preamble if present.
        Returns None if \\includeonly is not specified (all \\include commands allowed).
        Returns a set of normalized file names (without .tex extension) if specified.
        """
        clean = cls.strip_comments(preamble_text)
        m = re.search(r"\\includeonly\s*\{([^}]+)\}", clean)
        if not m:
            return None
        raw_list = m.group(1).split(",")
        allowed = set()
        for item in raw_list:
            norm = item.strip().replace("\\", "/").strip('"\'')
            if norm.lower().endswith(".tex"):
                norm = norm[:-4]
            if norm:
                allowed.add(norm)
                allowed.add(norm.lower())
        return allowed

    @classmethod
    def find_root_file(
        cls, version_dir: Path, metadata: Optional[dict] = None
    ) -> Optional[Path]:
        """
        Locates the root .tex file containing \\documentclass or \\documentstyle.
        Uses heuristics and metadata title matching in case of multiple candidates.
        Returns None if no candidate contains a valid document root structure.
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

        # If still no candidates with documentclass, documentstyle, or begin{document},
        # it is NOT a valid root document. Do NOT blindly pick a lone random .tex file.
        if not candidates:
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
            title_keywords = [
                w.lower()
                for w in re.findall(r"[A-Za-z0-9]{4,}", metadata["title"])
                if w.lower() not in {"with", "from", "that", "this", "using"}
            ]

        best_candidate = None
        best_score = -1

        for path, raw, clean in filtered_candidates:
            score = 0
            if "\\begin{document}" in clean:
                score += 50

            if path.name.lower() in cls.STANDARD_ROOT_NAMES:
                score += 30

            clean_lower = clean.lower()
            if title_keywords:
                matches = sum(1 for kw in title_keywords if kw in clean_lower)
                score += matches * 15

            depth = len(path.relative_to(version_dir).parts)
            score += max(0, 10 - depth * 2)

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
        normalized = target_str.strip().replace("\\", "/").strip('"\'')
        if not normalized:
            return None

        candidate_rel_paths = [normalized]
        if not normalized.lower().endswith(".tex"):
            candidate_rel_paths.append(f"{normalized}.tex")

        search_bases = [base_dir, root_dir]
        for base in search_bases:
            for rel in candidate_rel_paths:
                candidate = (base / rel).resolve()
                if candidate.is_file():
                    return candidate

        # Case-insensitive fallback
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
        Properly honors compilation path, \\includeonly, call-stack cycle detection,
        and accurately reports whether gathering was fully complete or failed.
        """
        version_dir = Path(version_dir).resolve()
        all_tex_files = sorted(list(version_dir.glob("**/*.tex")))

        if not all_tex_files:
            return {
                "success": False,
                "is_complete": False,
                "root_file": None,
                "compilation_files": [],
                "orphan_files": [],
                "inlined_content": "",
                "unresolved_includes": [],
                "cycle_errors": [],
                "read_errors": [],
                "error": "No .tex files found in version directory (empty version)",
            }

        root_file = cls.find_root_file(version_dir, metadata=metadata)
        if not root_file:
            return {
                "success": False,
                "is_complete": False,
                "root_file": None,
                "compilation_files": [],
                "orphan_files": [f.resolve() for f in all_tex_files],
                "inlined_content": "",
                "unresolved_includes": [],
                "cycle_errors": [],
                "read_errors": [],
                "error": "Failed to identify root .tex file (no valid \\documentclass or \\documentstyle)",
            }

        # Parse \includeonly if present in root file preamble
        includeonly_set = None
        try:
            root_raw = root_file.read_text(encoding="utf-8", errors="ignore")
            # Only search in preamble (before \begin{document})
            preamble_part = root_raw.split(r"\begin{document}")[0]
            includeonly_set = cls.parse_includeonly(preamble_part)
        except Exception:
            pass

        # Call stack tracks the active ancestor chain to detect real cycles
        # (Allows the same file to be legitimately included multiple times at different points)
        call_stack: List[Path] = []
        compilation_files: List[Path] = []

        unresolved_includes: List[str] = []
        cycle_errors: List[str] = []
        read_errors: List[str] = []

        def inline_recursive(current_file: Path) -> str:
            current_file = current_file.resolve()

            # Cycle detection: check against active call stack, not global set
            if current_file in call_stack:
                cycle_chain = " -> ".join([p.name for p in call_stack] + [current_file.name])
                cycle_errors.append(cycle_chain)
                return f"\n% [Cycle detected: {cycle_chain}]\n"

            call_stack.append(current_file)
            if current_file not in compilation_files:
                compilation_files.append(current_file)

            try:
                content = current_file.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                err_msg = f"{current_file.name}: {e}"
                read_errors.append(err_msg)
                call_stack.pop()
                return f"\n% [Read error in {err_msg}]\n"

            curr_dir = current_file.parent
            output_lines = []

            try:
                for line in content.splitlines(keepends=True):
                    clean_line = cls.strip_comments(line)
                    matches = list(cls.INPUT_INCLUDE_PATTERN.finditer(clean_line))

                    if not matches:
                        output_lines.append(line)
                        continue

                    last_idx = 0
                    line_parts = []
                    for match in matches:
                        start_char = match.start()
                        end_char = match.end()

                        cmd_type = match.group(1) or match.group(3) or match.group(5)
                        target_str = (match.group(2) or match.group(4) or match.group(6) or "").strip()

                        line_parts.append(clean_line[last_idx:start_char])
                        last_idx = end_char

                        # Check \includeonly constraint for \include commands
                        if cmd_type == "include" and includeonly_set is not None:
                            norm_target = target_str.replace("\\", "/").strip('"\'')
                            if norm_target.lower().endswith(".tex"):
                                norm_target = norm_target[:-4]
                            if (norm_target not in includeonly_set and 
                                norm_target.lower() not in includeonly_set):
                                # Excluded by \includeonly: skip inlining
                                line_parts.append(
                                    f"\n% [Excluded by \\includeonly: {target_str}]\n"
                                )
                                continue

                        resolved_path = cls.resolve_include_path(
                            curr_dir, version_dir, target_str
                        )
                        if resolved_path:
                            inlined_child = inline_recursive(resolved_path)
                            line_parts.append(
                                f"\n% --- BEGIN {cmd_type.upper()}: {resolved_path.name} ---\n"
                                f"{inlined_child}\n"
                                f"% --- END {cmd_type.upper()}: {resolved_path.name} ---\n"
                            )
                        else:
                            unresolved_includes.append(target_str)
                            line_parts.append(
                                f"\n% [Unresolved include: {target_str}]\n"
                            )

                    line_parts.append(clean_line[last_idx:])
                    output_lines.append("".join(line_parts) + "\n")
            finally:
                call_stack.pop()

            return "".join(output_lines)

        inlined_content = inline_recursive(root_file)

        compilation_set = {f.resolve() for f in compilation_files}
        orphan_files = [
            f.resolve() for f in all_tex_files if f.resolve() not in compilation_set
        ]

        # Incomplete if any include was unresolved, had a cycle, or failed to read
        is_complete = (
            len(unresolved_includes) == 0
            and len(cycle_errors) == 0
            and len(read_errors) == 0
        )

        error_msg = None
        if not is_complete:
            err_parts = []
            if unresolved_includes:
                err_parts.append(f"{len(unresolved_includes)} unresolved includes ({', '.join(unresolved_includes[:3])})")
            if cycle_errors:
                err_parts.append(f"{len(cycle_errors)} cycles ({', '.join(cycle_errors[:2])})")
            if read_errors:
                err_parts.append(f"{len(read_errors)} read errors")
            error_msg = f"Incomplete gathering: {'; '.join(err_parts)}"

        return {
            "success": is_complete,
            "is_complete": is_complete,
            "root_file": root_file,
            "compilation_files": compilation_files,
            "orphan_files": orphan_files,
            "inlined_content": inlined_content,
            "unresolved_includes": unresolved_includes,
            "cycle_errors": cycle_errors,
            "read_errors": read_errors,
            "error": error_msg,
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
            if compilation_files:
                files_to_delete.update(compilation_files[1:])

        for file_path in sorted(files_to_delete):
            try:
                if file_path.is_file():
                    file_path.unlink()
                    deleted_files.append(file_path)
            except Exception:
                pass

        for root, dirs, files in os.walk(version_dir, topdown=False):
            for d in dirs:
                dir_path = Path(root) / d
                try:
                    if dir_path.is_dir() and not any(dir_path.iterdir()):
                        dir_path.rmdir()
                except Exception:
                    pass

        return deleted_files