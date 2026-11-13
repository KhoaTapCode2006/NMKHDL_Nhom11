"""Clean and normalize content stored in generated hierarchy-tree JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .extractor import HierarchyExtractor


class HierarchyNormalizer:
    """Normalize node content while preserving the tree structure and IDs."""

    BLOCK_MATH_ENVIRONMENTS = (
        "equation\\*?|align\\*?|gather\\*?|multline\\*?|"
        "eqnarray\\*?|flalign\\*?|displaymath"
    )
    BLOCK_MATH_PATTERN = re.compile(
        rf"\\(?P<action>begin|end)\{{(?P<env>{BLOCK_MATH_ENVIRONMENTS})\}}",
        re.DOTALL,
    )

    @classmethod
    def normalize_content(cls, content: str, kind: Optional[str] = None) -> str:
        """Remove non-semantic LaTeX and normalize inline/display math delimiters."""
        normalized = HierarchyExtractor.standardize_math(content)

        if kind == "block_formula":
            normalized = cls._normalize_block_math_environments(normalized)

        return HierarchyExtractor.clean_latex(normalized)

    @classmethod
    def _normalize_block_math_environments(cls, content: str) -> str:
        """Normalize matched, non-nested display environments without mispairing tags."""
        stack: List[Dict[str, Any]] = []
        replacements: List[tuple[int, int, str]] = []

        for match in cls.BLOCK_MATH_PATTERN.finditer(content):
            env = match.group("env")
            if match.group("action") == "begin":
                for parent in stack:
                    parent["contains_nested_math"] = True
                stack.append(
                    {
                        "env": env,
                        "start": match.start(),
                        "body_start": match.end(),
                        "contains_nested_math": bool(stack),
                    }
                )
                continue

            if not stack or stack[-1]["env"] != env:
                continue

            frame = stack.pop()
            if frame["contains_nested_math"]:
                continue

            body = content[frame["body_start"]:match.start()].strip()
            replacement = f"\\begin{{equation}}\n{body}\n\\end{{equation}}"
            replacements.append((frame["start"], match.end(), replacement))

        for start, end, replacement in reversed(replacements):
            content = content[:start] + replacement + content[end:]
        return content

    @classmethod
    def normalize_tree(cls, tree: Dict[str, Any]) -> Dict[str, Any]:
        """Return a normalized copy of one hierarchy-tree JSON object."""
        if not isinstance(tree, dict):
            raise ValueError("Hierarchy tree must be a JSON object.")
        if "root" not in tree:
            raise ValueError("Hierarchy tree JSON is missing its 'root' field.")

        normalized = cls._copy_json_tree(tree)
        root = normalized["root"]
        if root is not None:
            cls._normalize_node(root, set())
        return normalized

    @staticmethod
    def _copy_json_tree(value: Any) -> Any:
        """Copy JSON containers iteratively so deeply nested trees avoid recursion limits."""
        if not isinstance(value, (dict, list)):
            return value

        root_copy = {} if isinstance(value, dict) else []
        copies = {id(value): root_copy}
        pending = [(value, root_copy)]
        while pending:
            source, target = pending.pop()
            items = source.items() if isinstance(source, dict) else enumerate(source)
            for key, child in items:
                if isinstance(child, (dict, list)):
                    child_copy = copies.get(id(child))
                    if child_copy is None:
                        child_copy = {} if isinstance(child, dict) else []
                        copies[id(child)] = child_copy
                        pending.append((child, child_copy))
                    if isinstance(target, dict):
                        target[key] = child_copy
                    else:
                        target.append(child_copy)
                elif isinstance(target, dict):
                    target[key] = child
                else:
                    target.append(child)
        return root_copy

    @classmethod
    def _normalize_node(cls, node: Dict[str, Any], active_nodes: Set[int]) -> None:
        pending = [(node, False)]
        while pending:
            current, exiting = pending.pop()
            if not isinstance(current, dict):
                raise ValueError("Each hierarchy node must be a JSON object.")

            node_identity = id(current)
            if exiting:
                active_nodes.remove(node_identity)
                continue
            if node_identity in active_nodes:
                raise ValueError("Hierarchy tree contains a cycle.")

            active_nodes.add(node_identity)
            content = current.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ValueError("Hierarchy node 'content' must be a string.")
                current["content"] = cls.normalize_content(
                    content,
                    kind=current.get("kind"),
                )

            children = current.get("children")
            if children is not None:
                if not isinstance(children, list):
                    raise ValueError("Hierarchy node 'children' must be a list.")
                pending.append((current, True))
                pending.extend((child, False) for child in reversed(children))
            else:
                active_nodes.remove(node_identity)

    @classmethod
    def normalize_file(cls, input_path: Path, output_path: Path) -> Path:
        """Normalize one tree file and write it to the destination, including in place."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        serialized = cls._normalized_file_content(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
        return output_path

    @classmethod
    def _normalized_file_content(cls, input_path: Path) -> str:
        try:
            tree = json.loads(input_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Unable to read hierarchy tree '{input_path}': {error}") from error

        normalized = cls.normalize_tree(tree)
        return json.dumps(normalized, ensure_ascii=False, indent=2) + "\n"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def normalize_path(
    input_path: Path,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> List[Path]:
    """Normalize a tree JSON file or all hierarchy-tree JSON files in a directory."""
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    is_directory = input_path.is_dir()
    if is_directory:
        tree_files = sorted(
            path
            for path in input_path.rglob("hierarchy_tree*.json")
            if not path.stem.endswith(".normalized")
        )
        if not tree_files:
            raise FileNotFoundError(
                f"No hierarchy_tree*.json files found under: {input_path}"
            )

        if output_dir is None:
            output_root = input_path.parent / f"{input_path.name}_normalized"
        else:
            output_root = Path(output_dir).resolve()
            if _is_relative_to(output_root, input_path):
                raise ValueError(
                    "For directory input, output directory must be outside the input "
                    "directory to avoid mixing generated files with source trees."
                )
    else:
        if input_path.suffix.lower() != ".json":
            raise ValueError("Input file must have a .json extension.")
        if input_path.stem.endswith(".normalized"):
            raise ValueError(f"Input file is already normalized: {input_path}")
        tree_files = [input_path]
        output_root = (
            Path(output_dir).resolve()
            if output_dir is not None
            else input_path.parent
        )

    output_paths: List[Path] = []
    planned_outputs: List[tuple[Path, Path, str]] = []
    for tree_file in tree_files:
        if is_directory:
            relative_path = tree_file.relative_to(input_path)
            target_dir = output_root / relative_path.parent
        else:
            target_dir = output_root

        output_path = target_dir / f"{tree_file.stem}.normalized.json"
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Pass --overwrite to replace it."
            )
        planned_outputs.append(
            (
                tree_file,
                output_path,
                HierarchyNormalizer._normalized_file_content(tree_file),
            )
        )

    for _, output_path, serialized in planned_outputs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
        output_paths.append(output_path)

    return output_paths


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean and normalize generated hierarchy-tree JSON files."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="A hierarchy_tree*.json file or a directory containing hierarchy tree files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: a sibling <input>_normalized directory for directory input).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace normalized output files that already exist.",
    )
    args = parser.parse_args(argv)

    outputs = normalize_path(
        args.input,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"Normalized {len(outputs)} hierarchy tree(s).")
    for output_path in outputs:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
