"""
Cross-version Element Deduplication and Hierarchy Edge Mapper.
Implements Milestone 2 deduplication specifications:
- Deduplicates identical elements across versions using (kind, normalized_content) key
- Keeps separate Document root elements for each version
- Remaps hierarchy edges to the unified element_ids
- Computes comprehensive deduplication statistics
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Any, Optional


class ElementDeduplicator:
    """Deduplicates publication elements across versions and remaps hierarchy."""

    @classmethod
    def normalize_content_key(cls, content: str) -> str:
        """Normalizes content string for hashing/matching (collapsing whitespace)."""
        return re.sub(r"\s+", " ", content).strip()

    @classmethod
    def deduplicate_publication(
        cls,
        arxiv_id: str,
        version_parsed_data: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
    ) -> Dict[str, Any]:
        """
        Deduplicates elements across all versions of a single publication.

        Args:
            arxiv_id: Base arXiv ID (e.g. "2411.13799")
            version_parsed_data: Map of {version_name: (elements_list, hierarchy_list)}

        Returns:
            {
                "arxiv_id": str,
                "elements": List[Dict[str, Any]],       # Deduplicated table: element_id, arxiv_id, kind, content
                "hierarchy": List[Dict[str, Any]],      # Unified hierarchy table: arxiv_id, version, element_id, parent_id
                "stats": {
                    "total_before": int,
                    "total_after": int,
                    "reduction_count": int,
                    "reduction_percentage": float,
                    "by_kind_before": Dict[str, int],
                    "by_kind_after": Dict[str, int],
                }
            }
        """
        # Element storage
        unique_elements: List[Dict[str, Any]] = []
        # Key: (kind, normalized_content) -> element_id
        content_to_id: Dict[Tuple[str, str], str] = {}

        # Counter by kind for assigning clean IDs like 2411.13799#s1, #sec1, #fig1, etc.
        kind_counters: Dict[str, int] = {}

        def get_kind_prefix(k: str) -> str:
            if k == "sentence":
                return "s"
            if k in ("section", "chapter"):
                return "sec"
            if k == "subsection":
                return "ssec"
            if k in ("subsubsection", "paragraph"):
                return "par"
            if k == "block_formula":
                return "eq"
            if k == "figure":
                return "fig"
            if k == "item":
                return "item"
            if k == "itemize":
                return "list"
            return "elem"

        total_before = 0
        by_kind_before: Dict[str, int] = {}
        unified_hierarchy: List[Dict[str, Any]] = []

        # Process versions in sorted order (v1, v2, ...)
        sorted_versions = sorted(version_parsed_data.keys())

        for v_name in sorted_versions:
            raw_elements, raw_hierarchy = version_parsed_data[v_name]
            raw_id_to_final_id: Dict[str, str] = {}

            for elem in raw_elements:
                total_before += 1
                kind = elem["kind"]
                content = elem["content"]
                raw_id = elem["raw_id"]

                by_kind_before[kind] = by_kind_before.get(kind, 0) + 1

                # Rule from Milestone 2: Each version has its own distinct Document root
                if kind == "document":
                    final_id = f"{arxiv_id}#doc_{v_name}"
                    raw_id_to_final_id[raw_id] = final_id
                    unique_elements.append({
                        "element_id": final_id,
                        "arxiv_id": arxiv_id,
                        "kind": kind,
                        "content": content,
                    })
                    continue

                # Content key for deduplication
                norm_key = (kind, cls.normalize_content_key(content))

                if norm_key in content_to_id:
                    # Reuse existing element_id
                    final_id = content_to_id[norm_key]
                else:
                    # Create new element_id
                    prefix = get_kind_prefix(kind)
                    count = kind_counters.get(prefix, 0) + 1
                    kind_counters[prefix] = count
                    final_id = f"{arxiv_id}#{prefix}{count}"

                    content_to_id[norm_key] = final_id
                    unique_elements.append({
                        "element_id": final_id,
                        "arxiv_id": arxiv_id,
                        "kind": kind,
                        "content": content,
                    })

                raw_id_to_final_id[raw_id] = final_id

            # Remap hierarchy edges for this version
            for edge in raw_hierarchy:
                raw_child = edge["element_raw_id"]
                raw_parent = edge["parent_raw_id"]

                final_child = raw_id_to_final_id.get(raw_child)
                final_parent = raw_id_to_final_id.get(raw_parent) if raw_parent else None

                if final_child:
                    unified_hierarchy.append({
                        "arxiv_id": arxiv_id,
                        "version": v_name,
                        "element_id": final_child,
                        "parent_id": final_parent,
                    })

        total_after = len(unique_elements)
        reduction_count = total_before - total_after
        reduction_pct = (
            (reduction_count / total_before * 100.0) if total_before > 0 else 0.0
        )

        by_kind_after: Dict[str, int] = {}
        for elem in unique_elements:
            k = elem["kind"]
            by_kind_after[k] = by_kind_after.get(k, 0) + 1

        return {
            "arxiv_id": arxiv_id,
            "elements": unique_elements,
            "hierarchy": unified_hierarchy,
            "stats": {
                "total_before": total_before,
                "total_after": total_after,
                "reduction_count": reduction_count,
                "reduction_percentage": round(reduction_pct, 2),
                "by_kind_before": by_kind_before,
                "by_kind_after": by_kind_after,
            },
        }
