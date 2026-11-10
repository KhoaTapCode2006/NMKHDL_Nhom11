"""
Khử trùng lặp các phần tử (Elements) xuyên version và ánh xạ lại cạnh cây phân cấp (Hierarchy).
- Gộp các phần tử có cùng (kind, nội_dung) giữa các version về chung 1 element_id.
- Mỗi version giữ 1 nút gốc Document riêng (<arxiv_id>#root1, #root2...).
- Cập nhật các cạnh trong bảng hierarchy với version kiểu int và parent_id không null.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Any


class ElementDeduplicator:
    """Khử trùng lặp phần tử giữa các version của bài báo và ánh xạ lại bảng hierarchy."""

    @classmethod
    def normalize_content_key(cls, content: str) -> str:
        """Chuẩn hóa khoảng trắng của chuỗi nội dung để so khớp trùng lặp."""
        return re.sub(r"\s+", " ", content).strip()

    @classmethod
    def deduplicate_publication(
        cls,
        arxiv_id: str,
        version_parsed_data: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
    ) -> Dict[str, Any]:
        """
        Khử trùng lặp phần tử trên tất cả các version của 1 bài báo.
        Trả về bảng elements duy nhất, bảng cạnh hierarchy và thống kê trước/sau dedup.
        """
        unique_elements: List[Dict[str, Any]] = []
        # Ánh xạ (kind, nội_dung_chuẩn_hóa) -> element_id
        content_to_id: Dict[Tuple[str, str], str] = {}

        # Bộ đếm theo từng loại phần tử để đánh mã ID (ví dụ: #s1, #sec1, #fig1)
        kind_counters: Dict[str, int] = {}

        def get_kind_prefix(k: str) -> str:
            if k == "sentence":
                return "s"
            if k in ("section", "chapter", "part"):
                return "sec"
            if k == "subsection":
                return "ssec"
            if k in ("subsubsection", "paragraph", "subparagraph"):
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

        # Duyệt các version theo thứ tự (v1, v2, ...)
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

                # Mỗi version có nút gốc Document riêng (<arxiv_id>#root1, #root2)
                v_num = int(v_name[1:]) if v_name.startswith("v") and v_name[1:].isdigit() else int(v_name)
                if kind == "document":
                    final_id = f"{arxiv_id}#root{v_num}"
                    raw_id_to_final_id[raw_id] = final_id
                    unique_elements.append({
                        "element_id": final_id,
                        "arxiv_id": arxiv_id,
                        "kind": kind,
                        "content": content,
                    })
                    continue

                # Khóa so khớp nội dung
                norm_key = (kind, cls.normalize_content_key(content))

                if norm_key in content_to_id:
                    # Tái sử dụng element_id đã có từ version trước
                    final_id = content_to_id[norm_key]
                else:
                    # Tạo element_id mới
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

            # Ánh xạ lại các cạnh cha-con (version kiểu int, parent_id không null, không trùng lặp cạnh)
            v_num = int(v_name[1:]) if v_name.startswith("v") and v_name[1:].isdigit() else int(v_name)
            seen_edges_in_version = set()
            for edge in raw_hierarchy:
                raw_child = edge["element_raw_id"]
                raw_parent = edge["parent_raw_id"]

                if raw_parent is None:
                    continue

                final_child = raw_id_to_final_id.get(raw_child)
                final_parent = raw_id_to_final_id.get(raw_parent)

                if final_child and final_parent is not None and final_child != final_parent:
                    edge_key = (v_num, final_child, final_parent)
                    if edge_key not in seen_edges_in_version:
                        seen_edges_in_version.add(edge_key)
                        unified_hierarchy.append({
                            "arxiv_id": arxiv_id,
                            "version": v_num,
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
