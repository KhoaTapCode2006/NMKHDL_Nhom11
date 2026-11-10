"""
Script chạy xây dựng cây phân cấp (Hierarchy Construction - Phần của Tân).
Đọc các file LaTeX trong `<pub_dir>/elements/`, dựng cây phân cấp cho từng version:
1. Giữ nguyên nội dung thô (Raw Content) ở nút lá cho Khoa chuẩn hóa.
2. Loại bỏ References khỏi cây nhưng lưu riêng vào `raw_references.json` cho Khoa tách BibTeX.
3. Gán ID nút gốc `<arxiv_id>#root1` và đảm bảo `parent_id` không null.
4. Ghi nhận version rỗng (Empty Record) nếu version không có file .tex.
5. Lưu cây phân cấp vào thư mục của từng version và thư mục `elements/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from parser.hierarchy_builder import HierarchyBuilder
from parser.dedup import ElementDeduplicator


def process_publication_hierarchy(pub_dir: Path) -> Dict[str, Any]:
    """Xây dựng cây phân cấp cho tất cả các version của 1 bài báo và lưu kết quả."""
    pub_dir = Path(pub_dir).resolve()
    arxiv_id = pub_dir.name
    elements_dir = pub_dir / "elements"
    elements_dir.mkdir(parents=True, exist_ok=True)

    # Tìm tất cả các thư mục version (v1, v2, ...)
    version_dirs = sorted(
        [
            d
            for d in pub_dir.iterdir()
            if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
        ],
        key=lambda x: int(x.name[1:]),
    )

    # Nếu thư mục vX đã bị dọn dẹp, lấy danh sách version từ file full_inlined_vX.tex
    if not version_dirs:
        inlined_files = sorted(elements_dir.glob("full_inlined_v*.tex"))
        version_names = [
            f.stem.replace("full_inlined_", "") for f in inlined_files
        ]
    else:
        version_names = [vd.name for vd in version_dirs]

    raw_version_data: Dict[str, Any] = {}
    preserved_references_by_version: Dict[str, Any] = {}
    version_records: Dict[str, Any] = {}
    compilation_manifest: Dict[str, Any] = {}

    manifest_path = elements_dir / "compilation_manifest.json"
    if manifest_path.is_file():
        try:
            compilation_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except Exception:
            compilation_manifest = {}

    tex_versions_count = 0
    empty_versions_count = 0

    for v_name in version_names:
        v_dir = pub_dir / v_name
        meta = None
        meta_p = v_dir / "metadata.json"
        if meta_p.is_file():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
            except Exception:
                pass

        inlined_tex, gather_res = HierarchyBuilder.load_version_tex_from_elements_or_source(
            pub_dir=pub_dir,
            version_name=v_name,
            metadata=meta,
        )

        # Lưu lại file LaTeX đã gộp vào elements/ nếu vừa mới gom
        if inlined_tex and gather_res:
            inlined_path = elements_dir / f"full_inlined_{v_name}.tex"
            if not inlined_path.is_file():
                inlined_path.write_text(inlined_tex, encoding="utf-8")
            compilation_manifest[v_name] = {
                "status": "success",
                "is_empty": False,
                "root_file": gather_res["root_file"].name if gather_res.get("root_file") else None,
                "compilation_files": [f.name for f in gather_res.get("compilation_files", [])],
                "orphan_files": [f.name for f in gather_res.get("orphan_files", [])],
            }

        if not inlined_tex:
            # Ghi nhận tường minh version rỗng (Empty Record) khi không có file .tex
            empty_versions_count += 1
            err_msg = (
                gather_res.get("error")
                if gather_res
                else "Không tìm thấy file .tex trong thư mục version (empty record)"
            )
            compilation_manifest[v_name] = {
                "status": "empty_version",
                "is_empty": True,
                "root_file": None,
                "compilation_files": [],
                "orphan_files": [],
                "error": err_msg,
            }
            v_out = HierarchyBuilder.build_version_hierarchy_outputs(
                arxiv_id=arxiv_id,
                version_name=v_name,
                raw_elements=[],
                raw_hierarchy=[],
                is_empty=True,
                error_msg=err_msg,
            )
            version_records[v_name] = v_out
            preserved_references_by_version[v_name] = {
                "status": "empty_version",
                "thebibliography_blocks": [],
                "bibitem_entries": [],
                "bibliography_commands": [],
                "reference_sections": [],
            }

            # Lưu bản ghi cây rỗng vào thư mục version và elements/
            if v_dir.is_dir():
                (v_dir / "hierarchy_tree.json").write_text(
                    json.dumps(v_out["tree"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            (elements_dir / f"hierarchy_tree_{v_name}.json").write_text(
                json.dumps(v_out["tree"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            continue

        # Phân tích version và dựng cây phân cấp (giữ nguyên Raw Content)
        tex_versions_count += 1
        raw_elems, raw_hier, preserved_refs = HierarchyBuilder.parse_version(
            arxiv_id=arxiv_id,
            version=v_name,
            inlined_tex=inlined_tex,
        )
        raw_version_data[v_name] = (raw_elems, raw_hier)
        preserved_references_by_version[v_name] = preserved_refs

        # Tạo dữ liệu elements, danh sách cạnh và cây phân cấp cho version hiện tại
        v_out = HierarchyBuilder.build_version_hierarchy_outputs(
            arxiv_id=arxiv_id,
            version_name=v_name,
            raw_elements=raw_elems,
            raw_hierarchy=raw_hier,
            is_empty=False,
        )
        version_records[v_name] = v_out

        # Lưu cây phân cấp vào `<pub_dir>/<v_name>/hierarchy_tree.json` và `<pub_dir>/elements/`
        if v_dir.is_dir():
            (v_dir / "hierarchy_tree.json").write_text(
                json.dumps(v_out["tree"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        (elements_dir / f"hierarchy_tree_{v_name}.json").write_text(
            json.dumps(v_out["tree"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (elements_dir / f"elements_{v_name}.json").write_text(
            json.dumps(v_out["elements"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (elements_dir / f"hierarchy_{v_name}.json").write_text(
            json.dumps(v_out["hierarchy"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Lưu riêng khối References thô để Khoa trích xuất BibTeX ở bước tiếp theo
    (elements_dir / "raw_references.json").write_text(
        json.dumps(preserved_references_by_version, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Cập nhật manifest (bao gồm cả trạng thái version rỗng)
    manifest_path.write_text(
        json.dumps(compilation_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Xuất bảng elements và hierarchy tổng hợp của bài báo
    dedup_res = ElementDeduplicator.deduplicate_publication(
        arxiv_id=arxiv_id,
        version_parsed_data=raw_version_data,
    )

    (elements_dir / "elements.json").write_text(
        json.dumps(dedup_res["elements"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (elements_dir / "hierarchy.json").write_text(
        json.dumps(dedup_res["hierarchy"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        import pandas as pd

        df_elements = pd.DataFrame(dedup_res["elements"])
        df_elements.to_csv(elements_dir / "elements.csv", index=False, encoding="utf-8")

        df_hierarchy = pd.DataFrame(dedup_res["hierarchy"])
        df_hierarchy.to_csv(elements_dir / "hierarchy.csv", index=False, encoding="utf-8")
    except Exception:
        pass

    return {
        "arxiv_id": arxiv_id,
        "total_versions": len(version_names),
        "tex_versions": tex_versions_count,
        "empty_versions": empty_versions_count,
        "elements_count": len(dedup_res["elements"]),
        "hierarchy_edges_count": len(dedup_res["hierarchy"]),
        "dedup_stats": dedup_res["stats"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Milestone 2 — Chạy xây dựng cây phân cấp (Phần của Tân)"
    )
    default_source = curr_dir / "sourcesArxiv"
    if not default_source.exists():
        default_source = curr_dir.parent / "source"

    parser.add_argument(
        "--source-dir",
        type=str,
        default=str(default_source),
        help="Đường dẫn tới thư mục chứa các bài báo (mặc định: src/sourcesArxiv hoặc ./source)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số lượng bài báo cần xử lý (tùy chọn)",
    )
    args = parser.parse_args()

    source_path = Path(args.source_dir).resolve()
    if not source_path.exists():
        print(f"Lỗi: Thư mục nguồn không tồn tại: {source_path}")
        sys.exit(1)

    pub_dirs = sorted(
        [
            d
            for d in source_path.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name != "elements"
        ]
    )
    if args.limit:
        pub_dirs = pub_dirs[: args.limit]

    print("=" * 68)
    print("  MILESTONE 2: XÂY DỰNG CÂY PHÂN CẤP (NHIỆM VỤ: TÂN)")
    print(f"  Thư mục nguồn: {source_path}")
    print(f"  Tổng số bài báo: {len(pub_dirs)}")
    print("=" * 68)

    start_time = time.perf_counter()
    total_elems = 0
    total_edges = 0
    total_tex_v = 0
    total_empty_v = 0

    for idx, p_dir in enumerate(pub_dirs, 1):
        res = process_publication_hierarchy(p_dir)
        total_elems += res["elements_count"]
        total_edges += res["hierarchy_edges_count"]
        total_tex_v += res["tex_versions"]
        total_empty_v += res["empty_versions"]
        print(
            f"[{idx:02d}/{len(pub_dirs):02d}] {res['arxiv_id']} | "
            f"versions: {res['tex_versions']} có TeX, {res['empty_versions']} rỗng | "
            f"elements: {res['elements_count']} | cạnh: {res['hierarchy_edges_count']}"
        )

    elapsed = time.perf_counter() - start_time
    print("=" * 68)
    print(f"  Hoàn thành {len(pub_dirs)} bài báo trong {elapsed:.2f}s")
    print(f"  Tổng số version: {total_tex_v} có TeX, {total_empty_v} bản ghi rỗng (Empty Records)")
    print(f"  Tổng số Elements: {total_elems} | Tổng số cạnh Hierarchy: {total_edges}")
    print("=" * 68)


if __name__ == "__main__":
    main()
