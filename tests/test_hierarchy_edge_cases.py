"""
Unit tests cho các ca biên (Corner / Edge Cases) của HierarchyBuilder:
Chạy bằng lệnh: python tests/test_hierarchy_edge_cases.py
(hoặc: pytest tests/)
"""

import unittest
import sys
from pathlib import Path

# Đảm bảo đường dẫn import tới src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser.hierarchy_builder import HierarchyBuilder


class TestHierarchyEdgeCases(unittest.TestCase):

    def test_sentence_split_with_trailing_url(self):
        """URL ở cuối câu không được nuốt dấu chấm kết thúc."""
        text = "Source code at https://arxiv.org. This is the next sentence."
        sents = HierarchyBuilder.split_sentences_safe(text)
        self.assertEqual(len(sents), 2)
        self.assertTrue(sents[0].endswith("https://arxiv.org."))
        self.assertEqual(sents[1], "This is the next sentence.")

    def test_sentence_split_with_unicode_and_lowercase(self):
        """Dấu chấm kết thúc câu nhưng câu sau bắt đầu bằng ký tự thường hoặc Unicode."""
        text = "Nghiệm hội tụ. do đó kết quả đúng."
        sents = HierarchyBuilder.split_sentences_safe(text)
        self.assertEqual(len(sents), 2)
        self.assertEqual(sents[0], "Nghiệm hội tụ.")
        self.assertEqual(sents[1], "do đó kết quả đúng.")

    def test_single_sentence_item_creates_sentence_leaf(self):
        """\\item có 1 câu duy nhất vẫn phải sinh leaf sentence con theo chuẩn."""
        tex = r"""
        \begin{itemize}
            \item Only one single sentence in this item.
        \end{itemize}
        """
        raw_elems, raw_hier, _ = HierarchyBuilder.parse_version("test", "v1", tex)
        kinds = [e["kind"] for e in raw_elems]
        self.assertIn("item", kinds)
        self.assertIn("sentence", kinds)

        # Kiểm tra quan hệ cha con item -> sentence
        item_id = [e["raw_id"] for e in raw_elems if e["kind"] == "item"][0]
        sent_id = [e["raw_id"] for e in raw_elems if e["kind"] == "sentence"][0]
        edges = [(h["element_raw_id"], h["parent_raw_id"]) for h in raw_hier]
        self.assertIn((sent_id, item_id), edges)

    def test_bibitem_trailing_clean(self):
        """\\bibitem cuối cùng không được chứa chuỗi \\end{thebibliography}."""
        tex = r"""
        \begin{thebibliography}{99}
            \bibitem{ref1} Author One, Paper One.
            \bibitem{ref2} Author Two, Paper Two.
        \end{thebibliography}
        """
        _, _, preserved_refs = HierarchyBuilder.parse_version("test", "v1", tex)
        entries = preserved_refs.get("bibitem_entries", [])
        self.assertEqual(len(entries), 2)
        for entry in entries:
            self.assertNotIn(r"\end{thebibliography}", entry)

    def test_empty_version_schema(self):
        """Version rỗng phải trả về status='empty_version' và is_empty=True."""
        out = HierarchyBuilder.build_version_hierarchy_outputs(
            arxiv_id="2510.99999",
            version_name="v1",
            raw_elements=[],
            raw_hierarchy=[],
            is_empty=True,
            status="empty_version",
        )
        self.assertEqual(out["status"], "empty_version")
        self.assertTrue(out["is_empty"])
        self.assertEqual(len(out["elements"]), 0)
        self.assertEqual(len(out["hierarchy"]), 0)
        self.assertIsNone(out["tree"]["root"])

    def test_failed_version_schema(self):
        """Version lỗi phải trả về status='failed' và is_empty=False kèm thông báo lỗi."""
        out = HierarchyBuilder.build_version_hierarchy_outputs(
            arxiv_id="2510.88888",
            version_name="v1",
            raw_elements=[],
            raw_hierarchy=[],
            is_empty=False,
            error_msg="Lỗi thử nghiệm",
            status="failed",
        )
        self.assertEqual(out["status"], "failed")
        self.assertFalse(out["is_empty"])
        self.assertEqual(out["error"], "Lỗi thử nghiệm")
        self.assertEqual(out["tree"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
