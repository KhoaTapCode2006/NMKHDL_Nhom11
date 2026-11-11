import json
import tempfile
import unittest
from pathlib import Path

from src.parser.hierarchy_normalizer import HierarchyNormalizer, normalize_path


class HierarchyNormalizerTests(unittest.TestCase):
    def test_normalizes_every_node_without_changing_tree_ids_or_source(self):
        source_tree = {
            "arxiv_id": "paper",
            "version": 1,
            "root": {
                "element_id": "paper#root1",
                "kind": "document",
                "content": "Document",
                "children": [
                    {
                        "element_id": "paper#fig1",
                        "kind": "figure",
                        "content": (
                            "\\begin{figure}[htpb]\n"
                            "\\centering\n"
                            "\\includegraphics{plot}\n"
                            "\\end{figure}"
                        ),
                        "children": [],
                    },
                    {
                        "element_id": "paper#s1",
                        "kind": "sentence",
                        "content": (
                            r"We estimate \(x+1\). See $y^2$."
                            "\n\\midrule\n"
                        ),
                        "children": [],
                    },
                ],
            },
        }

        result = HierarchyNormalizer.normalize_tree(source_tree)

        self.assertIn("element_id", result["root"])
        self.assertEqual(result["root"]["element_id"], "paper#root1")
        figure = result["root"]["children"][0]
        self.assertEqual(figure["element_id"], "paper#fig1")
        self.assertNotIn("[htpb]", figure["content"])
        self.assertNotIn(r"\centering", figure["content"])
        sentence = result["root"]["children"][1]
        self.assertEqual(sentence["element_id"], "paper#s1")
        self.assertIn("$x+1$", sentence["content"])
        self.assertIn("$y^2$", sentence["content"])
        self.assertNotIn(r"\midrule", sentence["content"])
        self.assertIn(r"\(x+1\)", source_tree["root"]["children"][1]["content"])

    def test_normalizes_display_math_environments_to_equation(self):
        content = (
            "$$x = 1$$\n"
            r"\begin{align*}a &= b \\ c &= d\end{align*}"
            "\n"
            r"\[z = 3\]"
        )

        normalized = HierarchyNormalizer.normalize_content(
            content,
            kind="block_formula",
        )

        self.assertEqual(normalized.count(r"\begin{equation}"), 3)
        self.assertEqual(normalized.count(r"\end{equation}"), 3)
        self.assertNotIn("$$", normalized)
        self.assertNotIn(r"\begin{align*}", normalized)
        self.assertNotIn(r"\[", normalized)

    def test_cleanup_removes_formatting_commands_and_placement_only(self):
        content = (
            r"\begin{table*}[!htbp]"
            "\n\\centering\n"
            "A & B \\\\\n"
            "\\midrule\n"
            r"\end{table*}"
        )

        normalized = HierarchyNormalizer.normalize_content(content, kind="figure")

        self.assertNotIn("[!htbp]", normalized)
        self.assertNotIn(r"\centering", normalized)
        self.assertNotIn(r"\midrule", normalized)
        self.assertIn(r"\begin{table*}", normalized)
        self.assertIn(r"\end{table*}", normalized)

    def test_empty_version_tree_is_preserved(self):
        tree = {"status": "empty_version", "root": None, "elements_count": 0}
        self.assertEqual(HierarchyNormalizer.normalize_tree(tree), tree)

    def test_directory_normalization_writes_separate_files_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "publication"
            input_dir.mkdir()
            source_file = input_dir / "hierarchy_tree_v1.json"
            source_file.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "root": {
                            "element_id": "paper#root1",
                            "kind": "document",
                            "content": r"\centering Document",
                            "children": [],
                        },
                    }
                ),
                encoding="utf-8",
            )

            outputs = normalize_path(input_dir)
            output_file = outputs[0]

            self.assertTrue(output_file.is_file())
            self.assertEqual(source_file.name, "hierarchy_tree_v1.json")
            saved = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["root"]["content"], "Document")
            with self.assertRaises(FileExistsError):
                normalize_path(input_dir)

    def test_malformed_tree_is_rejected(self):
        with self.assertRaises(ValueError):
            HierarchyNormalizer.normalize_tree({"status": "success"})


if __name__ == "__main__":
    unittest.main()
