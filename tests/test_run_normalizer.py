import json
import tempfile
import unittest
from pathlib import Path

from src.run_normalizer import normalize_publications


class RunNormalizerTests(unittest.TestCase):
    def test_normalizes_version_local_trees_in_place(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            version_dir = source_dir / "paper" / "v1"
            version_dir.mkdir(parents=True)
            tree_file = version_dir / "hierarchy_tree.json"
            tree_file.write_text(
                json.dumps(
                    {
                        "root": {
                            "element_id": "paper#root1",
                            "kind": "document",
                            "content": r"\centering Document",
                            "children": [],
                        }
                    }
                ),
                encoding="utf-8",
            )
            (source_dir / "paper" / "elements").mkdir()

            outputs = normalize_publications(source_dir)

            self.assertEqual(outputs, [tree_file])
            self.assertFalse((version_dir / "hierarchy_tree.normalized.json").exists())
            normalized = json.loads(tree_file.read_text(encoding="utf-8"))
            self.assertEqual(normalized["root"]["content"], "Document")

    def test_uses_elements_tree_when_version_directories_were_removed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            elements_dir = source_dir / "paper" / "elements"
            elements_dir.mkdir(parents=True)
            tree_file = elements_dir / "hierarchy_tree_v2.json"
            tree_file.write_text(
                json.dumps({"root": None, "status": "empty_version"}),
                encoding="utf-8",
            )

            outputs = normalize_publications(source_dir)

            self.assertEqual(
                outputs,
                [tree_file],
            )

    def test_in_place_normalization_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            version_dir = source_dir / "paper" / "v1"
            version_dir.mkdir(parents=True)
            tree_file = version_dir / "hierarchy_tree.json"
            tree_file.write_text(
                json.dumps(
                    {
                        "root": {
                            "element_id": "paper#root1",
                            "kind": "document",
                            "content": r"\centering Document",
                            "children": [],
                        }
                    }
                ),
                encoding="utf-8",
            )

            normalize_publications(source_dir)
            normalized_once = tree_file.read_text(encoding="utf-8")
            normalize_publications(source_dir)
            self.assertEqual(tree_file.read_text(encoding="utf-8"), normalized_once)


if __name__ == "__main__":
    unittest.main()
