import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.parser.pipeline import ParsingPipeline
from src.run_parser import main


class RunParserTests(unittest.TestCase):
    @classmethod
    def _walk_tree(cls, node):
        if node is None:
            return
        yield node
        for child in node.get("children", []):
            yield from cls._walk_tree(child)

    def test_gathers_compilation_path_and_splits_included_elements(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "sourcesArxiv"
            version_dir = source_dir / "2401.12345" / "v1"
            chapters_dir = version_dir / "chapters"
            chapters_dir.mkdir(parents=True)

            (version_dir / "main.tex").write_text(
                "\\documentclass{article}\n"
                "\\begin{document}\n"
                "\\section{Introduction}\n"
                "Root sentence with citation \\cite{old_key}.\n"
                "\\input{chapters/details}\n"
                "% \\input{orphan}\n"
                "\\begin{align*}x &= 1\\end{align*}\n"
                "\\begin{thebibliography}{9}\n"
                "\\bibitem{old_key} Jane Doe. Stable Citation Matching Study. "
                "Journal of Tests, 2021.\n"
                "\\end{thebibliography}\n"
                "\\end{document}\n",
                encoding="utf-8",
            )
            (chapters_dir / "details.tex").write_text(
                "\\begin{itemize}\n"
                "\\item Included first sentence. Included second sentence.\n"
                "\\end{itemize}\n",
                encoding="utf-8",
            )
            version_two = source_dir / "2401.12345" / "v2"
            version_two.mkdir(parents=True)
            (version_two / "main.tex").write_text(
                "\\documentclass{article}\n"
                "\\begin{document}\n"
                "\\section{Introduction}\n"
                "Root sentence with citation \\cite{new_key}.\n"
                "\\begin{thebibliography}{9}\n"
                "\\bibitem{new_key} Jane Doe. Stable Citation Matching Study. "
                "Journal of Tests, 2021. doi:10.1234/stable\n"
                "\\end{thebibliography}\n"
                "\\end{document}\n",
                encoding="utf-8",
            )
            (version_dir / "orphan.tex").write_text(
                "This file is not part of the compilation path.",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "--source-dir",
                        str(source_dir),
                        "--no-clean-tex",
                        "--report-md",
                        str(root / "report.md"),
                        "--stats-json",
                        str(root / "stats.json"),
                    ]
                )

            elements_dir = source_dir / "2401.12345" / "elements"
            elements = json.loads(
                (elements_dir / "elements.json").read_text(encoding="utf-8")
            )
            hierarchy = json.loads(
                (elements_dir / "hierarchy.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (elements_dir / "compilation_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            bibtex = json.loads(
                (elements_dir / "bibtex.json").read_text(encoding="utf-8")
            )
            v1_tree = json.loads(
                (elements_dir / "hierarchy_tree_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            v2_tree = json.loads(
                (elements_dir / "hierarchy_tree_v2.json").read_text(
                    encoding="utf-8"
                )
            )
            kinds_and_content = {
                (element["kind"], element["content"]) for element in elements
            }

            self.assertIn(
                ("sentence", "Root sentence with citation \\cite{old_key}."),
                kinds_and_content,
            )
            self.assertIn(("sentence", "Included first sentence."), kinds_and_content)
            self.assertIn(("sentence", "Included second sentence."), kinds_and_content)
            self.assertTrue(any(kind == "item" for kind, _ in kinds_and_content))
            self.assertNotIn(
                "This file is not part of the compilation path.",
                {content for _, content in kinds_and_content},
            )
            elements_by_id = {element["element_id"]: element for element in elements}
            parents_by_child = {
                edge["element_id"]: edge["parent_id"] for edge in hierarchy
            }
            for sentence in (
                "Included first sentence.",
                "Included second sentence.",
            ):
                sentence_id = next(
                    element["element_id"]
                    for element in elements
                    if element["content"] == sentence
                )
                parent = elements_by_id[parents_by_child[sentence_id]]
                self.assertEqual(parent["kind"], "item")
            self.assertEqual(
                set(manifest["v1"]["compilation_files"]),
                {"main.tex", "details.tex"},
            )
            self.assertEqual(manifest["v1"]["orphan_files"], ["orphan.tex"])
            self.assertEqual(len(bibtex), 1)
            self.assertEqual(bibtex[0]["citation_key"], "old_key")
            self.assertEqual(bibtex[0]["doi"], "10.1234/stable")
            self.assertTrue(
                any(
                    element["content"].endswith(r"\cite{old_key}.")
                    for element in elements
                    if element["kind"] == "sentence"
                )
            )
            v2_nodes = list(self._walk_tree(v2_tree["root"]))
            self.assertTrue(
                any(
                    node["content"].endswith(r"\cite{old_key}.")
                    for node in v2_nodes
                    if node["kind"] == "sentence"
                )
            )
            formulas = [
                node["content"] for node in self._walk_tree(v1_tree["root"])
                if node["kind"] == "block_formula"
            ]
            self.assertTrue(any(r"\begin{equation}" in item for item in formulas))
            self.assertFalse(any(r"\begin{align*}" in item for item in formulas))
            self.assertTrue((elements_dir / "full_inlined_v1.tex").is_file())
            self.assertTrue((root / "report.md").is_file())
            self.assertTrue((root / "stats.json").is_file())

    def test_rerun_reuses_inlined_tex_after_source_cleanup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "sourcesArxiv"
            publication = source_dir / "paper"
            version_dir = publication / "v1"
            version_dir.mkdir(parents=True)
            (version_dir / "main.tex").write_text(
                "\\documentclass{article}\n"
                "\\begin{document}Persistent sentence.\\end{document}\n",
                encoding="utf-8",
            )
            pipeline = ParsingPipeline(source_dir=source_dir)

            first = pipeline.process_publication(publication)
            second = pipeline.process_publication(publication)
            elements = json.loads(
                (publication / "elements" / "elements.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertFalse((version_dir / "main.tex").exists())
            self.assertEqual(first["tex_versions"], 1)
            self.assertEqual(second["tex_versions"], 1)
            self.assertIn(
                "Persistent sentence.",
                {element["content"] for element in elements},
            )

    def test_records_empty_and_failed_versions_separately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "sourcesArxiv"
            publication = source_dir / "paper"
            empty_version = publication / "v1"
            failed_version = publication / "v2"
            empty_version.mkdir(parents=True)
            failed_version.mkdir()
            (failed_version / "main.tex").write_text(
                "\\documentclass{article}\n\\input{missing}\n",
                encoding="utf-8",
            )
            (publication / "elements").mkdir()
            (publication / "elements" / "full_inlined_v2.tex").write_text(
                "\\documentclass{article}\n"
                "\\begin{document}Stale successful content.\\end{document}\n",
                encoding="utf-8",
            )

            result = ParsingPipeline(
                source_dir=source_dir,
                clean_redundant_tex=False,
            ).process_publication(publication)
            manifest = json.loads(
                (publication / "elements" / "compilation_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            stats = json.loads(
                (publication / "elements" / "dedup_stats.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(result["empty_versions"], 1)
            self.assertEqual(result["failed_versions"], 1)
            self.assertEqual(manifest["v1"]["status"], "empty_version")
            self.assertEqual(manifest["v2"]["status"], "failed")
            self.assertIn("unresolved includes", manifest["v2"]["error"])
            self.assertEqual(stats["empty_versions"], 1)
            self.assertEqual(stats["failed_versions"], 1)


if __name__ == "__main__":
    unittest.main()
