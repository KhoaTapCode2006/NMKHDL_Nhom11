import tempfile
import unittest
from pathlib import Path

from src.parser.gatherer import MultiFileGatherer


class MultiFileGathererTests(unittest.TestCase):
    def test_follows_transitive_includes_and_ignores_comments_and_orphans(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            version_dir = Path(temp_dir)
            (version_dir / "main.tex").write_text(
                "\\documentclass{article}\n"
                "\\begin{document}\n"
                "\\input{chapters/intro} % \\input{commented}\n"
                "% \\include{also_commented}\n"
                "\\end{document}\n",
                encoding="utf-8",
            )
            (version_dir / "chapters").mkdir()
            (version_dir / "chapters" / "intro.tex").write_text(
                "\\include{details}\n", encoding="utf-8"
            )
            (version_dir / "chapters" / "details.tex").write_text(
                "Transitive content.", encoding="utf-8"
            )
            (version_dir / "commented.tex").write_text(
                "Commented content.", encoding="utf-8"
            )
            (version_dir / "also_commented.tex").write_text(
                "Also commented content.", encoding="utf-8"
            )
            (version_dir / "unused.tex").write_text(
                "Unused content.", encoding="utf-8"
            )

            result = MultiFileGatherer.gather(version_dir)

            self.assertTrue(result["success"])
            self.assertEqual(result["root_file"], (version_dir / "main.tex").resolve())
            self.assertEqual(
                {path.name for path in result["compilation_files"]},
                {"main.tex", "intro.tex", "details.tex"},
            )
            self.assertEqual(
                {path.name for path in result["orphan_files"]},
                {"commented.tex", "also_commented.tex", "unused.tex"},
            )
            self.assertIn("Transitive content.", result["inlined_content"])
            self.assertNotIn("Commented content.", result["inlined_content"])
            self.assertNotIn("Unused content.", result["inlined_content"])

    def test_repeated_include_is_inlined_at_each_occurrence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            version_dir = Path(temp_dir)
            (version_dir / "main.tex").write_text(
                "\\documentclass{article}\n"
                "\\begin{document}\n"
                "\\input{shared}\n"
                "\\input{shared}\n"
                "\\end{document}\n",
                encoding="utf-8",
            )
            (version_dir / "shared.tex").write_text(
                "Repeated content.", encoding="utf-8"
            )

            result = MultiFileGatherer.gather(version_dir)

            self.assertTrue(result["success"])
            self.assertEqual(result["inlined_content"].count("Repeated content."), 2)

    def test_unbraced_input_is_followed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            version_dir = Path(temp_dir)
            (version_dir / "main.tex").write_text(
                "\\documentclass{article}\n"
                "\\begin{document}\n"
                "\\input chapter\n"
                "\\end{document}\n",
                encoding="utf-8",
            )
            (version_dir / "chapter.tex").write_text(
                "Included content.", encoding="utf-8"
            )

            result = MultiFileGatherer.gather(version_dir)

            self.assertIn("Included content.", result["inlined_content"])

    def test_includeonly_excludes_unselected_include_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            version_dir = Path(temp_dir)
            (version_dir / "main.tex").write_text(
                "\\documentclass{article}\n"
                "\\includeonly{selected}\n"
                "\\begin{document}\n"
                "\\include{selected}\n"
                "\\include{not_selected}\n"
                "\\end{document}\n",
                encoding="utf-8",
            )
            (version_dir / "selected.tex").write_text(
                "Selected content.", encoding="utf-8"
            )
            (version_dir / "not_selected.tex").write_text(
                "Excluded content.", encoding="utf-8"
            )

            result = MultiFileGatherer.gather(version_dir)

            self.assertIn("Selected content.", result["inlined_content"])
            self.assertNotIn("Excluded content.", result["inlined_content"])

    def test_arbitrary_single_tex_file_is_not_accepted_as_document_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            version_dir = Path(temp_dir)
            (version_dir / "fragment.tex").write_text(
                "A fragment, not a document.", encoding="utf-8"
            )

            self.assertIsNone(MultiFileGatherer.find_root_file(version_dir))

    def test_missing_include_is_not_reported_as_successful_gather(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            version_dir = Path(temp_dir)
            (version_dir / "main.tex").write_text(
                "\\documentclass{article}\n"
                "\\begin{document}\n"
                "\\input{missing}\n"
                "\\end{document}\n",
                encoding="utf-8",
            )

            result = MultiFileGatherer.gather(version_dir)

            self.assertFalse(result["success"])
            self.assertIsNotNone(result["error"])

    def test_cycle_is_not_reported_as_successful_gather(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            version_dir = Path(temp_dir)
            (version_dir / "main.tex").write_text(
                "\\documentclass{article}\n\\input{part}\n", encoding="utf-8"
            )
            (version_dir / "part.tex").write_text(
                "\\input{main}\n", encoding="utf-8"
            )

            result = MultiFileGatherer.gather(version_dir)

            self.assertFalse(result["success"])
            self.assertIsNotNone(result["error"])
            self.assertIn("Cycle detected", result["inlined_content"])
            self.assertEqual(len(result["compilation_files"]), 2)

    def test_clean_redundant_files_keeps_root_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            version_dir = Path(temp_dir)
            root = version_dir / "main.tex"
            included = version_dir / "part.tex"
            orphan = version_dir / "unused.tex"
            for path in (root, included, orphan):
                path.write_text(path.name, encoding="utf-8")

            deleted = MultiFileGatherer.clean_redundant_files(
                version_dir,
                compilation_files=[root, included],
                orphan_files=[orphan],
                keep_root=True,
            )

            self.assertEqual(set(deleted), {included, orphan})
            self.assertTrue(root.exists())
            self.assertFalse(included.exists())
            self.assertFalse(orphan.exists())


if __name__ == "__main__":
    unittest.main()
