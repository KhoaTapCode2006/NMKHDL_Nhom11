import tempfile
import unittest
from pathlib import Path

from src.run_hierarchy import process_publication_hierarchy


class RunHierarchyTests(unittest.TestCase):
    def test_writes_bibtex_from_inlined_tex_in_elements_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            publication_dir = Path(temp_dir) / "paper"
            version_dir = publication_dir / "v1"
            elements_dir = publication_dir / "elements"
            version_dir.mkdir(parents=True)
            elements_dir.mkdir()
            (elements_dir / "full_inlined_v1.tex").write_text(
                r"\begin{document}\begin{thebibliography}{9}"
                r"\bibitem{key} Author. Title, 2020."
                r"\end{thebibliography}\end{document}",
                encoding="utf-8",
            )

            result = process_publication_hierarchy(publication_dir)

            bib_file = elements_dir / "references_v1.bib"
            self.assertEqual(result["tex_versions"], 1)
            self.assertTrue(bib_file.is_file())
            bibtex = bib_file.read_text(encoding="utf-8")
            self.assertIn("@misc{key,", bibtex)
            self.assertIn("year = {2020}", bibtex)
            self.assertIn("note = {Author. Title, 2020.}", bibtex)


if __name__ == "__main__":
    unittest.main()
