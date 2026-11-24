import json
import tempfile
import unittest
from pathlib import Path

from src.parser.pipeline import ParsingPipeline


def _process_publication_hierarchy(pub_dir: Path):
    pub_dir = Path(pub_dir).resolve()
    return ParsingPipeline(
        source_dir=pub_dir.parent,
        clean_redundant_tex=False,
    ).process_publication(pub_dir)


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

            result = _process_publication_hierarchy(publication_dir)

            bibtex_dir = publication_dir / "bibtex"
            bib_file = bibtex_dir / "bibtex.bib"
            table = json.loads((bibtex_dir / "bibtex.json").read_text(encoding="utf-8"))
            self.assertEqual(result["tex_versions"], 1)
            self.assertTrue(bib_file.is_file())
            bibtex = bib_file.read_text(encoding="utf-8")
            self.assertIn("{key,", bibtex)
            self.assertIn("year = {2020}", bibtex)
            self.assertEqual([row["citation_key"] for row in table], ["key"])
            self.assertEqual(table[0]["year"], 2020)


if __name__ == "__main__":
    unittest.main()
