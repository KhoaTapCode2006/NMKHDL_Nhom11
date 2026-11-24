import json
import re
import tempfile
import unittest
from pathlib import Path

from src.parser.dedup import ReferenceDeduplicator
from src.parser.pipeline import ParsingPipeline


def process_publication_hierarchy(pub_dir: Path):
    pub_dir = Path(pub_dir).resolve()
    return ParsingPipeline(
        source_dir=pub_dir.parent,
        clean_redundant_tex=False,
    ).process_publication(pub_dir)


def parse_bibtex(text):
    entries = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break

        declaration = re.match(
            r"@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,",
            text[position:],
        )
        if not declaration:
            raise AssertionError(f"Invalid BibTeX declaration at offset {position}")
        entry_type, citation_key = declaration.groups()
        position += declaration.end()
        fields = {}

        while position < len(text):
            while position < len(text) and (
                text[position].isspace() or text[position] == ","
            ):
                position += 1
            if position < len(text) and text[position] == "}":
                position += 1
                break

            field_match = re.match(r"[A-Za-z][A-Za-z0-9_-]*", text[position:])
            if not field_match:
                raise AssertionError(f"Invalid BibTeX field at offset {position}")
            field_name = field_match.group(0).lower()
            position += field_match.end()
            while position < len(text) and text[position].isspace():
                position += 1
            if position == len(text) or text[position] != "=":
                raise AssertionError(f"Missing '=' after BibTeX field {field_name}")
            position += 1
            while position < len(text) and text[position].isspace():
                position += 1
            if position == len(text) or text[position] != "{":
                raise AssertionError(f"BibTeX field {field_name} is not braced")

            depth = 1
            value_start = position + 1
            position += 1
            while position < len(text) and depth:
                character = text[position]
                backslashes = 0
                cursor = position - 1
                while cursor >= value_start and text[cursor] == "\\":
                    backslashes += 1
                    cursor -= 1
                if backslashes % 2 == 0:
                    if character == "{":
                        depth += 1
                    elif character == "}":
                        depth -= 1
                position += 1
            if depth:
                raise AssertionError(f"Unclosed BibTeX value for field {field_name}")
            fields[field_name] = text[value_start:position - 1]

            while position < len(text) and text[position].isspace():
                position += 1
            if position < len(text) and text[position] not in ",}":
                raise AssertionError(f"Invalid separator after BibTeX field {field_name}")

        entries.append((entry_type, citation_key, fields))
    return entries


def assert_valid_bibtex(test_case, text):
    entries = parse_bibtex(text)
    test_case.assertTrue(entries, text)
    return entries


class BibtexReferenceTests(unittest.TestCase):
    def test_comma_separated_bibitem_authors_match_bibtex_without_doi(self):
        bibtex_entries = ReferenceDeduplicator.parse_bib_content(
            "@article{vaswani2017, "
            "title={Attention Is All You Need}, "
            "author={Vaswani, A. and Shazeer, N.}, "
            "year={2017}}",
            "2301.01234",
            "v1",
        )
        bibitem_entries = ReferenceDeduplicator.parse_bibitem_entries(
            [
                r"\bibitem{transformer} A. Vaswani, N. Shazeer. "
                "Attention Is All You Need. 2017."
            ],
            "2301.01234",
            "v1",
        )

        self.assertEqual(
            bibitem_entries[0]["authors"],
            ["A. Vaswani", "N. Shazeer"],
        )
        self.assertEqual(bibitem_entries[0]["journal"], "")
        result = ReferenceDeduplicator.deduplicate_bibtex(
            "2301.01234",
            bibtex_entries + bibitem_entries,
        )
        self.assertEqual(len(result["bibtex"]), 1)
        self.assertEqual(
            result["key_mapping"]["transformer"],
            "vaswani2017",
        )

    def test_booklet_entry_type_has_expected_priority(self):
        result = ReferenceDeduplicator.unionize_cluster(
            "2301.01234",
            [
                {"entry_type": "misc"},
                {"entry_type": "booklet"},
            ],
            "booklet_key",
        )

        self.assertEqual(result["entry_type"], "booklet")

    def test_extracts_complete_reference_metadata_and_serializes_valid_bibtex(self):
        entry = ReferenceDeduplicator.parse_bibitem_entries(
            [
                r"\bibitem{doe2022} "
                r"\bibinfo{author}{Jane Doe and John Smith}. "
                r"\bibinfo{title}{A {Nested}, Reliable Parser & Tests}. "
                r"\bibinfo{journal}{Journal of Data Tests}. "
                r"\bibinfo{year}{2022}. "
                r"\bibinfo{doi}{10.1234/example.2022}. "
                r"\url{https://example.org/paper}."
            ],
            "2301.01234",
            "v1",
        )[0]

        self.assertEqual(entry["citation_key"], "doe2022")
        self.assertEqual(entry["title"], "A Nested, Reliable Parser & Tests")
        self.assertEqual(entry["authors"], ["Jane Doe", "John Smith"])
        self.assertEqual(entry["year"], 2022)
        self.assertEqual(entry["doi"], "10.1234/example.2022")
        self.assertEqual(entry["journal"], "Journal of Data Tests")
        self.assertEqual(entry["url"], "https://example.org/paper")

        bibtex = ReferenceDeduplicator.entry_to_bibtex(entry)
        self.assertIn("@article{doe2022,", bibtex)
        self.assertIn("author = {Jane Doe and John Smith}", bibtex)
        self.assertIn(r"title = {A Nested, Reliable Parser \& Tests}", bibtex)
        parsed = assert_valid_bibtex(self, bibtex)
        self.assertEqual(parsed[0][1], "doe2022")
        self.assertIn("title", parsed[0][2])

    def test_partial_and_unstructured_references_do_not_invent_fields(self):
        partial, raw = ReferenceDeduplicator.parse_bibitem_entries(
            [
                r"\bibitem{partial} Jane Doe. An Identifiable Study. 2021.",
                r"\bibitem{raw} unstructured-reference-without-fields",
            ],
            "2301.01234",
            "v1",
        )

        self.assertEqual(partial["title"], "An Identifiable Study")
        self.assertEqual(partial["authors"], ["Jane Doe"])
        self.assertEqual(partial["year"], 2021)
        self.assertEqual(partial["doi"], "")
        self.assertEqual(partial["url"], "")
        self.assertEqual(partial["journal"], "")
        self.assertEqual(raw["title"], "")
        self.assertEqual(raw["authors"], [])
        raw_bibtex = ReferenceDeduplicator.entry_to_bibtex(raw)
        self.assertIn("@misc{raw,", raw_bibtex)
        self.assertEqual(
            assert_valid_bibtex(self, raw_bibtex)[0][2]["note"],
            "unstructured-reference-without-fields",
        )

    def test_deduplicates_versions_merges_fields_and_remaps_citations(self):
        parse = ReferenceDeduplicator.parse_bibitem_entries
        entries = (
            parse(
                [
                    r"\bibitem{original_key} Jane Doe. A Reliable Study of Parsers. "
                    "Journal of Tests, 2021.",
                    r"\bibitem{duplicate_key} Jane Doe. A Reliable Study of Parsers. "
                    "Journal of Tests, 2021.",
                ],
                "2301.01234",
                "v1",
            )
            + parse(
                [
                    r"\bibitem{renamed_key} Jane Doe. A Reliable Study of Parsers. "
                    "Journal of Tests, 2021. doi:10.1234/reliable",
                ],
                "2301.01234",
                "v2",
            )
            + parse(
                [
                    r"\bibitem{renamed_key} \bibinfo{author}{Jane Doe}. "
                    r"\bibinfo{title}{A Reliable Study of Parsers}. "
                    r"\bibinfo{doi}{10.1234/reliable}.",
                ],
                "2301.01234",
                "v3",
            )
        )

        result = ReferenceDeduplicator.deduplicate_bibtex("2301.01234", entries)
        self.assertEqual(len(result["bibtex"]), 1)
        record = result["bibtex"][0]
        self.assertEqual(record["citation_key"], "original_key")
        self.assertEqual(record["title"], "A Reliable Study of Parsers")
        self.assertEqual(record["authors"], ["Jane Doe"])
        self.assertEqual(record["year"], 2021)
        self.assertEqual(record["doi"], "10.1234/reliable")
        self.assertEqual(record["journal"], "Journal of Tests")
        self.assertEqual(
            set(record),
            {"arxiv_id", "citation_key", "entry_type", "title", "authors",
             "year", "doi", "journal", "url"},
        )
        by_version = result["key_mapping_by_version"]
        self.assertEqual(by_version["v1"]["duplicate_key"], "original_key")
        self.assertEqual(by_version["v2"]["renamed_key"], "original_key")
        rewritten, _, _ = ReferenceDeduplicator.rewrite_citations_in_text(
            r"\citep[see][p. 4]{renamed_key, other}", by_version["v2"]
        )
        self.assertEqual(rewritten, r"\citep[see][p. 4]{original_key, other}")

    def test_reused_key_for_another_paper_is_remapped_per_version(self):
        parse = ReferenceDeduplicator.parse_bibitem_entries
        entries = parse(
            [r"\bibitem{ref1} Alice Brown. Graph Neural Networks for Molecules. 2019."],
            "p", "v1",
        ) + parse(
            [r"\bibitem{ref1} Carol White. Quantum Error Correction Codes. 2023."],
            "p", "v2",
        )

        result = ReferenceDeduplicator.deduplicate_bibtex("p", entries)
        by_version = result["key_mapping_by_version"]

        self.assertEqual(len(result["bibtex"]), 2)
        self.assertEqual(by_version["v1"]["ref1"], "ref1")
        self.assertEqual(by_version["v2"]["ref1"], "ref1_v2")
        rewritten, _, _ = ReferenceDeduplicator.rewrite_citations_in_text(
            r"\cite{ref1}", by_version["v2"]
        )
        self.assertEqual(rewritten, r"\cite{ref1_v2}")

    def test_publication_writes_unified_schema_and_canonical_bibtex_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            publication = Path(temp_dir) / "2301.01234"
            elements = publication / "elements"
            (publication / "v1").mkdir(parents=True)
            (publication / "v2").mkdir()
            elements.mkdir()
            (elements / "full_inlined_v1.tex").write_text(
                r"\begin{document}\cite{old_key} We use this work."
                r"\begin{thebibliography}{9}"
                r"\bibitem{old_key} Jane Doe. Stable Title of the Study. Journal of Tests, 2021."
                r"\end{thebibliography}\end{document}",
                encoding="utf-8",
            )
            (elements / "full_inlined_v2.tex").write_text(
                r"\begin{document}\cite{new_key} We use this work."
                r"\begin{thebibliography}{9}"
                r"\bibitem{new_key} Jane Doe. Stable Title of the Study. Journal of Tests, 2021. "
                r"doi:10.1234/stable"
                r"\end{thebibliography}\end{document}",
                encoding="utf-8",
            )

            result = process_publication_hierarchy(publication)
            bibtex_dir = publication / "bibtex"
            table = json.loads((bibtex_dir / "bibtex.json").read_text(encoding="utf-8"))
            version_elements = json.loads(
                (elements / "elements_v2.json").read_text(encoding="utf-8")
            )
            csv_text = (bibtex_dir / "bibtex.csv").read_text(encoding="utf-8-sig")
            bib_file = (bibtex_dir / "bibtex.bib").read_text(encoding="utf-8")

            self.assertEqual(result["bibtex_entries_count"], 1)
            self.assertEqual(len(table), 1)
            self.assertEqual(table[0]["citation_key"], "old_key")
            self.assertEqual(table[0]["doi"], "10.1234/stable")
            self.assertIn(
                "arxiv_id,citation_key,entry_type,title,authors,year,doi,journal,url",
                csv_text,
            )
            self.assertFalse((elements / "bibtex.json").exists())
            self.assertIn("{old_key,", bib_file)
            self.assertNotIn("arxiv_id =", bib_file)
            assert_valid_bibtex(self, bib_file)
            self.assertTrue(
                any(r"\cite{old_key}" in e["content"] for e in version_elements)
            )
            self.assertFalse(
                any(r"\cite{new_key}" in e["content"] for e in version_elements)
            )

    def test_bibliography_command_selects_bib_file_and_uncited_entries_are_dropped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            publication = Path(temp_dir) / "paper"
            version = publication / "v1"
            version.mkdir(parents=True)
            (version / "main.tex").write_text(
                "\\documentclass{article}\n\\begin{document}\n"
                "We follow \\cite{used} closely.\n"
                "\\bibliography{refs}\n\\end{document}\n",
                encoding="utf-8",
            )
            (version / "refs.bib").write_text(
                "@article{used, title={A Paper That Is Cited Here}, year={2020}}\n"
                "@article{unused, title={A Paper Nobody Cites At All}, year={2021}}\n",
                encoding="utf-8",
            )
            (version / "template-sample.bib").write_text(
                "@article{template, title={Template Example Entry Title}, year={2000}}\n",
                encoding="utf-8",
            )

            result = process_publication_hierarchy(publication)
            table = json.loads(
                (publication / "bibtex" / "bibtex.json").read_text(encoding="utf-8")
            )

        self.assertEqual([row["citation_key"] for row in table], ["used"])
        self.assertEqual(result["bibtex_stats"]["uncited_bib_entries_dropped"], 1)


if __name__ == "__main__":
    unittest.main()
