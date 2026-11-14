import json
import re
import tempfile
import unittest
from pathlib import Path

from src.parser.dedup import ReferenceDeduplicator
from src.parser.hierarchy_builder import HierarchyBuilder
from src.run_hierarchy import process_publication_hierarchy


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
        reference = {
            "citation_key": "doe2022",
            "label": "",
            "content": (
                r"\bibinfo{author}{Jane Doe and John Smith}. "
                r"\bibinfo{title}{A {Nested}, Reliable Parser & Tests}. "
                r"\bibinfo{journal}{Journal of Data Tests}. "
                r"\bibinfo{year}{2022}. "
                r"\bibinfo{doi}{10.1234/example.2022}. "
                r"\url{https://example.org/paper}."
            ),
        }

        record = HierarchyBuilder.extract_reference_fields(reference, "2301.01234")
        bibtex = HierarchyBuilder.convert_references_to_bibtex([reference])[0]

        self.assertEqual(
            record,
            {
                "arxiv_id": "2301.01234",
                "citation_key": "doe2022",
                "entry_type": "misc",
                "title": "A Nested, Reliable Parser & Tests",
                "authors": ["Jane Doe", "John Smith"],
                "year": 2022,
                "doi": "10.1234/example.2022",
                "journal": "Journal of Data Tests",
                "url": "https://example.org/paper",
            },
        )
        self.assertIn("@misc{doe2022,", bibtex)
        self.assertIn("author = {Jane Doe and John Smith}", bibtex)
        self.assertIn(r"title = {A Nested, Reliable Parser \& Tests}", bibtex)
        self.assertIn("note =", bibtex)
        parsed = assert_valid_bibtex(self, bibtex)
        self.assertEqual(parsed[0][0], "misc")
        self.assertEqual(parsed[0][1], "doe2022")
        self.assertIn("title", parsed[0][2])

    def test_partial_and_unstructured_references_do_not_invent_fields(self):
        partial = {
            "citation_key": "partial",
            "label": "",
            "content": "Jane Doe. An Identifiable Study. 2021.",
        }
        unstructured = {
            "citation_key": "raw",
            "label": "",
            "content": "unstructured-reference-without-fields",
        }

        partial_record = HierarchyBuilder.extract_reference_fields(
            partial, "2301.01234"
        )
        raw_record = HierarchyBuilder.extract_reference_fields(
            unstructured, "2301.01234"
        )
        raw_bibtex = HierarchyBuilder.convert_references_to_bibtex([unstructured])[0]

        self.assertEqual(partial_record["title"], "An Identifiable Study")
        self.assertEqual(partial_record["authors"], ["Jane Doe"])
        self.assertEqual(partial_record["year"], 2021)
        self.assertIsNone(partial_record["doi"])
        self.assertIsNone(partial_record["url"])
        self.assertIsNone(partial_record["journal"])
        self.assertIsNone(raw_record["title"])
        self.assertEqual(raw_record["authors"], [])
        self.assertIn("note = {unstructured-reference-without-fields}", raw_bibtex)
        self.assertEqual(assert_valid_bibtex(self, raw_bibtex)[0][2]["note"],
                         "unstructured-reference-without-fields")

    def test_deduplicates_versions_merges_fields_and_remaps_citations(self):
        refs = {
            "v1": [
                {
                    "citation_key": "original_key",
                    "label": "",
                    "content": (
                        "Jane Doe. A Reliable Study. Journal of Tests, 2021."
                    ),
                },
                {
                    "citation_key": "duplicate_key",
                    "label": "",
                    "content": (
                        "Jane Doe. A Reliable Study. Journal of Tests, 2021."
                    ),
                },
            ],
            "v2": [
                {
                    "citation_key": "renamed_key",
                    "label": "",
                    "content": (
                        "Jane Doe. A Reliable Study. Journal of Tests, 2021. "
                        "doi:10.1234/reliable"
                    ),
                }
            ],
            "v3": [
                {
                    "citation_key": "renamed_key",
                    "label": "",
                    "content": (
                        r"\bibinfo{author}{Jane Doe}. "
                        r"\bibinfo{title}{A Reliable Study}. "
                        r"\bibinfo{doi}{10.1234/reliable}."
                    ),
                }
            ],
        }

        result = HierarchyBuilder.deduplicate_references_by_version(
            "2301.01234", refs
        )
        record = result["entries"][0]

        self.assertEqual(len(result["entries"]), 1)
        self.assertEqual(record["citation_key"], "original_key")
        self.assertEqual(record["arxiv_id"], "2301.01234")
        self.assertEqual(record["title"], "A Reliable Study")
        self.assertEqual(record["authors"], ["Jane Doe"])
        self.assertEqual(record["year"], 2021)
        self.assertEqual(record["doi"], "10.1234/reliable")
        self.assertEqual(record["journal"], "Journal of Tests")
        self.assertEqual(
            set(record),
            {
                "arxiv_id",
                "citation_key",
                "entry_type",
                "title",
                "authors",
                "year",
                "doi",
                "journal",
                "url",
            },
        )
        self.assertEqual(
            result["aliases_by_version"]["v2"]["renamed_key"],
            "original_key",
        )
        self.assertEqual(
            result["aliases_by_version"]["v1"]["duplicate_key"],
            "original_key",
        )
        self.assertEqual(
            HierarchyBuilder.remap_citation_keys(
                r"\citep[see][p. 4]{renamed_key, other} % \cite{renamed_key}",
                result["aliases_by_version"]["v2"],
            ),
            r"\citep[see][p. 4]{original_key, other} % \cite{renamed_key}",
        )
        for entries in result["bibtex_by_version"].values():
            self.assertEqual(len(entries), 1)
            self.assertIn("@misc{original_key,", entries[0])
            assert_valid_bibtex(self, entries[0])

    def test_publication_writes_unified_schema_and_versioned_canonical_bibtex(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            publication = Path(temp_dir) / "2301.01234"
            elements = publication / "elements"
            (publication / "v1").mkdir(parents=True)
            (publication / "v2").mkdir()
            elements.mkdir()
            (elements / "full_inlined_v1.tex").write_text(
                r"\begin{document}\cite{old_key} We use this work."
                r"\begin{thebibliography}{9}"
                r"\bibitem{old_key} Jane Doe. Stable Title. Journal of Tests, 2021."
                r"\end{thebibliography}\end{document}",
                encoding="utf-8",
            )
            (elements / "full_inlined_v2.tex").write_text(
                r"\begin{document}\cite{new_key} We use this work."
                r"\begin{thebibliography}{9}"
                r"\bibitem{new_key} Jane Doe. Stable Title. Journal of Tests, 2021. "
                r"doi:10.1234/stable"
                r"\end{thebibliography}\end{document}",
                encoding="utf-8",
            )

            result = process_publication_hierarchy(publication)
            table = json.loads((elements / "bibtex.json").read_text(encoding="utf-8"))
            version_elements = json.loads(
                (elements / "elements_v2.json").read_text(encoding="utf-8")
            )
            csv_text = (elements / "bibtex.csv").read_text(encoding="utf-8-sig")
            rewritten = (elements / "full_inlined_v2.tex").read_text(
                encoding="utf-8"
            )
            version_bib = (elements / "references_v2.bib").read_text(
                encoding="utf-8"
            )

            self.assertEqual(result["bibtex_entries_count"], 1)
            self.assertEqual(len(table), 1)
            self.assertEqual(table[0]["arxiv_id"], "2301.01234")
            self.assertEqual(table[0]["citation_key"], "old_key")
            self.assertEqual(table[0]["doi"], "10.1234/stable")
            self.assertIn("arxiv_id,citation_key,entry_type,title,authors,year,doi,journal,url",
                          csv_text)
            self.assertIn('"[""Jane Doe""]"', csv_text)
            self.assertNotIn("arxiv_id =", version_bib)
            self.assertIn("@misc{old_key,", version_bib)
            self.assertIn(r"\cite{old_key}", rewritten)
            self.assertTrue(
                any(
                    r"\cite{old_key}" in element["content"]
                    for element in version_elements
                )
            )
            self.assertFalse(
                any(
                    r"\cite{new_key}" in element["content"]
                    for element in version_elements
                )
            )
            self.assertEqual(
                set(table[0]),
                {
                    "arxiv_id",
                    "citation_key",
                    "entry_type",
                    "title",
                    "authors",
                    "year",
                    "doi",
                    "journal",
                    "url",
                },
            )
            assert_valid_bibtex(self, version_bib)


if __name__ == "__main__":
    unittest.main()
