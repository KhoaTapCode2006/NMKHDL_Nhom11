import unittest

from src.parser.hierarchy_builder import HierarchyBuilder


class HierarchyBuilderTests(unittest.TestCase):
    def test_sentences_split_after_period_even_when_next_sentence_is_lowercase(self):
        self.assertEqual(
            HierarchyBuilder.split_sentences_safe("First sentence. second sentence."),
            ["First sentence.", "second sentence."],
        )

    def test_sentence_split_protects_abbreviations_commands_urls_and_decimals(self):
        text = (
            "Prior work by Smith et al. supports 3.14. "
            r"See Fig. 1 and \cite{smith.2020}. "
            "Visit https://example.org/page. Next sentence."
        )
        self.assertEqual(
            HierarchyBuilder.split_sentences_safe(text),
            [
                "Prior work by Smith et al. supports 3.14.",
                r"See Fig. 1 and \cite{smith.2020}.",
                "Visit https://example.org/page.",
                "Next sentence.",
            ],
        )

    def test_parse_version_builds_heading_edges_and_excludes_references(self):
        tex = r"""
        \begin{document}
        \section{Introduction}
        Opening sentence. next sentence.
        \subsection{Details}
        Supporting text.
        \section{References}
        This must not enter the hierarchy.
        \subsection{Reference notes}
        Nor should this.
        \section*{Acknowledgements}
        Thanks to the reviewers.
        \end{document}
        """
        elements, edges, _ = HierarchyBuilder.parse_version("paper", "v2", tex)

        by_id = {element["raw_id"]: element for element in elements}
        parent_by_child = {
            edge["element_raw_id"]: edge["parent_raw_id"] for edge in edges
        }
        headings = {
            element["content"]: element["raw_id"]
            for element in elements
            if element["kind"] in {"section", "subsection"}
        }
        self.assertEqual(by_id["root2"]["kind"], "document")
        self.assertNotIn("References", headings)
        self.assertNotIn("Reference notes", headings)
        self.assertEqual(parent_by_child[headings["Introduction"]], "root2")
        self.assertEqual(
            parent_by_child[headings["Details"]], headings["Introduction"]
        )
        self.assertEqual(parent_by_child[headings["Acknowledgements"]], "root2")
        self.assertEqual(
            [element["content"] for element in elements if element["kind"] == "sentence"],
            [
                "Opening sentence.",
                "next sentence.",
                "Supporting text.",
                "Thanks to the reviewers.",
            ],
        )

    def test_bibliography_capture_does_not_include_environment_closer_as_item_text(self):
        _, _, preserved = HierarchyBuilder.parse_version(
            "paper",
            "v1",
            r"""
            \begin{document}
            \begin{thebibliography}{9}
            \bibitem{key} A reference entry.
            \end{thebibliography}
            \end{document}
            """,
        )
        self.assertEqual(
            preserved["bibitem_entries"],
            [r"\bibitem{key} A reference entry."],
        )

    def test_extracts_and_converts_bibitems_from_inlined_tex(self):
        tex = r"""
        \begin{document}
        \begin{thebibliography}{9}
        \bibitem[Smith et al.(2021)]{smith2021}
        Smith et al. A useful paper. Journal, 2021.
        DOI: 10.1234/example.2021. https://example.org/paper.
        \bibitem{other}
        Another reference, 1999.
        \end{thebibliography}
        % \bibitem{commented} This must be ignored.
        \end{document}
        """

        references = HierarchyBuilder.extract_bibitems(tex)
        bibtex = HierarchyBuilder.convert_references_to_bibtex(references)

        self.assertEqual(
            [reference["citation_key"] for reference in references],
            ["smith2021", "other"],
        )
        self.assertEqual(references[0]["label"], "Smith et al.(2021)")
        self.assertIn("@misc{smith2021,", bibtex[0])
        self.assertIn("year = {2021}", bibtex[0])
        self.assertIn("doi = {10.1234/example.2021}", bibtex[0])
        self.assertIn("url = {https://example.org/paper}", bibtex[0])
        self.assertIn(
            "note = {[Smith et al.(2021)] Smith et al. A useful paper. Journal, 2021. "
            "DOI: 10.1234/example.2021. https://example.org/paper.}",
            bibtex[0],
        )
        self.assertIn("@misc{other,", bibtex[1])
        self.assertNotIn("commented", "\n".join(bibtex))

    def test_list_items_have_sentence_children_even_when_they_contain_one_sentence(self):
        elements, edges, _ = HierarchyBuilder.parse_version(
            "paper",
            "v1",
            r"""
            \begin{document}
            \begin{itemize}
            \item First list point.
            \item Second list point.
            \end{itemize}
            \end{document}
            """,
        )
        content_by_id = {element["raw_id"]: element["content"] for element in elements}
        kind_by_id = {element["raw_id"]: element["kind"] for element in elements}
        children_by_parent = {}
        for edge in edges:
            children_by_parent.setdefault(edge["parent_raw_id"], []).append(
                edge["element_raw_id"]
            )

        item_ids = [
            element["raw_id"] for element in elements if element["kind"] == "item"
        ]
        self.assertEqual(len(item_ids), 2)
        for item_id in item_ids:
            child_ids = children_by_parent.get(item_id, [])
            self.assertTrue(child_ids)
            self.assertTrue(all(kind_by_id[child_id] == "sentence" for child_id in child_ids))
        self.assertEqual(
            {
                content_by_id[child_id]
                for item_id in item_ids
                for child_id in children_by_parent.get(item_id, [])
            },
            {"First list point.", "Second list point."},
        )

    def test_formulas_figures_output_ids_and_empty_version(self):
        elements, edges, _ = HierarchyBuilder.parse_version(
            "paper",
            "v1",
            r"""
            \begin{document}
            \section{Content}
            \begin{equation}x = 1\end{equation}
            \begin{table}\caption{A table}\end{table}
            \end{document}
            """,
        )
        output = HierarchyBuilder.build_version_hierarchy_outputs(
            "paper", "v1", elements, edges
        )
        self.assertEqual(output["tree"]["root"]["element_id"], "paper#root1")
        self.assertEqual(output["tree"]["edges_count"], len(output["hierarchy"]))
        self.assertTrue(all(edge["parent_id"] for edge in output["hierarchy"]))
        self.assertIn("block_formula", {element["kind"] for element in elements})
        self.assertIn("figure", {element["kind"] for element in elements})

        empty = HierarchyBuilder.build_version_hierarchy_outputs(
            "paper", "v2", [], [], is_empty=True
        )
        self.assertTrue(empty["is_empty"])
        self.assertEqual(empty["elements"], [])
        self.assertIsNone(empty["tree"]["root"])

    def test_nested_tree_does_not_create_unknown_nodes_for_dangling_edges(self):
        result = HierarchyBuilder.build_nested_tree(
            "paper",
            "v1",
            [
                {
                    "element_id": "paper#root1",
                    "kind": "document",
                    "content": "root",
                }
            ],
            [
                {
                    "version": 1,
                    "parent_id": "paper#root1",
                    "element_id": "missing-element",
                }
            ],
        )
        self.assertEqual(result["root"]["children"], [])


if __name__ == "__main__":
    unittest.main()
