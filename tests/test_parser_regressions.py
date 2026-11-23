"""Regression tests for the Milestone 2 review fixes."""

import json
import tempfile
import tracemalloc
import unittest
from collections import Counter
from pathlib import Path

from src.parser.benchmark import ParserBenchmark
from src.parser.gatherer import MultiFileGatherer
from src.parser.hierarchy_builder import HierarchyBuilder
from src.parser.hierarchy_normalizer import HierarchyNormalizer
from src.parser.pipeline import ParsingPipeline, parse_args


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def kinds_by_content(elements):
    return {e["content"]: e["kind"] for e in elements}


class GatheringAndPipelineTests(unittest.TestCase):
    def test_non_tex_input_is_skipped_without_failing_the_gather(self):
        with tempfile.TemporaryDirectory() as tmp:
            version = Path(tmp) / "v1"
            write(version / "main.tex",
                  "\\documentclass{article}\n\\begin{document}\nText.\n"
                  "\\input{plot.tikz}\n\\end{document}\n")
            result = MultiFileGatherer.gather(version)

        self.assertTrue(result["success"])
        self.assertEqual(result["skipped_non_tex_includes"], ["plot.tikz"])
        self.assertEqual(result["unresolved_includes"], [])

    def test_commented_include_is_kept_as_plain_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            version = Path(tmp) / "v1"
            write(version / "main.tex",
                  "\\documentclass{article}\n\\begin{document}\n"
                  "Keep me. % \\input{orphan}\n\\end{document}\n")
            write(version / "orphan.tex", "ORPHAN")
            result = MultiFileGatherer.gather(version)

        self.assertIn("% \\input{orphan}", result["inlined_content"])
        self.assertNotIn("ORPHAN", result["inlined_content"])

    def test_missing_included_tex_file_gives_a_partial_version_not_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src"
            publication = source / "paper"
            write(publication / "v1" / "main.tex",
                  "\\documentclass{article}\n\\begin{document}\n"
                  "\\section{Intro}\nThis sentence survives.\n"
                  "\\input{sections/missing}\n\\end{document}\n")

            result = ParsingPipeline(source_dir=source).process_publication(publication)
            manifest = json.loads(
                (publication / "elements" / "compilation_manifest.json").read_text("utf-8")
            )
            elements = json.loads((publication / "elements" / "elements.json").read_text("utf-8"))

        self.assertEqual(manifest["v1"]["status"], "partial")
        self.assertEqual(manifest["v1"]["unresolved_includes"], ["sections/missing"])
        self.assertEqual(result["failed_versions"], 0)
        self.assertEqual(result["partial_versions"], 1)
        self.assertIn("This sentence survives.", {e["content"] for e in elements})

    def test_original_tex_files_are_kept_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src"
            main = write(source / "paper" / "v1" / "main.tex",
                         "\\documentclass{article}\n\\begin{document}Hi there.\\end{document}\n")
            ParsingPipeline(source_dir=source).process_publication(source / "paper")
            self.assertTrue(main.exists())
        self.assertFalse(parse_args(["--source-dir", "x"]).clean_tex)
        self.assertTrue(parse_args(["--source-dir", "x", "--clean-tex"]).clean_tex)

    def test_benchmark_does_not_trace_memory_unless_asked(self):
        benchmark = ParserBenchmark()
        benchmark.start()
        tracing = tracemalloc.is_tracing()
        benchmark.stop()
        summary = benchmark.generate_summary()

        self.assertFalse(tracing)
        self.assertFalse(summary["performance"]["tracemalloc_enabled"])
        self.assertGreaterEqual(summary["performance"]["rss_samples"], 1)


class HierarchyTests(unittest.TestCase):
    def parse(self, body: str, preamble: str = ""):
        tex = f"\\documentclass{{article}}\n{preamble}\n\\begin{{document}}\n{body}\n\\end{{document}}"
        elements, hierarchy, _refs = HierarchyBuilder.parse_version("p", "v1", tex)
        return elements, hierarchy

    def test_structural_macros_are_expanded(self):
        elements, _ = self.parse(
            "\\sect{Method}\nWe solve it.\n\\beq x = y \\eeq",
            preamble=(
                "\\newcommand{\\beq}{\\begin{equation}}\n"
                "\\def\\eeq{\\end{equation}}\n"
                "\\newcommand{\\sect}[1]{\\section{#1}}\n"
                "\\newcommand{\\R}{\\mathbb{R}}"
            ),
        )
        kinds = kinds_by_content(elements)

        self.assertEqual(kinds.get("Method"), "section")
        self.assertIn("block_formula", kinds.values())
        self.assertFalse(any("\\beq" in e["content"] for e in elements))

    def test_no_at_the_end_of_a_sentence_still_splits(self):
        sentences = HierarchyBuilder.split_sentences_safe(
            "There is no. Another sentence follows. See No. 3 and pp. 12 here. "
            "The delay is 10 ms. Dr. Smith agrees."
        )
        self.assertEqual(
            sentences,
            [
                "There is no.",
                "Another sentence follows.",
                "See No. 3 and pp. 12 here.",
                "The delay is 10 ms.",
                "Dr. Smith agrees.",
            ],
        )

    def test_heading_inside_an_item_becomes_a_heading_node(self):
        elements, hierarchy = self.parse(
            "\\begin{itemize}\n\\item \\paragraph{Setup} We train it.\n\\end{itemize}"
        )
        by_id = {e["raw_id"]: e for e in elements}
        parent = {h["element_raw_id"]: h["parent_raw_id"] for h in hierarchy}
        heading = next(e for e in elements if e["content"] == "Setup")

        self.assertEqual(heading["kind"], "paragraph")
        self.assertEqual(by_id[parent[heading["raw_id"]]]["kind"], "item")

    def test_bare_acknowledgments_command_becomes_a_section(self):
        elements, _ = self.parse(
            "\\section{Intro}\nText here.\n\\acknowledgments\nWe thank the reviewers."
        )
        kinds = kinds_by_content(elements)
        self.assertEqual(kinds.get("Acknowledgements"), "section")

    def test_repeated_sentence_in_one_version_keeps_one_parent_per_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src"
            write(source / "paper" / "v1" / "main.tex",
                  "\\documentclass{article}\n\\begin{document}\n"
                  "\\section{A}\nSame sentence here.\n"
                  "\\section{B}\nSame sentence here.\n\\end{document}\n")
            ParsingPipeline(source_dir=source).process_publication(source / "paper")
            hierarchy = json.loads(
                (source / "paper" / "elements" / "hierarchy.json").read_text("utf-8")
            )

        parents_per_node = Counter((h["version"], h["element_id"]) for h in hierarchy)
        self.assertEqual(max(parents_per_node.values()), 1)


class NormalizationTests(unittest.TestCase):
    def test_align_becomes_equation_with_aligned_inside(self):
        result = HierarchyNormalizer.normalize_content(
            "\\begin{align} a &= b \\\\ c &= d \\end{align}", kind="block_formula"
        )
        self.assertTrue(result.startswith("\\begin{equation}\n\\begin{aligned}"))
        self.assertIn("a &= b \\\\ c &= d", result)
        self.assertTrue(result.endswith("\\end{aligned}\n\\end{equation}"))

    def test_eqnarray_keeps_a_single_alignment_point(self):
        result = HierarchyNormalizer.normalize_content(
            "\\begin{eqnarray} a &=& b \\\\ c &=& d \\end{eqnarray}", kind="block_formula"
        )
        self.assertIn("a &= b", result)
        self.assertNotIn("&=&", result)

    def test_single_line_display_stays_a_plain_equation(self):
        result = HierarchyNormalizer.normalize_content(
            "\\begin{gather} x = 1 \\end{gather}", kind="block_formula"
        )
        self.assertEqual(result, "\\begin{equation}\nx = 1\n\\end{equation}")

    def test_more_layout_commands_and_wrappers_are_removed(self):
        result = HierarchyNormalizer.normalize_content(
            "\\begin{table}[!t]\\FloatBarrier\\resizebox{\\linewidth}{!}{TAB}"
            "\\cmidrule(lr){2-3}\\addlinespace\\vskip 2pt\\end{table}",
            kind="figure",
        )
        self.assertEqual(result, "\\begin{table}TAB\\end{table}")

    def test_math_environment_and_ensuremath_become_dollars(self):
        result = HierarchyNormalizer.normalize_content(
            "Let \\begin{math}x\\end{math} and \\ensuremath{y^2} but keep $\\ensuremath{z}$.",
            kind="sentence",
        )
        self.assertEqual(result, "Let $x$ and $y^2$ but keep $\\ensuremath{z}$.")


if __name__ == "__main__":
    unittest.main()
