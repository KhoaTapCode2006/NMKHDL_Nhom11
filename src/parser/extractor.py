"""
LaTeX Hierarchy Extractor & Content Normalizer.
Parses LaTeX documents into a tree structure (Document -> Section -> Leaves: Sentence, Formula, Figure, Item)
and performs LaTeX cleanup & math normalization according to Milestone 2 specifications.
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, List, Optional, Tuple, Any

from .hierarchy_normalizer import HierarchyNormalizer


class HierarchyExtractor:
    """Extracts hierarchical elements from LaTeX text and standardizes content."""

    # Heading hierarchy definitions
    HEADING_LEVELS = {
        "chapter": 1,
        "section": 2,
        "subsection": 3,
        "subsubsection": 4,
        "paragraph": 5,
    }

    # Commands that should be removed during text cleanup
    CLEANUP_COMMANDS = HierarchyNormalizer.CLEANUP_COMMANDS

    ABBREVIATIONS = [
        r"\be\.g\.",
        r"\bi\.e\.",
        r"\bet al\.",
        r"\bFig\.",
        r"\bFigs\.",
        r"\bTab\.",
        r"\bTabs\.",
        r"\bEq\.",
        r"\bEqs\.",
        r"\bSec\.",
        r"\bSecs\.",
        r"\bRef\.",
        r"\bRefs\.",
        r"\bvs\.",
        r"\bcf\.",
        r"\bapprox\.",
        r"\bno\.",
        r"\bvol\.",
        r"\bpp\.",
        r"\bp\.",
        r"\bed\.",
        r"\beds\.",
        r"\bal\.",
        r"\bDr\.",
        r"\bProf\.",
        r"\bMr\.",
        r"\bMrs\.",
        r"\bMs\.",
        r"\bUniv\.",
    ]

    @classmethod
    def standardize_math(cls, text: str) -> str:
        """Compatibility delegate; normalization rules live in HierarchyNormalizer."""
        return HierarchyNormalizer.standardize_math(text)

    @classmethod
    def clean_latex(cls, text: str) -> str:
        """Compatibility delegate; cleanup rules live in HierarchyNormalizer."""
        return HierarchyNormalizer.clean_latex(text)

    @classmethod
    def strip_references_section(cls, text: str) -> str:
        """Removes the bibliography / references environment from full text."""
        # Remove \begin{thebibliography} ... \end{thebibliography}
        text = re.sub(
            r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
            "",
            text,
            flags=re.DOTALL,
        )
        # Remove \bibliography{...}
        text = re.sub(r"\\bibliography\{[^}]*\}", "", text)
        return text

    @classmethod
    def extract_document_body(cls, text: str) -> str:
        """Extracts text between \\begin{document} and \\end{document}."""
        doc_match = re.search(
            r"\\begin\{document\}(.*?)(?:\\end\{document\}|$)",
            text,
            flags=re.DOTALL,
        )
        if doc_match:
            return doc_match.group(1)
        return text

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """
        Splits prose text into sentences, protecting inline math, abbreviations,
        URLs, and decimal numbers from splitting.
        """
        if not text.strip():
            return []

        # 1. Protect inline math $...$
        math_tokens: Dict[str, str] = {}
        def replace_math(m):
            token = f"__MATH_TOKEN_{len(math_tokens)}__"
            math_tokens[token] = m.group(0)
            return token

        tokenized = re.sub(r"\$[^$]+\$", replace_math, text)

        # 2. Protect URLs
        url_tokens: Dict[str, str] = {}
        def replace_url(m):
            token = f"__URL_TOKEN_{len(url_tokens)}__"
            url_tokens[token] = m.group(0)
            return token

        tokenized = re.sub(r"https?://[^\s]+", replace_url, tokenized)

        # 3. Protect decimal numbers (e.g. 3.14 -> 3<DEC>14)
        tokenized = re.sub(r"(\d+)\.(\d+)", r"\1__DECIMAL_DOT__\2", tokenized)

        # 4. Protect abbreviations
        abbrev_tokens: Dict[str, str] = {}
        for abbrev_regex in cls.ABBREVIATIONS:
            def replace_abbrev(m):
                token = f"__ABBR_TOKEN_{len(abbrev_tokens)}__"
                abbrev_tokens[token] = m.group(0)
                return token
            tokenized = re.sub(abbrev_regex, replace_abbrev, tokenized, flags=re.IGNORECASE)

        # 5. Split by sentence terminators (. ! ?) followed by whitespace and uppercase/quote
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\(\[\"\'])", tokenized)

        # 6. Restore tokens and clean up sentences
        sentences = []
        for s in raw_sentences:
            s = s.strip()
            if not s:
                continue

            # Restore decimal dots
            s = s.replace("__DECIMAL_DOT__", ".")

            # Restore abbreviations
            for token, orig in abbrev_tokens.items():
                s = s.replace(token, orig)

            # Restore URLs
            for token, orig in url_tokens.items():
                s = s.replace(token, orig)

            # Restore math
            for token, orig in math_tokens.items():
                s = s.replace(token, orig)

            cleaned_s = cls.clean_latex(s)
            # Filter out empty or trivially short non-prose tokens
            if len(cleaned_s) > 2 and any(c.isalnum() for c in cleaned_s):
                sentences.append(cleaned_s)

        return sentences

    @classmethod
    def parse_version(
        cls,
        arxiv_id: str,
        version: str,
        inlined_tex: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Parses inlined TeX of a version into (elements, hierarchy_edges).
        Elements format:
            {"raw_id": str, "kind": str, "content": str}
        Hierarchy format:
            {"arxiv_id": str, "version": str, "element_raw_id": str, "parent_raw_id": Optional[str]}
        """
        # Standardize math and remove references section
        std_tex = cls.standardize_math(inlined_tex)
        no_ref_tex = cls.strip_references_section(std_tex)
        body_tex = cls.extract_document_body(no_ref_tex)

        elements: List[Dict[str, Any]] = []
        hierarchy: List[Dict[str, Any]] = []

        counter = 0
        def next_id(prefix: str) -> str:
            nonlocal counter
            counter += 1
            return f"{prefix}_{counter}"

        # 1. Document Root
        doc_root_id = f"doc_{version}"
        elements.append({
            "raw_id": doc_root_id,
            "kind": "document",
            "content": f"Document {arxiv_id} {version}",
        })
        hierarchy.append({
            "arxiv_id": arxiv_id,
            "version": version,
            "element_raw_id": doc_root_id,
            "parent_raw_id": None,
        })

        # Stack of (level, element_id): level 0 is document root
        heading_stack: List[Tuple[int, str]] = [(0, doc_root_id)]

        def get_current_parent() -> str:
            return heading_stack[-1][1]

        # Regex for block structures:
        # - Headings: \section*{...}, \subsection{...}, etc.
        # - Block formula: \begin{equation}...\end{equation}, \begin{align}...\end{align}, etc.
        # - Figures & Tables: \begin{figure}...\end{figure}, \begin{table}...\end{table}
        # - Lists: \begin{itemize}...\end{itemize}, \begin{enumerate}...\end{enumerate}
        BLOCK_SPLIT_REGEX = re.compile(
            r"("
            r"\\(?:chapter|section|subsection|subsubsection|paragraph)\*?\s*\{[^}]*\}|"
            r"\\begin\{(?:equation|equation\*|align|align\*|gather|gather\*)\}.*?\\end\{(?:equation|equation\*|align|align\*|gather|gather\*)\}|"
            r"\\begin\{(?:figure|figure\*|table|table\*)\}.*?\\end\{(?:figure|figure\*|table|table\*)\}|"
            r"\\begin\{(?:itemize|enumerate)\}.*?\\end\{(?:itemize|enumerate)\}"
            r")",
            re.DOTALL,
        )

        tokens = BLOCK_SPLIT_REGEX.split(body_tex)

        for token in tokens:
            token = token.strip()
            if not token:
                continue

            # Check if token is a Heading
            heading_match = re.match(
                r"\\(chapter|section|subsection|subsubsection|paragraph)\*?\s*\{([^}]*)\}",
                token,
                re.DOTALL,
            )
            if heading_match:
                h_type = heading_match.group(1).lower()
                h_title = cls.clean_latex(heading_match.group(2).strip())
                h_level = cls.HEADING_LEVELS.get(h_type, 2)

                # Pop headings of greater or equal level
                while len(heading_stack) > 1 and heading_stack[-1][0] >= h_level:
                    heading_stack.pop()

                parent_id = get_current_parent()
                h_elem_id = next_id(h_type)

                elements.append({
                    "raw_id": h_elem_id,
                    "kind": h_type,
                    "content": h_title,
                })
                hierarchy.append({
                    "arxiv_id": arxiv_id,
                    "version": version,
                    "element_raw_id": h_elem_id,
                    "parent_raw_id": parent_id,
                })

                heading_stack.append((h_level, h_elem_id))
                continue

            # Check if token is a Block Formula
            if re.match(r"\\begin\{(?:equation|equation\*|align|align\*|gather|gather\*)\}", token):
                clean_formula = cls.clean_latex(token)
                f_elem_id = next_id("formula")
                elements.append({
                    "raw_id": f_elem_id,
                    "kind": "block_formula",
                    "content": clean_formula,
                })
                hierarchy.append({
                    "arxiv_id": arxiv_id,
                    "version": version,
                    "element_raw_id": f_elem_id,
                    "parent_raw_id": get_current_parent(),
                })
                continue

            # Check if token is a Figure or Table
            if re.match(r"\\begin\{(?:figure|figure\*|table|table\*)\}", token):
                clean_fig = cls.clean_latex(token)
                fig_elem_id = next_id("figure")
                elements.append({
                    "raw_id": fig_elem_id,
                    "kind": "figure",
                    "content": clean_fig,
                })
                hierarchy.append({
                    "arxiv_id": arxiv_id,
                    "version": version,
                    "element_raw_id": fig_elem_id,
                    "parent_raw_id": get_current_parent(),
                })
                continue

            # Check if token is Itemize / Enumerate
            if re.match(r"\\begin\{(?:itemize|enumerate)\}", token):
                list_match = re.search(r"\\begin\{(?:itemize|enumerate)\}(.*?)\\end\{(?:itemize|enumerate)\}", token, re.DOTALL)
                list_content = list_match.group(1) if list_match else token

                list_parent_id = next_id("itemize")
                elements.append({
                    "raw_id": list_parent_id,
                    "kind": "itemize",
                    "content": "itemize_environment",
                })
                hierarchy.append({
                    "arxiv_id": arxiv_id,
                    "version": version,
                    "element_raw_id": list_parent_id,
                    "parent_raw_id": get_current_parent(),
                })

                # Split by \item
                items = re.split(r"\\item\b", list_content)
                for it in items:
                    it_clean = cls.clean_latex(it.strip())
                    if it_clean:
                        it_elem_id = next_id("item")
                        elements.append({
                            "raw_id": it_elem_id,
                            "kind": "item",
                            "content": it_clean,
                        })
                        hierarchy.append({
                            "arxiv_id": arxiv_id,
                            "version": version,
                            "element_raw_id": it_elem_id,
                            "parent_raw_id": list_parent_id,
                        })
                continue

            # Otherwise: prose text -> split into sentences
            sentences = cls.split_sentences(token)
            for sentence in sentences:
                s_elem_id = next_id("s")
                elements.append({
                    "raw_id": s_elem_id,
                    "kind": "sentence",
                    "content": sentence,
                })
                hierarchy.append({
                    "arxiv_id": arxiv_id,
                    "version": version,
                    "element_raw_id": s_elem_id,
                    "parent_raw_id": get_current_parent(),
                })

        return elements, hierarchy
