"""
Hierarchy construction for one version of a publication.

Turns the inlined LaTeX of a version into a tree following the Milestone 2
parsing notes:

1. Node content keeps the raw LaTeX; cleanup and math normalization happen
   later in ``HierarchyNormalizer``.
2. The References section is excluded from the tree. Its raw blocks are kept
   in ``raw_references.json`` so the BibTeX stage can convert them.
3. The root node ``<arxiv_id>#root<N>`` only acts as a parent, so ``parent_id``
   is never null.
4. Versions without TeX sources are recorded explicitly as empty records.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .gatherer import MultiFileGatherer

logger = logging.getLogger(__name__)


class HierarchyBuilder:
    """Build the hierarchy tree of each version of a paper from its LaTeX source."""

    # Level of each sectioning command (smaller = closer to the root)
    HEADING_LEVELS: Dict[str, int] = {
        "part": 1,
        "chapter": 1,
        "section": 2,
        "subsection": 3,
        "subsubsection": 4,
        "paragraph": 5,
        "subparagraph": 6,
    }

    # Titles of the bibliography section, which stays out of the tree
    REFERENCE_TITLES = {
        "references",
        "reference",
        "bibliography",
        "works cited",
        "literature cited",
        "references and notes",
    }

    # Abbreviations that never end a sentence (matched case-insensitively)
    ABBREVIATION_PATTERNS = [
        r"\be\.g\.",
        r"\bi\.e\.",
        r"\bet\s+al\.",
        r"\bviz\.",
        r"\bvs\.",
        r"\bcf\.",
        r"\bapprox\.",
        r"\bresp\.",
        r"\bw\.r\.t\.",
        r"\ba\.k\.a\.",
        r"\b(?:Fig|Figs|Tab|Tabs|Eq|Eqs|Sec|Secs|Ref|Refs|Thm|Lem|Def|Prop|Cor|Alg)\.",
    ]
    # Titles and company suffixes: case-sensitive, so "10 ms." still ends a sentence
    CASE_SENSITIVE_ABBREVIATIONS = [
        r"\b(?:Dr|Prof|Mr|Mrs|Ms|St|Jr|Sr|Inc|Ltd|Univ)\.",
    ]
    # Only abbreviations when a number follows ("No. 3", "pp. 12"); otherwise a
    # sentence such as "There is no." must still end at the dot
    NUMBERED_ABBREVIATIONS = [
        r"\b(?:No|Nos|Vol|Vols|pp|p|Ch|Chap|App)\.(?=\s*~?\s*\(?\d)",
    ]

    HEADING_PARSE_RE = re.compile(
        r"^\\(?P<type>part|chapter|section|subsection|subsubsection|paragraph|subparagraph)"
        r"\*?(?:\[[^\]]*\])?\s*\{(?P<title>.*)\}$",
        re.DOTALL,
    )

    # Tokens that make a user macro relevant to the document structure
    STRUCTURAL_MACRO_TOKENS = (
        "\\begin{", "\\end{", "\\section", "\\subsection", "\\chapter",
        "\\paragraph", "\\part", "\\item", "$$", "\\[", "\\]", "\\appendix",
    )
    MACRO_DEFINITION_RE = re.compile(
        r"\\(?:newcommand|renewcommand|providecommand)\*?\s*"
        r"(?:\{\s*\\(?P<name1>[A-Za-z@]+)\s*\}|\\(?P<name2>[A-Za-z@]+))"
        r"\s*(?:\[(?P<nargs>\d)\])?\s*(?:\[[^\]]*\])?\s*\{"
        r"|\\def\s*\\(?P<name3>[A-Za-z@]+)\s*(?P<params>(?:#\d)*)\s*\{"
    )

    @classmethod
    def strip_comments(cls, text: str) -> str:
        """Remove LaTeX comments (% ...) while keeping escaped \\%."""
        return re.sub(r"(?<!\\)(?:\\\\)*%.*$", "", text, flags=re.MULTILINE)

    @classmethod
    def normalize_whitespace_only(cls, text: str) -> str:
        """Collapse whitespace while keeping every LaTeX command and formula intact."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
        return text.strip()

    # ------------------------------------------------------------------ #
    # User macros
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_braced(text: str, open_pos: int) -> Tuple[Optional[str], int]:
        """Return the content of the brace group opening at ``open_pos`` and its end."""
        if open_pos >= len(text) or text[open_pos] != "{":
            return None, open_pos
        depth = 0
        pos = open_pos
        while pos < len(text):
            char = text[pos]
            if char == "\\":
                pos += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[open_pos + 1:pos], pos + 1
            pos += 1
        return None, open_pos

    @classmethod
    def expand_structural_macros(cls, tex: str, max_rounds: int = 4) -> str:
        """Expand user macros that hide document structure.

        Papers often write ``\\newcommand{\\beq}{\\begin{equation}}`` or
        ``\\newcommand{\\sect}[1]{\\section{#1}}``. Without expansion such
        formulas and headings would be read as plain sentences. Only macros
        whose body contains a structural token (environments, sectioning,
        ``\\item``, display math) are expanded; other macros stay untouched.
        """
        macros: Dict[str, Tuple[int, str]] = {}
        spans: List[Tuple[int, int]] = []
        for match in cls.MACRO_DEFINITION_RE.finditer(tex):
            name = match.group("name1") or match.group("name2") or match.group("name3")
            body, end = cls._read_braced(tex, match.end() - 1)
            if body is None:
                continue
            if match.group("name3"):
                nargs = len(re.findall(r"#\d", match.group("params") or ""))
            else:
                nargs = int(match.group("nargs") or 0)
            if any(token in body for token in cls.STRUCTURAL_MACRO_TOKENS):
                macros[name] = (nargs, body)
                spans.append((match.start(), end))

        if not macros:
            return tex

        # Drop the definitions first so that their own name is not expanded.
        for start, end in reversed(spans):
            tex = tex[:start] + tex[end:]

        pattern = re.compile(
            r"\\(" + "|".join(re.escape(n) for n in sorted(macros, key=len, reverse=True))
            + r")(?![A-Za-z@])"
        )
        for _ in range(max_rounds):
            out: List[str] = []
            pos = 0
            changed = False
            for match in pattern.finditer(tex):
                if match.start() < pos:
                    continue
                nargs, body = macros[match.group(1)]
                cursor = match.end()
                args: List[str] = []
                for _arg in range(nargs):
                    while cursor < len(tex) and tex[cursor].isspace():
                        cursor += 1
                    value, next_cursor = cls._read_braced(tex, cursor)
                    if value is None:
                        break
                    args.append(value)
                    cursor = next_cursor
                if len(args) != nargs:
                    continue
                replacement = body
                for index, value in enumerate(args, start=1):
                    replacement = replacement.replace(f"#{index}", value)
                out.append(tex[pos:match.start()])
                out.append(replacement)
                pos = cursor
                changed = True
            out.append(tex[pos:])
            tex = "".join(out)
            if not changed:
                break
        return tex

    # ------------------------------------------------------------------ #
    # Body extraction
    # ------------------------------------------------------------------ #
    @classmethod
    def extract_body_and_preprocess(
        cls, raw_tex: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Extract the document body (between \\begin{document} and \\end{document}),
        set the References blocks aside, turn Abstract/Acknowledgements/Appendix
        into sections so they enter the tree, and drop front-matter commands.
        """
        text = cls.strip_comments(raw_tex)
        text = cls.expand_structural_macros(text)

        doc_match = re.search(
            r"\\begin\{document\}(.*?)(?:\\end\{document\}|$)",
            text,
            flags=re.DOTALL,
        )
        body = doc_match.group(1) if doc_match else text

        preserved_bib_envs: List[str] = []
        preserved_bib_commands: List[str] = []
        preserved_bibitems: List[str] = []

        # 1. Keep thebibliography blocks and their \bibitem entries aside
        def _capture_bib_env(m: re.Match) -> str:
            block = m.group(0).strip()
            preserved_bib_envs.append(block)
            # Drop the closing tag before splitting the \bibitem entries
            inner_block = re.sub(
                r"\\end\{(?:thebibliography|references|bibunit)\}\s*$",
                "",
                block,
                flags=re.IGNORECASE,
            )
            items = re.split(r"(?=\\bibitem\b)", inner_block)
            for it in items:
                if it.strip().startswith("\\bibitem"):
                    preserved_bibitems.append(it.strip())
            return ""

        body = re.sub(
            r"\\begin\{(?:thebibliography|references|bibunit)\}.*?\\end\{(?:thebibliography|references|bibunit)\}",
            _capture_bib_env,
            body,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Keep \bibliography{...} and \printbibliography commands
        for m_bib in re.finditer(r"\\bibliography(?:style)?\s*\{[^{}]*\}|\\printbibliography(?:\s*\[[^\]]*\])?", body):
            preserved_bib_commands.append(m_bib.group(0).strip())

        body = re.sub(r"\\bibliography(?:style)?\s*\{[^{}]*\}", "", body)
        body = re.sub(r"\\printbibliography(?:\s*\[[^\]]*\])?", "", body)

        # 2. Acknowledgements / abstract / appendix become sections of the tree
        body = re.sub(
            r"\\begin\{(?:acknowledg(?:e)?ments?|acks)\*?\}(.*?)\\end\{(?:acknowledg(?:e)?ments?|acks)\*?\}",
            r"\n\\section*{Acknowledgements}\n\1\n",
            body,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # Unpaired forms used by revtex/aastex: \begin{acknowledgments} without
        # \end, a stray \end, or the bare \acknowledgments command
        body = re.sub(
            r"\\begin\{(?:acknowledg(?:e)?ments?|acks)\*?\}",
            r"\n\\section*{Acknowledgements}\n",
            body,
            flags=re.IGNORECASE,
        )
        body = re.sub(r"\\end\{(?:acknowledg(?:e)?ments?|acks)\*?\}", "", body, flags=re.IGNORECASE)
        body = re.sub(
            r"\\(?:acknowledg(?:e)?ments?|acks)\b(?!\s*\{)",
            r"\n\\section*{Acknowledgements}\n",
            body,
            flags=re.IGNORECASE,
        )
        body = re.sub(
            r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
            r"\n\\section*{Abstract}\n\1\n",
            body,
            flags=re.DOTALL | re.IGNORECASE,
        )

        def _convert_appendix_env(m: re.Match) -> str:
            inner_app = m.group(1)
            # An appendix environment that already has \section/\chapter acts as \appendix
            if re.search(r"\\(?:chapter|section)\*?\s*(?:\[[^\]]*\])?\{", inner_app):
                return f"\n\\appendix\n{inner_app}\n"
            return f"\n\\section*{{Appendix}}\n{inner_app}\n"

        body = re.sub(
            r"\\begin\{appendi(?:x|ces)\}(.*?)\\end\{appendi(?:x|ces)\}",
            _convert_appendix_env,
            body,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # 3. Drop front-matter commands (title, authors, affiliations, ...)
        frontmatter_cmds = [
            "title",
            "author",
            "affiliation",
            "address",
            "email",
            "date",
            "preprint",
            "pacs",
            "keywords",
            "thanks",
            "dedicatory",
        ]
        for cmd in frontmatter_cmds:
            pattern = (
                rf"\\{cmd}\*?(?:\[[^\]]*\])?\s*"
                r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"
            )
            body = re.sub(pattern, "", body, flags=re.DOTALL)
        body = re.sub(r"\\maketitle\b", "", body)

        preserved_refs = {
            "thebibliography_blocks": preserved_bib_envs,
            "bibitem_entries": preserved_bibitems,
            "bibliography_commands": preserved_bib_commands,
            "reference_sections": [],
        }

        return body, preserved_refs

    # ------------------------------------------------------------------ #
    # Sentences
    # ------------------------------------------------------------------ #
    @classmethod
    def split_sentences_safe(cls, text: str) -> List[str]:
        """
        Split prose into sentences by masking what must not be split: inline
        math, LaTeX commands, URLs, decimal numbers and abbreviations. The
        original LaTeX is restored in every returned sentence.
        """
        if not text or not text.strip():
            return []

        protected_tokens: Dict[str, str] = {}

        def mask(match: re.Match, prefix: str) -> str:
            key = f"__{prefix}_{len(protected_tokens)}__"
            protected_tokens[key] = match.group(0)
            return key

        work = text

        # 1. Inline math \(...\) and $...$ (an escaped \$ is not a delimiter)
        work = re.sub(r"\\\((.*?)\\\)", lambda m: mask(m, "INLMATH"), work, flags=re.DOTALL)
        work = re.sub(r"(?<!\\)\$(?:\\.|[^$\\])+(?<!\\)\$", lambda m: mask(m, "MATH"), work)

        # 2. URLs (without a sentence-final .,;:!?) at their end)
        work = re.sub(r"https?://[^\s\}]+(?<![.,;:!?\)\]])", lambda m: mask(m, "URL"), work)

        # 3. Commands with arguments (\cite{...}, \ref{fig.1}, \href{...}{...})
        work = re.sub(
            r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{(?:[^{}]|\{[^{}]*\})*\})+",
            lambda m: mask(m, "CMD"),
            work,
        )

        # 4. Decimal points (3.14)
        work = re.sub(r"(\d+)\.(\d+)", r"\1__DEC_DOT__\2", work)

        # 5. Abbreviations (i.e., e.g., et al., Fig. 1, Dr., No. 3 ...)
        for abbr_pat in cls.ABBREVIATION_PATTERNS + cls.NUMBERED_ABBREVIATIONS:
            work = re.sub(abbr_pat, lambda m: mask(m, "ABBR"), work, flags=re.IGNORECASE)
        for abbr_pat in cls.CASE_SENSITIVE_ABBREVIATIONS:
            work = re.sub(abbr_pat, lambda m: mask(m, "ABBR"), work)

        # 6. Break after [.!?] (also when followed by closing quotes/brackets or a
        #    masked \footnote / \cite)
        work_with_breaks = re.sub(
            r"([.!?](?:[\)\]\'\"]+|__CMD_\d+__)*)\s+(?=[\w\(\[\"\'\\]|__(?:MATH|INLMATH|CMD|ABBR|URL)_)",
            r"\1\n__SENT_BREAK__\n",
            work,
        )
        raw_parts = work_with_breaks.split("\n__SENT_BREAK__\n")

        sentences: List[str] = []
        for part in raw_parts:
            s = part.strip()
            if not s:
                continue
            s = s.replace("__DEC_DOT__", ".")
            # Restore in reverse order: later masks may contain earlier ones
            for key in reversed(list(protected_tokens.keys())):
                if key in s:
                    s = s.replace(key, protected_tokens[key])

            raw_sentence = cls.normalize_whitespace_only(s)
            if len(raw_sentence) > 2 and any(c.isalnum() for c in raw_sentence):
                sentences.append(raw_sentence)

        return sentences

    # ------------------------------------------------------------------ #
    # Blocks
    # ------------------------------------------------------------------ #
    @classmethod
    def _extract_top_level_blocks(cls, text: str) -> List[Tuple[str, str]]:
        """
        Split text into ordered blocks: heading, appendix_marker, block_formula,
        figure (tables included), list, and prose.
        """
        TOKEN_FINDER = re.compile(
            r"(?P<heading>\\(?:part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?(?:\[[^\]]*\])?\s*\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})|"
            r"(?P<appendix>\\appendix\b)|"
            r"(?P<formula_env>\\begin\{(?P<f_env>equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?|flalign\*?|displaymath)\}.*?\\end\{(?P=f_env)\})|"
            r"(?P<formula_dollar>(?<!\\)\$\$.*?(?<!\\)\$\$)|"
            r"(?P<formula_bracket>\\\[.*?\\\])|"
            r"(?P<figure>\\begin\{(?P<fig_env>figure\*?|table\*?|algorithm\*?|wrapfigure|sidewaysfigure|sidewaystable|longtable)\}.*?\\end\{(?P=fig_env)\})|"
            r"(?P<list_begin>\\begin\{(?:itemize|enumerate|description)\})",
            re.DOTALL,
        )

        LIST_ENV_TAG = re.compile(r"\\(?P<tag>begin|end)\{(?P<env>itemize|enumerate|description)\}")

        blocks: List[Tuple[str, str]] = []
        pos = 0
        n = len(text)

        while pos < n:
            m = TOKEN_FINDER.search(text, pos)
            if not m:
                rem = text[pos:].strip()
                if rem:
                    blocks.append(("prose", rem))
                break

            start = m.start()
            if start > pos:
                prose = text[pos:start].strip()
                if prose:
                    blocks.append(("prose", prose))

            if m.group("heading"):
                blocks.append(("heading", m.group("heading")))
                pos = m.end()
            elif m.group("appendix"):
                blocks.append(("appendix_marker", m.group("appendix")))
                pos = m.end()
            elif m.group("formula_env") or m.group("formula_dollar") or m.group("formula_bracket"):
                raw_f = m.group("formula_env") or m.group("formula_dollar") or m.group("formula_bracket")
                blocks.append(("block_formula", raw_f))
                pos = m.end()
            elif m.group("figure"):
                blocks.append(("figure", m.group("figure")))
                pos = m.end()
            elif m.group("list_begin"):
                # Count begin/end pairs so that nested lists stay in one block
                depth = 0
                scan_pos = start
                list_end = n
                while scan_pos < n:
                    lm = LIST_ENV_TAG.search(text, scan_pos)
                    if not lm:
                        break
                    if lm.group("tag") == "begin":
                        depth += 1
                    else:
                        depth -= 1
                        if depth == 0:
                            list_end = lm.end()
                            break
                    scan_pos = lm.end()

                blocks.append(("list", text[start:list_end]))
                pos = list_end

        return blocks

    @classmethod
    def _split_list_items(cls, list_block: str) -> Tuple[str, List[str]]:
        """Split the top-level \\item entries of a list (nested lists stay inside their item)."""
        m_open = re.match(r"\\begin\{(?P<env>itemize|enumerate|description)\}(?:\[[^\]]*\])?", list_block)
        env_name = m_open.group("env") if m_open else "itemize"
        inner_start = m_open.end() if m_open else 0

        m_close = re.search(rf"\\end\{{{env_name}\}}\s*$", list_block)
        inner_end = m_close.start() if m_close else len(list_block)
        inner = list_block[inner_start:inner_end]

        TAG_OR_ITEM = re.compile(
            r"\\(?P<tag>begin|end)\{(?:itemize|enumerate|description)\}|(?P<item>\\item(?:\s*\[[^\]]*\])?\b)"
        )
        depth = 0
        item_starts: List[int] = []
        item_content_starts: List[int] = []

        for match in TAG_OR_ITEM.finditer(inner):
            if match.group("tag") == "begin":
                depth += 1
            elif match.group("tag") == "end":
                depth = max(0, depth - 1)
            elif match.group("item") and depth == 0:
                item_starts.append(match.start())
                item_content_starts.append(match.end())

        items: List[str] = []
        for idx, c_start in enumerate(item_content_starts):
            c_end = item_starts[idx + 1] if idx + 1 < len(item_starts) else len(inner)
            seg = inner[c_start:c_end].strip()
            if seg:
                items.append(seg)

        return env_name, items

    # ------------------------------------------------------------------ #
    # Version parsing
    # ------------------------------------------------------------------ #
    @classmethod
    def parse_version(
        cls,
        arxiv_id: str,
        version: str,
        inlined_tex: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Parse one version into:
        - elements: every node with its raw content;
        - hierarchy: child-parent edges (the root is only ever a parent);
        - preserved_refs: the bibliography blocks set aside for BibTeX extraction.
        """
        v_num = int(version[1:]) if version.startswith("v") and version[1:].isdigit() else int(version)
        body_tex, preserved_refs = cls.extract_body_and_preprocess(inlined_tex)

        elements: List[Dict[str, Any]] = []
        hierarchy: List[Dict[str, Any]] = []
        counter = 0

        def add_node(kind: str, content: str, parent_id: Optional[str], prefix: str) -> str:
            nonlocal counter
            counter += 1
            node_id = f"{prefix}_{counter}"
            elements.append({"raw_id": node_id, "kind": kind, "content": content})
            if parent_id is not None:
                hierarchy.append({
                    "arxiv_id": arxiv_id,
                    "version": v_num,
                    "element_raw_id": node_id,
                    "parent_raw_id": parent_id,
                })
            return node_id

        # Root document node (root1, root2, ...): stored in elements only
        doc_root_id = f"root{v_num}"
        elements.append({
            "raw_id": doc_root_id,
            "kind": "document",
            "content": f"Version {v_num} root",
        })

        # Stack of (level, element id) used to find the parent of each heading
        heading_stack: List[Tuple[int, str]] = [(0, doc_root_id)]
        in_appendix = False
        skip_until_level: Optional[int] = None

        def add_leaf_block(b_type: str, b_text: str, parent_id: str) -> None:
            """Attach one formula/figure/list/prose block below ``parent_id``."""
            if b_type == "block_formula":
                raw_f = cls.normalize_whitespace_only(b_text)
                if raw_f:
                    add_node("block_formula", raw_f, parent_id, "formula")
            elif b_type == "figure":
                raw_fig = cls.normalize_whitespace_only(b_text)
                if raw_fig:
                    add_node("figure", raw_fig, parent_id, "figure")
            elif b_type == "list":
                process_list_block(b_text, parent_id)
            elif b_type == "prose":
                for sent in cls.split_sentences_safe(b_text):
                    add_node("sentence", sent, parent_id, "s")

        def parse_content_into_parent(content_str: str, parent_id: str) -> None:
            """Split the content of an \\item into leaves, nested lists and headings."""
            for b_type, b_text in cls._extract_top_level_blocks(content_str):
                if b_type == "heading":
                    hm = cls.HEADING_PARSE_RE.match(b_text.strip())
                    title = cls.normalize_whitespace_only(hm.group("title")) if hm else ""
                    if hm and title:
                        add_node(hm.group("type").lower(), title, parent_id, hm.group("type").lower())
                elif b_type != "appendix_marker":
                    add_leaf_block(b_type, b_text, parent_id)

        def process_list_block(list_text: str, parent_id: str) -> None:
            """A list is a branch node, each \\item a child, its sentences the leaves."""
            _env_name, items = cls._split_list_items(list_text)
            list_id = add_node(
                "itemize", cls.normalize_whitespace_only(list_text), parent_id, "itemize"
            )
            for item_raw in items:
                item_clean = cls.normalize_whitespace_only(item_raw)
                if not item_clean:
                    continue
                item_id = add_node("item", f"\\item {item_clean}", list_id, "item")
                before_len = len(elements)
                parse_content_into_parent(item_raw, item_id)
                if len(elements) == before_len:
                    add_node("sentence", item_clean, item_id, "s")

        for b_type, b_text in cls._extract_top_level_blocks(body_tex):
            if b_type == "appendix_marker":
                in_appendix = True
                skip_until_level = None
                continue

            if b_type == "heading":
                hm = cls.HEADING_PARSE_RE.match(b_text.strip())
                if not hm:
                    continue
                h_type = hm.group("type").lower()
                raw_title = cls.normalize_whitespace_only(hm.group("title"))
                if not raw_title:
                    # Empty headings such as \paragraph{} only break lines
                    continue
                h_level = cls.HEADING_LEVELS.get(h_type, 2)

                # Plain-letter title used to recognise the References section
                norm_title = re.sub(r"\\[a-zA-Z]+\*?", "", raw_title)
                norm_title = re.sub(r"[^a-zA-Z\s]", "", norm_title).strip().lower()
                norm_title = re.sub(r"\s+", " ", norm_title)

                if skip_until_level is not None:
                    if h_level <= skip_until_level:
                        skip_until_level = None
                    else:
                        # Subsection of References: keep aside, not in the tree
                        preserved_refs["reference_sections"].append(b_text)
                        continue

                if norm_title in cls.REFERENCE_TITLES:
                    skip_until_level = h_level
                    preserved_refs["reference_sections"].append(b_text)
                    continue

                # A section titled "Appendix"/"Appendices" becomes the parent of
                # the appendix sections that follow it
                if norm_title in ("appendix", "appendices") and h_type == "section":
                    h_level = 1
                    in_appendix = True
                elif in_appendix and h_type in ("chapter", "section") and not raw_title.lower().startswith("appendix"):
                    raw_title = f"Appendix: {raw_title}"

                # Pop headings of the same or a deeper level to find the parent
                while len(heading_stack) > 1 and heading_stack[-1][0] >= h_level:
                    heading_stack.pop()

                h_id = add_node(h_type, raw_title, heading_stack[-1][1], h_type)
                heading_stack.append((h_level, h_id))
                continue

            # Inside References: keep the content aside for BibTeX extraction
            if skip_until_level is not None:
                preserved_refs["reference_sections"].append(b_text)
                clean_ref_text = re.sub(
                    r"\\end\{(?:thebibliography|references|bibunit)\}\s*$",
                    "",
                    b_text,
                    flags=re.IGNORECASE,
                )
                for m_item in re.split(r"(?=\\bibitem\b)", clean_ref_text):
                    if m_item.strip().startswith("\\bibitem"):
                        preserved_refs["bibitem_entries"].append(m_item.strip())
                continue

            add_leaf_block(b_type, b_text, heading_stack[-1][1])

        return elements, hierarchy, preserved_refs

    @classmethod
    def build_version_hierarchy_outputs(
        cls,
        arxiv_id: str,
        version_name: str,
        raw_elements: List[Dict[str, Any]],
        raw_hierarchy: List[Dict[str, Any]],
        is_empty: bool = False,
        error_msg: Optional[str] = None,
        status: str = "success",
    ) -> Dict[str, Any]:
        """
        Assign version-level IDs (`<arxiv_id>#root1`, `<arxiv_id>#v1_s1`, ...)
        and build the nested tree. Empty versions (empty_version), failed
        versions (failed) and partially gathered versions (partial) stay distinct.
        """
        v_num = int(version_name[1:]) if version_name.startswith("v") and version_name[1:].isdigit() else int(version_name)

        if status == "failed":
            return {
                "arxiv_id": arxiv_id,
                "version": v_num,
                "version_name": version_name,
                "status": "failed",
                "is_empty": False,
                "error": error_msg or "Gathering or parsing the TeX source of this version failed",
                "elements": [],
                "hierarchy": [],
                "tree": {
                    "arxiv_id": arxiv_id,
                    "version": v_num,
                    "version_name": version_name,
                    "status": "failed",
                    "is_empty": False,
                    "elements_count": 0,
                    "edges_count": 0,
                    "root": None,
                },
            }

        if is_empty or status == "empty_version" or not raw_elements:
            return {
                "arxiv_id": arxiv_id,
                "version": v_num,
                "version_name": version_name,
                "status": "empty_version",
                "is_empty": True,
                "error": error_msg or "Version without TeX sources (empty record)",
                "elements": [],
                "hierarchy": [],
                "tree": {
                    "arxiv_id": arxiv_id,
                    "version": v_num,
                    "version_name": version_name,
                    "status": "empty_version",
                    "is_empty": True,
                    "elements_count": 0,
                    "edges_count": 0,
                    "root": None,
                },
            }

        kind_counters: Dict[str, int] = {}

        def get_prefix(k: str) -> str:
            if k == "sentence":
                return "s"
            if k in ("part", "chapter", "section"):
                return "h"
            if k == "subsection":
                return "ssec"
            if k in ("subsubsection", "paragraph", "subparagraph"):
                return "par"
            if k == "block_formula":
                return "eq"
            if k == "figure":
                return "fig"
            if k == "item":
                return "item"
            if k == "itemize":
                return "list"
            return "elem"

        raw_to_version_id: Dict[str, str] = {}
        version_elements: List[Dict[str, Any]] = []

        for elem in raw_elements:
            r_id = elem["raw_id"]
            k = elem["kind"]
            content = elem["content"]

            if k == "document":
                e_id = f"{arxiv_id}#root{v_num}"
            else:
                pref = get_prefix(k)
                cnt = kind_counters.get(pref, 0) + 1
                kind_counters[pref] = cnt
                e_id = f"{arxiv_id}#v{v_num}_{pref}{cnt}"

            raw_to_version_id[r_id] = e_id
            version_elements.append({
                "element_id": e_id,
                "arxiv_id": arxiv_id,
                "kind": k,
                "content": content,
            })

        version_edges: List[Dict[str, Any]] = []
        for edge in raw_hierarchy:
            c_raw = edge["element_raw_id"]
            p_raw = edge["parent_raw_id"]
            if p_raw is None:
                continue  # parent_id is never null
            c_id = raw_to_version_id.get(c_raw)
            p_id = raw_to_version_id.get(p_raw)
            if c_id and p_id:
                version_edges.append({
                    "arxiv_id": arxiv_id,
                    "version": v_num,
                    "element_id": c_id,
                    "parent_id": p_id,
                })

        nested_tree = cls.build_nested_tree(
            arxiv_id=arxiv_id,
            version_name=version_name,
            elements=version_elements,
            hierarchy_edges=version_edges,
        )
        nested_tree["status"] = status

        return {
            "arxiv_id": arxiv_id,
            "version": v_num,
            "version_name": version_name,
            "status": status,
            "is_empty": False,
            "error": error_msg,
            "elements": version_elements,
            "hierarchy": version_edges,
            "tree": nested_tree,
        }

    @classmethod
    def build_nested_tree(
        cls,
        arxiv_id: str,
        version_name: str,
        elements: List[Dict[str, Any]],
        hierarchy_edges: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the nested JSON tree of one version."""
        v_num = int(version_name[1:]) if version_name.startswith("v") and version_name[1:].isdigit() else int(version_name)
        elem_lookup = {
            e.get("element_id", e.get("raw_id")): e for e in elements
        }

        candidate_roots = [
            f"{arxiv_id}#root{v_num}",
            f"{arxiv_id}#doc_{version_name}",
            f"root{v_num}",
            f"doc_{version_name}",
        ]
        root_id = None
        for cr in candidate_roots:
            if cr in elem_lookup:
                root_id = cr
                break

        if root_id is None:
            raise ValueError(
                f"Invalid hierarchy: no root node for {arxiv_id} {version_name} in the elements."
            )

        children_map: Dict[str, List[str]] = {}
        for edge in hierarchy_edges:
            edge_v = edge.get("version")
            if edge_v == v_num or edge_v == version_name:
                p_id = edge.get("parent_id") or edge.get("parent_raw_id")
                c_id = edge.get("element_id") or edge.get("element_raw_id")
                if not p_id or not c_id:
                    raise ValueError(
                        f"Invalid hierarchy: edge without parent_id or element_id ({edge})."
                    )
                if p_id not in elem_lookup or c_id not in elem_lookup:
                    # A dangling edge must not create unknown nodes nor abort the
                    # whole publication; report it and leave it out of the tree.
                    logger.warning(
                        "Skipping dangling hierarchy edge %s -> %s in %s %s",
                        c_id, p_id, arxiv_id, version_name,
                    )
                    continue
                children_map.setdefault(p_id, []).append(c_id)

        active_path: set[str] = set()

        def build_node(node_id: str) -> Dict[str, Any]:
            if node_id not in elem_lookup:
                raise ValueError(
                    f"Invalid hierarchy: node '{node_id}' is not in the elements."
                )
            if node_id in active_path:
                raise ValueError(
                    f"Invalid hierarchy: cycle detected at node '{node_id}'."
                )
            active_path.add(node_id)
            info = elem_lookup[node_id]
            child_ids = children_map.get(node_id, [])
            children_nodes = [build_node(cid) for cid in child_ids]
            active_path.remove(node_id)
            return {
                "element_id": info.get("element_id", info.get("raw_id", node_id)),
                "kind": info["kind"],
                "content": info["content"],
                "children": children_nodes,
            }

        return {
            "arxiv_id": arxiv_id,
            "version": v_num,
            "version_name": version_name,
            "status": "success",
            "is_empty": False,
            "elements_count": len(elements),
            "edges_count": len(hierarchy_edges),
            "root": build_node(root_id),
        }

    @classmethod
    def load_version_tex_from_elements_or_source(
        cls,
        pub_dir: Path,
        version_name: str,
        metadata: Optional[dict] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Read the inlined LaTeX of a version from `<pub_dir>/elements/`.
        When it is missing, gather it from `<pub_dir>/<version_name>` with MultiFileGatherer.
        """
        elements_dir = pub_dir / "elements"
        candidates = [
            elements_dir / f"full_inlined_{version_name}.tex",
            elements_dir / f"{version_name}.tex",
            elements_dir / version_name / "main.tex",
        ]
        for cand in candidates:
            if cand.is_file():
                try:
                    content = cand.read_text(encoding="utf-8", errors="ignore")
                    if content.strip():
                        return content, None
                except Exception:
                    pass

        elem_v_dir = elements_dir / version_name
        if elem_v_dir.is_dir():
            g_res = MultiFileGatherer.gather(elem_v_dir, metadata=metadata)
            if g_res["success"] and g_res["inlined_content"].strip():
                return g_res["inlined_content"], g_res

        version_dir = pub_dir / version_name
        if version_dir.is_dir():
            g_res = MultiFileGatherer.gather(version_dir, metadata=metadata)
            if g_res["success"] and g_res["inlined_content"].strip():
                return g_res["inlined_content"], g_res
            return None, g_res

        return None, None