"""
Module xây dựng cây phân cấp (Hierarchy Construction).
Đọc dữ liệu LaTeX từ folder `elements/` và dựng cây phân cấp cho từng version
theo đúng quy tắc Parsing Notes của Milestone 2:
1. Giữ nguyên nội dung LaTeX thô (Raw Content) ở các nút để ông Khoa làm sạch/chuẩn hóa cho tiện nhé.
2. Loại bỏ References khỏi cây nhưng lưu riêng vào `raw_references.json` cho Khoa tách BibTeX.
3. Nút gốc có ID `<arxiv_id>#root1`, chỉ đóng vai trò `parent_id` (đảm bảo `parent_id` không null).
4. Ghi nhận tường minh các version rỗng (Empty Records) khi không có file .tex.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .gatherer import MultiFileGatherer


class HierarchyBuilder:
    """Xây dựng cây phân cấp cho từng version của bài báo từ mã nguồn LaTeX."""

    # Cấp độ phân cấp của các tiêu đề mục
    HEADING_LEVELS: Dict[str, int] = {
        "part": 1,
        "chapter": 1,
        "section": 2,
        "subsection": 3,
        "subsubsection": 4,
        "paragraph": 5,
        "subparagraph": 6,
    }

    # Các tiêu đề mục tài liệu tham khảo cần loại khỏi cây
    REFERENCE_TITLES = {
        "references",
        "reference",
        "bibliography",
        "works cited",
        "literature cited",
        "references and notes",
    }

    # Các từ viết tắt có dấu chấm nhưng không phải kết thúc câu
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
        r"\b(?:Fig|Figs|Tab|Tabs|Eq|Eqs|Sec|Secs|Ref|Refs|Ch|Chap|App|Vol|No|pp|Dr|Prof|Mr|Mrs|Ms|St|Jr|Sr|Inc|Ltd|Univ)\.(?:\s*(?=[0-9\(\[]))?",
    ]

    @classmethod
    def strip_comments(cls, text: str) -> str:
        """Xóa comment LaTeX (% ...) nhưng giữ lại ký tự \\% hợp lệ."""
        return re.sub(r"(?<!\\)(?:\\\\)*%.*$", "", text, flags=re.MULTILINE)

    @classmethod
    def normalize_whitespace_only(cls, text: str) -> str:
        """Gộp khoảng trắng thừa nhưng giữ nguyên 100% lệnh LaTeX và công thức toán gốc."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
        return text.strip()

    @classmethod
    def extract_body_and_preprocess(
        cls, raw_tex: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Tách phần thân bài (\\begin{document}...\\end{document}),
        lưu riêng khối References cho Khoa trước khi xóa khỏi cây,
        chuyển Acknowledgements/Appendix về dạng section, và bỏ các lệnh tiêu đề đầu bài.
        """
        text = cls.strip_comments(raw_tex)

        # Lấy nội dung giữa \begin{document} và \end{document}
        doc_match = re.search(
            r"\\begin\{document\}(.*?)(?:\\end\{document\}|$)",
            text,
            flags=re.DOTALL,
        )
        body = doc_match.group(1) if doc_match else text

        preserved_bib_envs: List[str] = []
        preserved_bib_commands: List[str] = []
        preserved_bibitems: List[str] = []

        # 1. Lưu riêng khối thebibliography và \bibitem cho Khoa rồi xóa khỏi thân bài
        def _capture_bib_env(m: re.Match) -> str:
            block = m.group(0).strip()
            preserved_bib_envs.append(block)
            # Cắt bỏ thẻ đóng \end{thebibliography} trước khi tách các \bibitem
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

        # Lưu lại các lệnh \bibliography{...} và \printbibliography
        for m_bib in re.finditer(r"\\bibliography(?:style)?\s*\{[^{}]*\}|\\printbibliography(?:\s*\[[^\]]*\])?", body):
            preserved_bib_commands.append(m_bib.group(0).strip())

        body = re.sub(r"\\bibliography(?:style)?\s*\{[^{}]*\}", "", body)
        body = re.sub(r"\\printbibliography(?:\s*\[[^\]]*\])?", "", body)

        # 2. Chuyển môi trường acknowledgments/abstract/appendix thành \section* hoặc \appendix để đưa vào cây
        body = re.sub(
            r"\\begin\{acknowledg(?:e)?ments\*?\}(.*?)\\end\{acknowledg(?:e)?ments\*?\}",
            r"\n\\section*{Acknowledgements}\n\1\n",
            body,
            flags=re.DOTALL | re.IGNORECASE,
        )
        body = re.sub(
            r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
            r"\n\\section*{Abstract}\n\1\n",
            body,
            flags=re.DOTALL | re.IGNORECASE,
        )

        def _convert_appendix_env(m: re.Match) -> str:
            inner_app = m.group(1)
            # Nếu bên trong môi trường appendix đã có sẵn \section hoặc \chapter thì dùng cờ \appendix
            if re.search(r"\\(?:chapter|section)\*?\s*(?:\[[^\]]*\])?\{", inner_app):
                return f"\n\\appendix\n{inner_app}\n"
            return f"\n\\section*{{Appendix}}\n{inner_app}\n"

        body = re.sub(
            r"\\begin\{appendi(?:x|ces)\}(.*?)\\end\{appendi(?:x|ces)\}",
            _convert_appendix_env,
            body,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # 3. Bỏ các lệnh metadata đầu bài (title, author, affiliation, email, date...)
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

    @classmethod
    def split_sentences_safe(cls, text: str) -> List[str]:
        """
        Tách văn bản thành các câu (sentence) bằng kỹ thuật che token (masking):
        Không tách câu bên trong toán inline, lệnh LaTeX, URL, số thập phân và từ viết tắt.
        Sau khi tách xong, hoàn trả (unmask) lại đúng chuỗi LaTeX gốc cho Khoa xử lý.
        """
        if not text or not text.strip():
            return []

        protected_tokens: Dict[str, str] = {}

        def mask(match: re.Match, prefix: str) -> str:
            key = f"__{prefix}_{len(protected_tokens)}__"
            protected_tokens[key] = match.group(0)
            return key

        work = text

        # 1. Che công thức toán inline: \(...\) và $...$ (bỏ qua ký tự \$ đã escape)
        work = re.sub(r"\\\((.*?)\\\)", lambda m: mask(m, "INLMATH"), work, flags=re.DOTALL)
        work = re.sub(r"(?<!\\)\$(?:\\.|[^$\\])+(?<!\\)\$", lambda m: mask(m, "MATH"), work)

        # 2. Che đường dẫn URL (không ăn dấu câu kết thúc câu như .,;:!?) ở cuối URL)
        work = re.sub(r"https?://[^\s\}]+(?<![.,;:!?\)\]])", lambda m: mask(m, "URL"), work)

        # 3. Che các lệnh LaTeX có tham số (ví dụ: \cite{...}, \ref{fig.1}, \href{...}{...})
        work = re.sub(
            r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{(?:[^{}]|\{[^{}]*\})*\})+",
            lambda m: mask(m, "CMD"),
            work,
        )

        # 4. Che dấu chấm trong số thập phân (ví dụ: 3.14)
        work = re.sub(r"(\d+)\.(\d+)", r"\1__DEC_DOT__\2", work)

        # 5. Che các từ viết tắt học thuật (i.e., e.g., et al., Fig. 1...)
        for abbr_pat in cls.ABBREVIATION_PATTERNS:
            work = re.sub(
                abbr_pat,
                lambda m: mask(m, "ABBR"),
                work,
                flags=re.IGNORECASE,
            )

        # 6. Tách câu tại dấu [.!?] (kể cả khi theo sau dấu chấm là dấu đóng ngoặc/nháy hoặc lệnh \footnote/\cite)
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

            # Khôi phục dấu chấm số thập phân
            s = s.replace("__DEC_DOT__", ".")

            # Khôi phục toàn bộ token về chuỗi LaTeX gốc
            for key in reversed(list(protected_tokens.keys())):
                if key in s:
                    s = s.replace(key, protected_tokens[key])

            raw_sentence = cls.normalize_whitespace_only(s)
            if len(raw_sentence) > 2 and any(c.isalnum() for c in raw_sentence):
                sentences.append(raw_sentence)

        return sentences

    @classmethod
    def _extract_top_level_blocks(cls, text: str) -> List[Tuple[str, str]]:
        """
        Tách văn bản thành các khối cấu trúc theo thứ tự xuất hiện:
        heading, appendix_marker, block_formula, figure (gồm cả table), list, và prose.
        """
        TOKEN_FINDER = re.compile(
            r"(?P<heading>\\(?:part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?(?:\[[^\]]*\])?\s*\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})|"
            r"(?P<appendix>\\appendix\b)|"
            r"(?P<formula_env>\\begin\{(?P<f_env>equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?|flalign\*?|displaymath)\}.*?\\end\{(?P=f_env)\})|"
            r"(?P<formula_dollar>(?<!\\)\$\$.*?(?<!\\)\$\$)|"
            r"(?P<formula_bracket>\\\[.*?\\\])|"
            r"(?P<figure>\\begin\{(?P<fig_env>figure\*?|table\*?|algorithm\*?)\}.*?\\end\{(?P=fig_env)\})|"
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
                # Đếm độ sâu đóng/mở để bắt đúng khối danh sách lồng nhau (nested list)
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
        """Tách các mục \\item cùng cấp trong danh sách (không cắt nhầm \\item của danh sách con)."""
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

    @classmethod
    def parse_version(
        cls,
        arxiv_id: str,
        version: str,
        inlined_tex: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Phân tích 1 version thành:
        - elements: Danh sách các phần tử (giữ nguyên Raw Content).
        - hierarchy: Danh sách các cạnh cha-con (nút gốc chỉ làm parent, không có dòng parent=None).
        - preserved_refs: Khối tài liệu tham khảo lưu riêng cho Khoa.
        """
        v_num = int(version[1:]) if version.startswith("v") and version[1:].isdigit() else int(version)
        body_tex, preserved_refs = cls.extract_body_and_preprocess(inlined_tex)

        elements: List[Dict[str, Any]] = []
        hierarchy: List[Dict[str, Any]] = []

        counter = 0

        def next_id(prefix: str) -> str:
            nonlocal counter
            counter += 1
            return f"{prefix}_{counter}"

        # 1. Tạo nút gốc Document (root1, root2...): chỉ lưu vào bảng elements
        doc_root_id = f"root{v_num}"
        elements.append({
            "raw_id": doc_root_id,
            "kind": "document",
            "content": f"Version {v_num} root",
        })

        # Ngăn xếp (stack) lưu (cấp_độ, id_phần_tử) để tự động xác định cha-con
        heading_stack: List[Tuple[int, str]] = [(0, doc_root_id)]
        in_appendix = False
        skip_until_level: Optional[int] = None

        def get_current_parent() -> str:
            return heading_stack[-1][1]

        def parse_content_into_parent(content_str: str, parent_id: str) -> None:
            """Tách câu, công thức, hình/bảng và danh sách con gắn vào parent_id."""
            sub_blocks = cls._extract_top_level_blocks(content_str)
            for b_type, b_text in sub_blocks:
                if b_type == "block_formula":
                    raw_f = cls.normalize_whitespace_only(b_text)
                    if raw_f:
                        f_id = next_id("formula")
                        elements.append({
                            "raw_id": f_id,
                            "kind": "block_formula",
                            "content": raw_f,
                        })
                        hierarchy.append({
                            "arxiv_id": arxiv_id,
                            "version": v_num,
                            "element_raw_id": f_id,
                            "parent_raw_id": parent_id,
                        })
                elif b_type == "figure":
                    raw_fig = cls.normalize_whitespace_only(b_text)
                    if raw_fig:
                        fig_id = next_id("figure")
                        elements.append({
                            "raw_id": fig_id,
                            "kind": "figure",
                            "content": raw_fig,
                        })
                        hierarchy.append({
                            "arxiv_id": arxiv_id,
                            "version": v_num,
                            "element_raw_id": fig_id,
                            "parent_raw_id": parent_id,
                        })
                elif b_type == "list":
                    process_list_block(b_text, parent_id)
                elif b_type in ("prose", "heading"):
                    for sent in cls.split_sentences_safe(b_text):
                        s_id = next_id("s")
                        elements.append({
                            "raw_id": s_id,
                            "kind": "sentence",
                            "content": sent,
                        })
                        hierarchy.append({
                            "arxiv_id": arxiv_id,
                            "version": v_num,
                            "element_raw_id": s_id,
                            "parent_raw_id": parent_id,
                        })

        def process_list_block(list_text: str, parent_id: str) -> None:
            """Xử lý khối itemize/enumerate làm nhánh cha, từng \\item làm nhánh trung gian và tạo các sentence leaf bên dưới."""
            env_name, items = cls._split_list_items(list_text)
            list_id = next_id("itemize")
            elements.append({
                "raw_id": list_id,
                "kind": "itemize",
                "content": cls.normalize_whitespace_only(list_text),
            })
            hierarchy.append({
                "arxiv_id": arxiv_id,
                "version": v_num,
                "element_raw_id": list_id,
                "parent_raw_id": parent_id,
            })

            for item_raw in items:
                item_clean = cls.normalize_whitespace_only(item_raw)
                if not item_clean:
                    continue

                item_id = next_id("item")
                elements.append({
                    "raw_id": item_id,
                    "kind": "item",
                    "content": f"\\item {item_clean}",
                })
                hierarchy.append({
                    "arxiv_id": arxiv_id,
                    "version": v_num,
                    "element_raw_id": item_id,
                    "parent_raw_id": list_id,
                })

                # Luôn tách nội dung bên trong \item thành các nút lá (sentence, block_formula, figure) hoặc list con
                before_len = len(elements)
                parse_content_into_parent(item_raw, item_id)
                if len(elements) == before_len:
                    s_id = next_id("s")
                    elements.append({
                        "raw_id": s_id,
                        "kind": "sentence",
                        "content": item_clean,
                    })
                    hierarchy.append({
                        "arxiv_id": arxiv_id,
                        "version": v_num,
                        "element_raw_id": s_id,
                        "parent_raw_id": item_id,
                    })

        top_blocks = cls._extract_top_level_blocks(body_tex)

        HEADING_PARSE_RE = re.compile(
            r"^\\(?P<type>part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?(?:\[[^\]]*\])?\s*\{(?P<title>.*)\}$",
            re.DOTALL,
        )

        for b_type, b_text in top_blocks:
            if b_type == "appendix_marker":
                in_appendix = True
                skip_until_level = None
                continue

            if b_type == "heading":
                hm = HEADING_PARSE_RE.match(b_text.strip())
                if not hm:
                    continue
                h_type = hm.group("type").lower()
                raw_title = cls.normalize_whitespace_only(hm.group("title"))
                if not raw_title:
                    # Bỏ qua tiêu đề rỗng (ví dụ: \paragraph{} dùng để ngắt dòng)
                    continue
                h_level = cls.HEADING_LEVELS.get(h_type, 2)

                # Chuẩn hóa tên tiêu đề tạm thời để kiểm tra xem có phải mục References không
                norm_title = re.sub(r"\\[a-zA-Z]+\*?", "", raw_title)
                norm_title = re.sub(r"[^a-zA-Z\s]", "", norm_title).strip().lower()
                norm_title = re.sub(r"\s+", " ", norm_title)

                if skip_until_level is not None:
                    if h_level <= skip_until_level:
                        skip_until_level = None
                    else:
                        # Mục con nằm trong References -> lưu cho Khoa và bỏ qua khỏi cây
                        preserved_refs["reference_sections"].append(b_text)
                        continue

                # Nếu gặp mục References / Bibliography -> lưu cho Khoa và không đưa vào cây
                if norm_title in cls.REFERENCE_TITLES:
                    skip_until_level = h_level
                    preserved_refs["reference_sections"].append(b_text)
                    continue

                # Nếu gặp tiêu đề bao ngoài là "Appendix" hoặc "Appendices", đặt cấp = 1 để làm cha cho các \section phụ lục bên dưới
                if norm_title in ("appendix", "appendices") and h_type == "section":
                    h_level = 1
                    in_appendix = True
                elif in_appendix and h_type in ("chapter", "section") and not raw_title.lower().startswith("appendix"):
                    raw_title = f"Appendix: {raw_title}"

                # Rút các mục cùng cấp hoặc cấp thấp hơn ra khỏi stack để tìm đúng nút cha
                while len(heading_stack) > 1 and heading_stack[-1][0] >= h_level:
                    heading_stack.pop()

                parent_id = get_current_parent()
                h_id = next_id(h_type)

                elements.append({
                    "raw_id": h_id,
                    "kind": h_type,
                    "content": raw_title,
                })
                hierarchy.append({
                    "arxiv_id": arxiv_id,
                    "version": v_num,
                    "element_raw_id": h_id,
                    "parent_raw_id": parent_id,
                })
                heading_stack.append((h_level, h_id))
                continue

            # Nếu đang nằm trong mục References thì lưu nội dung cho Khoa và bỏ qua khỏi cây
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

            # Gắn các phần tử (công thức, hình/bảng, danh sách, câu) vào tiêu đề cha hiện tại
            curr_parent = get_current_parent()
            if b_type == "block_formula":
                raw_f = cls.normalize_whitespace_only(b_text)
                if raw_f:
                    f_id = next_id("formula")
                    elements.append({
                        "raw_id": f_id,
                        "kind": "block_formula",
                        "content": raw_f,
                    })
                    hierarchy.append({
                        "arxiv_id": arxiv_id,
                        "version": v_num,
                        "element_raw_id": f_id,
                        "parent_raw_id": curr_parent,
                    })
            elif b_type == "figure":
                raw_fig = cls.normalize_whitespace_only(b_text)
                if raw_fig:
                    fig_id = next_id("figure")
                    elements.append({
                        "raw_id": fig_id,
                        "kind": "figure",
                        "content": raw_fig,
                    })
                    hierarchy.append({
                        "arxiv_id": arxiv_id,
                        "version": v_num,
                        "element_raw_id": fig_id,
                        "parent_raw_id": curr_parent,
                    })
            elif b_type == "list":
                process_list_block(b_text, curr_parent)
            elif b_type == "prose":
                for sent in cls.split_sentences_safe(b_text):
                    s_id = next_id("s")
                    elements.append({
                        "raw_id": s_id,
                        "kind": "sentence",
                        "content": sent,
                    })
                    hierarchy.append({
                        "arxiv_id": arxiv_id,
                        "version": v_num,
                        "element_raw_id": s_id,
                        "parent_raw_id": curr_parent,
                    })

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
        Chuẩn hóa ID cho từng version (`<arxiv_id>#root1`, `<arxiv_id>#v1_s1`...)
        và tạo cấu trúc cây lồng nhau. Phân biệt rõ giữa version rỗng (empty_version)
        và version bị lỗi gather/parse (failed).
        """
        v_num = int(version_name[1:]) if version_name.startswith("v") and version_name[1:].isdigit() else int(version_name)

        if status == "failed":
            return {
                "arxiv_id": arxiv_id,
                "version": v_num,
                "version_name": version_name,
                "status": "failed",
                "is_empty": False,
                "error": error_msg or "Lỗi gather/parse mã nguồn TeX cho version này",
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
                "error": error_msg or "Version không có mã nguồn TeX (empty record)",
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
                continue  # Đảm bảo parent_id không bao giờ bị null
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

        return {
            "arxiv_id": arxiv_id,
            "version": v_num,
            "version_name": version_name,
            "status": "success",
            "is_empty": False,
            "error": None,
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
        """Dựng cây phân cấp dạng JSON lồng nhau (nested tree) cho 1 version."""
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
                f"Dữ liệu hierarchy không hợp lệ: Không tìm thấy nút gốc (root) cho {arxiv_id} version {version_name} trong bảng elements."
            )

        children_map: Dict[str, List[str]] = {}
        for edge in hierarchy_edges:
            edge_v = edge.get("version")
            if edge_v == v_num or edge_v == version_name:
                p_id = edge.get("parent_id") or edge.get("parent_raw_id")
                c_id = edge.get("element_id") or edge.get("element_raw_id")
                if not p_id or not c_id:
                    raise ValueError(
                        f"Dữ liệu hierarchy không hợp lệ: Cạnh thiếu parent_id hoặc element_id ({edge})."
                    )
                if p_id not in elem_lookup:
                    raise ValueError(
                        f"Dữ liệu hierarchy không hợp lệ: parent_id '{p_id}' không tồn tại trong bảng elements."
                    )
                if c_id not in elem_lookup:
                    raise ValueError(
                        f"Dữ liệu hierarchy không hợp lệ: element_id '{c_id}' không tồn tại trong bảng elements."
                    )
                children_map.setdefault(p_id, []).append(c_id)

        active_path: set[str] = set()

        def build_node(node_id: str) -> Dict[str, Any]:
            if node_id not in elem_lookup:
                raise ValueError(
                    f"Dữ liệu hierarchy không hợp lệ: node '{node_id}' không tồn tại trong bảng elements."
                )
            if node_id in active_path:
                raise ValueError(
                    f"Dữ liệu hierarchy không hợp lệ: Phát hiện chu trình (cycle) tại node '{node_id}'."
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
        Đọc file LaTeX đã gom của version từ folder `<pub_dir>/elements/`.
        Nếu chưa có sẵn trong `elements/`, tự động gọi MultiFileGatherer để gom từ `<pub_dir>/<version_name>`.
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