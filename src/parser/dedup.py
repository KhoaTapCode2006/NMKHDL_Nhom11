"""
Module Khử Trùng Lặp (Deduplication - Phần của Kiệt) cho Milestone 2.
Thực hiện khử trùng lặp ở 2 cấp độ xuyên suốt các version của từng publication:
1. Reference Entries (BibTeX Deduplication):
   - Phát hiện các mục tài liệu tham khảo trùng lặp trong cùng version hoặc xuyên version
     (bằng DOI chuẩn hóa, Title chuẩn hóa + kiểm tra tác giả/năm, hoặc Citation Key).
   - Chọn 1 citation_key đại diện (canonical key) và hợp nhất (union) toàn bộ các trường
     (entry_type, title, authors, year, doi, journal, url) để không mất thông tin.
   - Đổi tên mọi lệnh trích dẫn \\cite{...} (và các biến thể \\citep, \\citet...) trong
     nội dung các elements sang citation_key đại diện trước khi khử trùng lặp Full-text.
2. Full-text Content (Elements & Hierarchy Deduplication):
   - Khử trùng lặp các phần tử văn bản (elements) sau khi đã làm sạch và đồng bộ \\cite.
   - Bảo toàn cấu trúc cây phân cấp (không gộp nhầm các khối itemize khác nội dung hoặc
     các tiêu đề trùng tên trong cùng 1 version; mỗi version giữ nút gốc #root riêng).
   - Ánh xạ lại bảng cạnh hierarchy và cập nhật trực tiếp vào cây hierarchy_tree.json
     của từng version (trong phần của Tân) cũng như thư mục bibtex/ (trong phần của Khoa).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .extractor import HierarchyExtractor
from .hierarchy_builder import HierarchyBuilder


class ReferenceDeduplicator:
    """
    Khử trùng lặp tài liệu tham khảo (BibTeX / \\bibitem) xuyên version,
    hợp nhất trường thông tin (field unionization) và đổi tên lệnh \\cite{...}.
    """

    # Thứ tự ưu tiên của entry_type khi hợp nhất (loại cụ thể ưu tiên hơn loại chung)
    ENTRY_TYPE_PRIORITY: Dict[str, int] = {
        "article": 10,
        "inproceedings": 10,
        "conference": 9,
        "book": 9,
        "incollection": 8,
        "inbook": 8,
        "phdthesis": 8,
        "mastersthesis": 7,
        "proceedings": 6,
        "techreport": 5,
        "manual": 4,
        " booklet": 4,
        "unpublished": 3,
        "online": 2,
        "misc": 1,
    }

    # Tất cả các lệnh trích dẫn LaTeX hợp lệ cần cập nhật citation_key
    # (Bỏ qua các lệnh cấu hình như \setcitestyle, \citestyle, \defcitealias, \bstctlcite)
    CITE_COMMAND_PATTERN = re.compile(
        r"(\\(?:cite|citep|citet|citealt|citealp|citeauthor|citeyear|"
        r"citepalias|citetalias|parencite|textcite|autocite|nocite|hyperlinkcite)"
        r"\*?(?:\s*\[[^\]]*\])*\s*\{)([^}]+)(\})"
    )

    # Bảng chuyển đổi dấu câu/ký tự đặc biệt LaTeX phổ biến trong tiêu đề và tên tác giả
    LATEX_ACCENT_PATTERNS: List[Tuple[str, str]] = [
        (r"\\[\"'`\^~=.uvHcdb]\s*\{?\s*([a-zA-Z])\s*\}?", r"\1"),
        (r"\\(?:aa|AA)\b", "a"),
        (r"\\(?:ae|AE)\b", "ae"),
        (r"\\(?:oe|OE)\b", "oe"),
        (r"\\(?:o|O)\b", "o"),
        (r"\\(?:l|L)\b", "l"),
        (r"\\(?:ss)\b", "ss"),
    ]

    @classmethod
    def strip_latex_accents_and_braces(cls, text: str) -> str:
        """Loại bỏ dấu LaTeX (ví dụ \\\"{a} -> a), lệnh định dạng và ngoặc nhọn."""
        if not text:
            return ""
        work = text
        for pat, repl in cls.LATEX_ACCENT_PATTERNS:
            work = re.sub(pat, repl, work)
        # Bỏ các lệnh định dạng chữ như \emph, \textit, \textbf, \textrm, \textsc
        work = re.sub(r"\\(?:emph|textit|textbf|textrm|textsc|textsf|texttt|mathrm|mathbf)\b", "", work)
        work = work.replace("{", "").replace("}", "")
        work = re.sub(r"\s+", " ", work).strip()
        return work

    @classmethod
    def normalize_doi(cls, doi: Optional[str]) -> str:
        """Chuẩn hóa DOI về dạng chữ thường, bỏ tiền tố URL doi.org."""
        if not doi:
            return ""
        d = str(doi).strip().lower()
        d = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", d)
        d = re.sub(r"^doi:\s*", "", d)
        d = d.strip().strip(".").strip(",").strip("}")
        # Kiểm tra cấu trúc DOI cơ bản (10.xxxx/...)
        if d.startswith("10.") and "/" in d:
            return d
        return ""

    @classmethod
    def normalize_title_key(cls, title: Optional[str]) -> str:
        """
        Chuẩn hóa tiêu đề bài báo về chuỗi chữ-số viết thường [a-z0-9]
        để so khớp chính xác giữa các version và giữa các cách gõ LaTeX khác nhau.
        """
        if not title:
            return ""
        clean = cls.strip_latex_accents_and_braces(title)
        # Bỏ các ký tự toán inline đơn giản nhưng giữ chữ cái bên trong
        clean = clean.replace("$", "")
        clean = re.sub(r"\\[a-zA-Z]+", " ", clean)
        norm = re.sub(r"[^a-z0-9]", "", clean.lower())
        return norm

    # Các tiêu đề mẫu/placeholder không được dùng để gộp tự động nếu khác citation_key
    PLACEHOLDER_NORM_TITLES: Set[str] = {
        "articletitle",
        "manuscriptinpreparation",
        "inpreparation",
        "toappear",
        "technicalreport",
        "preprint",
        "submitted",
        "unpublished",
        "untitled",
    }

    @classmethod
    def clean_display_title(cls, title: Optional[str]) -> str:
        """Làm sạch tiêu đề để lưu vào bảng bibtex chuẩn."""
        if not title:
            return ""
        t = cls.strip_latex_accents_and_braces(str(title))
        t = re.sub(r"\s+", " ", t).strip()
        # Bỏ dấu chấm dư ở cuối tiêu đề nếu có
        if t.endswith(".") and not t.endswith("..."):
            t = t[:-1].strip()
        return t

    @classmethod
    def parse_authors_list(cls, raw_authors: Any) -> List[str]:
        """Chuyển đổi trường author (chuỗi hoặc list) sang danh sách các tên tác giả sạch."""
        if not raw_authors:
            return []
        if isinstance(raw_authors, list):
            result = []
            for a in raw_authors:
                a_clean = cls.strip_latex_accents_and_braces(str(a)).strip(" ,;")
                if a_clean and a_clean.lower() not in ("others", "et al", "et al."):
                    result.append(a_clean)
            return result

        auth_str = str(raw_authors).strip()
        if not auth_str:
            return []

        # Tách theo từ khóa ' and ' chuẩn của BibTeX trước
        if re.search(r"\band\b", auth_str, flags=re.IGNORECASE):
            parts = re.split(r"\s+\band\b\s+", auth_str, flags=re.IGNORECASE)
        elif ";" in auth_str:
            parts = auth_str.split(";")
        else:
            # Nếu định dạng từ \bibitem như "A. Vaswani, N. Shazeer, ..."
            parts = [auth_str]

        cleaned_list: List[str] = []
        for p in parts:
            c = cls.strip_latex_accents_and_braces(p).strip(" ,;.")
            if not c:
                continue
            if c.lower() in ("others", "et al", "et al."):
                continue
            cleaned_list.append(c)
        return cleaned_list

    @classmethod
    def extract_author_surnames(cls, authors: List[str]) -> Set[str]:
        """Trích xuất tập hợp họ (surname) viết thường của các tác giả để kiểm tra chống gộp nhầm."""
        surnames: Set[str] = set()
        for auth in authors:
            a = auth.strip()
            if not a:
                continue
            if "," in a:
                # Dạng "Last, First"
                surname = a.split(",", 1)[0].strip()
            else:
                # Dạng "First Last"
                tokens = a.split()
                surname = tokens[-1] if tokens else ""
            norm_s = re.sub(r"[^a-z]", "", surname.lower())
            if len(norm_s) >= 2:
                surnames.add(norm_s)
        return surnames

    @classmethod
    def parse_year_int(cls, raw_year: Any) -> Optional[int]:
        """Chuyển đổi năm xuất bản sang kiểu int 4 chữ số (1800..2035) hoặc None."""
        if raw_year is None:
            return None
        if isinstance(raw_year, int):
            return raw_year if 1800 <= raw_year <= 2035 else None
        m = re.search(r"\b(18\d{2}|19\d{2}|20[0-3]\d)\b", str(raw_year))
        if m:
            return int(m.group(1))
        return None

    @classmethod
    def _extract_balanced_value(cls, text: str, start_pos: int) -> Tuple[str, int]:
        """
        Đọc giá trị của một trường trong file .bib bắt đầu từ vị trí start_pos
        (hỗ trợ giá trị bọc trong {...} có ngoặc nhọn lồng nhau, "..." hoặc chuỗi trần).
        """
        n = len(text)
        pos = start_pos
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            return "", pos

        ch = text[pos]
        if ch == "{":
            depth = 0
            val_start = pos + 1
            i = pos
            while i < n:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[val_start:i], i + 1
                i += 1
            return text[val_start:], n
        elif ch == '"':
            val_start = pos + 1
            i = pos + 1
            while i < n:
                if text[i] == '"' and text[i - 1] != "\\":
                    return text[val_start:i], i + 1
                i += 1
            return text[val_start:], n
        else:
            # Giá trị số hoặc macro không có ngoặc (ví dụ: year = 2022, month = apr)
            i = pos
            while i < n and text[i] not in (",", "}", "\n", "\r"):
                i += 1
            return text[pos:i].strip(), i

    @classmethod
    def parse_bib_content(
        cls, bib_text: str, arxiv_id: str, version: str = "v1"
    ) -> List[Dict[str, Any]]:
        """
        Phân tích nội dung file .bib thành danh sách các entry chuẩn,
        hỗ trợ ngoặc nhọn lồng nhau (nested braces) không bị cắt cụt tại \\\"{a}.
        """
        if not bib_text:
            return []

        # Xóa comment dòng bắt đầu bằng %
        clean_text = re.sub(r"(?<!\\)%.*$", "", bib_text, flags=re.MULTILINE)
        entries: List[Dict[str, Any]] = []

        entry_header_re = re.compile(r"@([a-zA-Z]+)\s*\{\s*([^,\s\}]+)\s*,")
        pos = 0
        n = len(clean_text)

        while pos < n:
            m = entry_header_re.search(clean_text, pos)
            if not m:
                break

            entry_type = m.group(1).strip().lower()
            citation_key = m.group(2).strip()

            # Tìm dấu đóng '}' của cả khối @entry{...}
            open_brace_idx = clean_text.find("{", m.start())
            if open_brace_idx == -1:
                pos = m.end()
                continue

            depth = 0
            i = open_brace_idx
            entry_end = n
            while i < n:
                if clean_text[i] == "{":
                    depth += 1
                elif clean_text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        entry_end = i
                        break
                i += 1

            if entry_type in ("string", "preamble", "comment"):
                pos = entry_end + 1
                continue

            body = clean_text[m.end() : entry_end]
            fields: Dict[str, str] = {}

            field_re = re.compile(r"([a-zA-Z0-9_\-]+)\s*=\s*")
            f_pos = 0
            body_len = len(body)
            while f_pos < body_len:
                fm = field_re.search(body, f_pos)
                if not fm:
                    break
                f_name = fm.group(1).strip().lower()
                val, next_pos = cls._extract_balanced_value(body, fm.end())
                fields[f_name] = val.strip()
                f_pos = next_pos

            title = cls.clean_display_title(fields.get("title", ""))
            authors = cls.parse_authors_list(fields.get("author", ""))
            year = cls.parse_year_int(fields.get("year") or fields.get("date"))
            doi = cls.normalize_doi(fields.get("doi", ""))
            journal = cls.clean_display_title(
                fields.get("journal")
                or fields.get("journaltitle")
                or fields.get("booktitle")
                or fields.get("series")
                or fields.get("publisher")
                or ""
            )
            url = fields.get("url", "").strip()
            if not url and fields.get("eprint"):
                eprint = fields["eprint"].strip()
                if re.match(r"^\d{4}\.\d{4,5}", eprint):
                    url = f"https://arxiv.org/abs/{eprint}"

            entries.append({
                "arxiv_id": arxiv_id,
                "version": version,
                "citation_key": citation_key,
                "entry_type": entry_type,
                "title": title,
                "authors": authors,
                "year": year,
                "doi": doi,
                "journal": journal,
                "url": url,
            })

            pos = entry_end + 1

        return entries

    @classmethod
    def parse_bibitem_entries(
        cls, bibitems: List[str], arxiv_id: str, version: str = "v1"
    ) -> List[Dict[str, Any]]:
        """
        Trích xuất các mục \\bibitem từ file .tex (hoặc raw_references.json của Tân)
        sang cấu trúc BibTeX chuẩn bằng biểu thức chính quy (Regular Expressions).
        """
        entries: List[Dict[str, Any]] = []
        bibitem_re = re.compile(
            r"\\bibitem(?:\s*\[[^\]]*\])?\s*\{([^}]+)\}(.*)",
            re.DOTALL,
        )

        for raw_item in bibitems:
            m = bibitem_re.search(raw_item.strip())
            if not m:
                continue
            citation_key = m.group(1).strip()
            content = m.group(2).strip()

            # Xóa comment và chuẩn hóa khoảng trắng
            content = re.sub(r"(?<!\\)%.*$", "", content, flags=re.MULTILINE)
            content = re.sub(r"\s+", " ", content).strip()

            # Tìm URL và DOI nếu có
            url = ""
            url_m = re.search(r"\\url\{([^}]+)\}|https?://[^\s\}]+", content)
            if url_m:
                url = (url_m.group(1) or url_m.group(0)).strip(" .,;}")

            doi = ""
            doi_m = re.search(
                r"(?:doi[:\s]*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)",
                content,
                re.IGNORECASE,
            )
            if doi_m:
                doi = cls.normalize_doi(doi_m.group(1))

            year = cls.parse_year_int(content)

            # Cắt bỏ phần doi:... hoặc url ở cuối để không lẫn vào title/journal
            content_no_doi = re.sub(
                r"\bdoi[:\s]*10\.\d{4,9}/[-._;()/:A-Za-z0-9]+",
                "",
                content,
                flags=re.IGNORECASE,
            ).strip(" .,;")

            # Tách theo \newblock hoặc theo cấu trúc câu/in nghiêng (\emph, \textit, {\em ...})
            title = ""
            journal = ""
            authors_raw = ""

            if "\\newblock" in content_no_doi:
                blocks = [b.strip(" .,;") for b in content_no_doi.split("\\newblock") if b.strip()]
                if len(blocks) >= 1:
                    authors_raw = blocks[0]
                if len(blocks) >= 2:
                    title = blocks[1]
                if len(blocks) >= 3:
                    journal = " ".join(blocks[2:])
            else:
                # Tìm phần in nghiêng {\em ...}, {\it ...}, \emph{...}, \textit{...} hoặc chuỗi trong "..."
                quote_m = re.search(r"``([^']+)''|\"([^\"]+)\"", content_no_doi)
                emph_matches = list(
                    re.finditer(
                        r"\\(?:emph|textit)\{([^{}]+)\}|\{\\(?:em|it)\s+([^{}]+)\}",
                        content_no_doi,
                    )
                )
                # Kiểm tra định dạng thiên văn/vật lý: "Authors 2023, Journal, Vol, Page"
                astro_m = re.match(
                    r"^(.*?)(?:\\?\s+|\s*,\s*)(18\d{2}|19\d{2}|20[0-3]\d)[a-z]?\s*,\s*(.+)$",
                    content_no_doi,
                )
                if quote_m:
                    title = (quote_m.group(1) or quote_m.group(2)).strip()
                    authors_raw = content_no_doi[: quote_m.start()].strip(" .,;:")
                    rest = content_no_doi[quote_m.end() :].strip(" .,;:")
                    journal = rest
                elif emph_matches:
                    first_emph = emph_matches[0]
                    authors_raw = content_no_doi[: first_emph.start()].strip(" .,;:")
                    title = (first_emph.group(1) or first_emph.group(2)).strip()
                    journal = content_no_doi[first_emph.end() :].strip(" .,;:")
                elif astro_m:
                    authors_raw = astro_m.group(1).strip(" .,;:")
                    after_year = [p.strip() for p in astro_m.group(3).split(",") if p.strip()]
                    if len(after_year) >= 3 and len(after_year[1]) > 20:
                        journal = after_year[0]
                        title = after_year[1]
                    else:
                        journal = ", ".join(after_year)
                        title = ""
                else:
                    # Tách theo dấu chấm nhưng KHÔNG tách sau chữ viết tắt tên (ví dụ A. B. hoặc V.~V.)
                    parts = [
                        p.strip()
                        for p in re.split(r"(?<![A-Z~]\.)(?<![A-Z])\.\s+", content_no_doi)
                        if p.strip()
                    ]
                    if len(parts) >= 2:
                        authors_raw = parts[0]
                        title = parts[1]
                        journal = ". ".join(parts[2:])
                    elif len(parts) == 1:
                        title = parts[0]

            clean_title = cls.clean_display_title(cls.strip_latex_accents_and_braces(title))
            clean_journal = cls.clean_display_title(cls.strip_latex_accents_and_braces(journal))
            authors_list = cls.parse_authors_list(authors_raw)

            # Đoán entry_type từ nội dung
            lower_c = content.lower()
            if any(k in lower_c for k in ("proceedings", "conf.", "conference", "symposium", "workshop")):
                entry_type = "inproceedings"
            elif any(k in lower_c for k in ("press", "springer", "wiley", "edition", "book")):
                entry_type = "book"
            elif "thesis" in lower_c:
                entry_type = "phdthesis"
            elif clean_journal:
                entry_type = "article"
            else:
                entry_type = "misc"

            entries.append({
                "arxiv_id": arxiv_id,
                "version": version,
                "citation_key": citation_key,
                "entry_type": entry_type,
                "title": clean_title,
                "authors": authors_list,
                "year": year,
                "doi": doi,
                "journal": clean_journal,
                "url": url,
            })

        return entries

    @classmethod
    def standardize_entry_dict(
        cls, raw: Dict[str, Any], arxiv_id: str, default_version: str = "v1"
    ) -> Optional[Dict[str, Any]]:
        """Chuẩn hóa một dictionary entry bất kỳ (từ JSON/CSV của Khoa) về schema thống nhất."""
        citation_key = str(
            raw.get("citation_key") or raw.get("key") or raw.get("id") or ""
        ).strip()
        if not citation_key:
            return None

        entry_type = str(
            raw.get("entry_type") or raw.get("type") or raw.get("ENTRYTYPE") or "misc"
        ).strip().lower()

        title = cls.clean_display_title(raw.get("title") or "")
        authors = cls.parse_authors_list(raw.get("authors") if "authors" in raw else raw.get("author"))
        year = cls.parse_year_int(raw.get("year") or raw.get("date"))
        doi = cls.normalize_doi(raw.get("doi") or raw.get("DOI"))
        journal = cls.clean_display_title(
            raw.get("journal")
            or raw.get("booktitle")
            or raw.get("venue")
            or ""
        )
        url = str(raw.get("url") or raw.get("URL") or "").strip()
        version = str(raw.get("version") or default_version)

        return {
            "arxiv_id": arxiv_id,
            "version": version,
            "citation_key": citation_key,
            "entry_type": entry_type,
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
            "journal": journal,
            "url": url,
        }

    @classmethod
    def load_publication_references(
        cls, pub_dir: Path, version_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Tải toàn bộ danh sách Reference Entries của một bài báo qua các version.
        Hỗ trợ chế độ kép (Dual-mode):
        1. Nếu Khoa đã xuất sẵn dữ liệu trong `<pub_dir>/bibtex/` (dạng JSON, CSV hoặc .bib theo version),
           ưu tiên đọc dữ liệu từ `<pub_dir>/bibtex/` kết hợp với các nguồn theo version.
        2. Tự động đọc các file `.bib` trong `<pub_dir>/<v_name>/` và `elements/raw_references.json`
           (của Tân) hoặc `\bibitem` trong `.tex` để không bỏ sót bất kỳ tài liệu tham khảo nào.
        """
        pub_dir = Path(pub_dir).resolve()
        arxiv_id = pub_dir.name
        bibtex_dir = pub_dir / "bibtex"
        elements_dir = pub_dir / "elements"

        collected_entries: List[Dict[str, Any]] = []

        # 1. Kiểm tra nếu có các file JSON/CSV/BIB theo từng version trong <pub_dir>/bibtex/ (từ Khoa)
        loaded_versions_from_bibtex_dir: Set[str] = set()
        if bibtex_dir.is_dir():
            for v_name in version_names:
                for cand_name in (
                    f"bibtex_{v_name}.json",
                    f"references_{v_name}.json",
                    f"{v_name}.json",
                ):
                    cand_p = bibtex_dir / cand_name
                    if cand_p.is_file():
                        try:
                            data = json.loads(cand_p.read_text(encoding="utf-8"))
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict):
                                        std_item = cls.standardize_entry_dict(item, arxiv_id, v_name)
                                        if std_item:
                                            collected_entries.append(std_item)
                                loaded_versions_from_bibtex_dir.add(v_name)
                                break
                        except Exception:
                            pass

            # Kiểm tra nếu Khoa lưu file raw_bibtex.json hoặc bibtex_all.json chứa toàn bộ entries trước dedup
            for raw_all_name in ("raw_bibtex.json", "bibtex_raw.json", "all_references.json"):
                raw_all_p = bibtex_dir / raw_all_name
                if raw_all_p.is_file() and not collected_entries:
                    try:
                        data = json.loads(raw_all_p.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    std_item = cls.standardize_entry_dict(item, arxiv_id, "v1")
                                    if std_item:
                                        collected_entries.append(std_item)
                    except Exception:
                        pass

        # 2. Đọc `elements/raw_references.json` (do Tân tách sẵn cho Khoa)
        raw_refs_by_ver: Dict[str, Any] = {}
        raw_refs_path = elements_dir / "raw_references.json"
        if raw_refs_path.is_file():
            try:
                raw_refs_by_ver = json.loads(raw_refs_path.read_text(encoding="utf-8"))
            except Exception:
                raw_refs_by_ver = {}

        # 3. Với mỗi version chưa có trong loaded_versions_from_bibtex_dir, quét .bib và \bibitem
        for v_name in version_names:
            if v_name in loaded_versions_from_bibtex_dir:
                continue

            v_dir = pub_dir / v_name
            v_entries: List[Dict[str, Any]] = []

            # 3a. Quét tất cả file .bib trong thư mục version (bỏ qua file .bib mẫu của template nếu có file .bib thật)
            if v_dir.is_dir():
                bib_files = sorted(v_dir.glob("**/*.bib"))
                if len(bib_files) > 1:
                    non_sample = [
                        bf
                        for bf in bib_files
                        if not any(
                            ign in bf.name.lower()
                            for ign in ("sample-base", "ws-pro-sample", "acmart-sample")
                        )
                    ]
                    if non_sample:
                        bib_files = non_sample
                for bf in bib_files:
                    try:
                        bib_txt = bf.read_text(encoding="utf-8", errors="ignore")
                        v_entries.extend(cls.parse_bib_content(bib_txt, arxiv_id, v_name))
                    except Exception:
                        pass

            # 3b. Quét \bibitem từ raw_references.json của Tân hoặc từ full_inlined_vX.tex / .tex gốc
            bibitems_list: List[str] = []
            if v_name in raw_refs_by_ver and isinstance(raw_refs_by_ver[v_name], dict):
                bibitems_list = raw_refs_by_ver[v_name].get("bibitem_entries", [])

            if not bibitems_list:
                # Fallback tìm trong full_inlined_vX.tex hoặc các file .tex trong v_dir
                tex_sources: List[Path] = []
                inlined_p = elements_dir / f"full_inlined_{v_name}.tex"
                if inlined_p.is_file():
                    tex_sources.append(inlined_p)
                elif v_dir.is_dir():
                    tex_sources.extend(sorted(v_dir.glob("**/*.tex")))

                for tp in tex_sources:
                    try:
                        t_content = tp.read_text(encoding="utf-8", errors="ignore")
                        for block_m in re.finditer(
                            r"\\begin\{(?:thebibliography|references)\}(.*?)\\end\{(?:thebibliography|references)\}",
                            t_content,
                            flags=re.DOTALL | re.IGNORECASE,
                        ):
                            for part in re.split(r"(?=\\bibitem\b)", block_m.group(1)):
                                if part.strip().startswith("\\bibitem"):
                                    bibitems_list.append(part.strip())
                    except Exception:
                        pass

            if bibitems_list:
                v_entries.extend(cls.parse_bibitem_entries(bibitems_list, arxiv_id, v_name))

            collected_entries.extend(v_entries)

        # 4. Nếu vẫn rỗng mà trong <pub_dir>/bibtex/ có file .bib hoặc bibtex.json của Khoa thì đọc
        if not collected_entries and bibtex_dir.is_dir():
            for bf in sorted(bibtex_dir.glob("*.bib")):
                try:
                    bib_txt = bf.read_text(encoding="utf-8", errors="ignore")
                    collected_entries.extend(cls.parse_bib_content(bib_txt, arxiv_id, "v1"))
                except Exception:
                    pass
            bib_json_p = bibtex_dir / "bibtex.json"
            if not collected_entries and bib_json_p.is_file():
                try:
                    data = json.loads(bib_json_p.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                std_item = cls.standardize_entry_dict(item, arxiv_id, "v1")
                                if std_item:
                                    collected_entries.append(std_item)
                except Exception:
                    pass

        return collected_entries

    @classmethod
    def _are_entries_same_reference(
        cls, e1: Dict[str, Any], e2: Dict[str, Any]
    ) -> bool:
        """
        Kiểm tra 2 BibTeX entry có cùng trỏ tới một tài liệu tham khảo hay không
        theo 3 tầng tiêu chí:
        - Tầng 1: Khớp chính xác DOI chuẩn hóa (khi cả 2 đều có DOI).
        - Tầng 2: Khớp chính xác Title chuẩn hóa (độ dài >= 10) + Kiểm tra an toàn
          (không mâu thuẫn DOI, và tương thích về họ tác giả hoặc năm xuất bản).
        - Tầng 3: Cùng citation_key giữa các version và nội dung tương thích
          (không phải trường hợp tái sử dụng key cho một bài báo hoàn toàn khác).
        """
        doi1 = e1.get("_norm_doi", "")
        doi2 = e2.get("_norm_doi", "")

        # Tầng 1: Nếu cả 2 đều có DOI hợp lệ
        if doi1 and doi2:
            return doi1 == doi2

        t1 = e1.get("_norm_title", "")
        t2 = e2.get("_norm_title", "")
        surnames1: Set[str] = e1.get("_surnames", set())
        surnames2: Set[str] = e2.get("_surnames", set())
        y1 = e1.get("year")
        y2 = e2.get("year")

        # Tầng 2: Khớp theo Title đã chuẩn hóa (bỏ qua các tiêu đề mẫu/placeholder)
        if (
            t1
            and t2
            and len(t1) >= 15
            and t1 == t2
            and t1 not in cls.PLACEHOLDER_NORM_TITLES
        ):
            # Nếu cả 2 đều có năm xuất bản hợp lệ nhưng chênh lệch quá 2 năm -> không gộp
            if y1 is not None and y2 is not None and abs(y1 - y2) > 2:
                return False
            # Nếu cả 2 đều có danh sách họ tác giả, kiểm tra xem có ít nhất 1 họ chung
            # hoặc tiêu đề đủ dài (>= 25 ký tự) và năm xuất bản sát nhau
            if surnames1 and surnames2:
                if surnames1.intersection(surnames2):
                    return True
                if y1 is not None and y2 is not None and abs(y1 - y2) <= 1 and len(t1) >= 25:
                    return True
                return False
            return True

        # Tầng 3: Cùng citation_key (trong cùng version hoặc xuyên version)
        if e1["citation_key"] == e2["citation_key"]:
            # Nếu cả 2 đều có title dài nhưng hoàn toàn khác nhau VÀ tác giả cũng khác nhau
            # -> Tác giả đã dùng lại key (ví dụ ref1) cho bài báo khác ở version sau
            if t1 and t2 and t1 != t2:
                if surnames1 and surnames2 and not surnames1.intersection(surnames2):
                    return False
                # Nếu tiêu đề khác nhau hoàn toàn (không phải bản rút gọn của nhau)
                if t1 not in t2 and t2 not in t1 and surnames1 != surnames2:
                    return False
            return True

        return False

    @classmethod
    def unionize_cluster(
        cls, arxiv_id: str, cluster: List[Dict[str, Any]], canonical_key: str
    ) -> Dict[str, Any]:
        """
        Hợp nhất (Union) các trường thông tin từ tất cả các bản ghi trùng lặp trong cụm.
        Đảm bảo giữ lại đầy đủ thông tin nhất (title, authors, year, doi, journal, url, entry_type).
        """
        # 1. Chọn entry_type cụ thể nhất
        best_type = "misc"
        best_type_score = -1
        for e in cluster:
            et = (e.get("entry_type") or "misc").lower()
            score = cls.ENTRY_TYPE_PRIORITY.get(et, 1)
            if score > best_type_score:
                best_type_score = score
                best_type = et

        # 2. Chọn title đầy đủ nhất (ưu tiên bản ở version mới hơn nếu độ dài tương đương)
        best_title = ""
        for e in cluster:
            t = (e.get("title") or "").strip()
            if len(t) > len(best_title):
                best_title = t

        # 3. Hợp nhất danh sách tác giả (ưu tiên danh sách có nhiều tác giả và tên đầy đủ nhất)
        best_authors: List[str] = []
        best_authors_char_len = -1
        for e in cluster:
            auths = e.get("authors") or []
            char_len = sum(len(a) for a in auths)
            if len(auths) > len(best_authors) or (
                len(auths) == len(best_authors) and char_len > best_authors_char_len
            ):
                best_authors = list(auths)
                best_authors_char_len = char_len

        # 4. Hợp nhất year (ưu tiên giá trị khác None; nếu version sau cập nhật năm xuất bản thì lấy)
        best_year: Optional[int] = None
        for e in cluster:
            y = e.get("year")
            if y is not None:
                best_year = y

        # 5. Hợp nhất DOI (lấy DOI hợp lệ đầu tiên/mới nhất không rỗng)
        best_doi = ""
        for e in cluster:
            d = (e.get("doi") or "").strip()
            if d:
                best_doi = d

        # 6. Hợp nhất Journal / Venue (ưu tiên tên tạp chí/hội nghị chính thức hơn "arXiv preprint")
        best_journal = ""
        for e in cluster:
            j = (e.get("journal") or "").strip()
            if not j:
                continue
            if not best_journal:
                best_journal = j
            else:
                j_is_preprint = "arxiv" in j.lower() or "preprint" in j.lower()
                best_is_preprint = "arxiv" in best_journal.lower() or "preprint" in best_journal.lower()
                if best_is_preprint and not j_is_preprint:
                    best_journal = j
                elif not j_is_preprint and len(j) > len(best_journal):
                    best_journal = j

        # 7. Hợp nhất URL (lấy URL không rỗng; nếu không có URL nhưng có DOI thì tạo URL chuẩn từ DOI)
        best_url = ""
        for e in cluster:
            u = (e.get("url") or "").strip()
            if u:
                best_url = u
        if not best_url and best_doi:
            best_url = f"https://doi.org/{best_doi}"

        return {
            "arxiv_id": arxiv_id,
            "citation_key": canonical_key,
            "entry_type": best_type,
            "title": best_title,
            "authors": best_authors,
            "year": best_year,
            "doi": best_doi,
            "journal": best_journal,
            "url": best_url,
        }

    @classmethod
    def deduplicate_bibtex(
        cls, arxiv_id: str, raw_entries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Thực hiện khử trùng lặp danh sách BibTeX entries xuyên version cho 1 bài báo.
        Trả về:
            - bibtex: Danh sách các dòng chuẩn của bảng `bibtex` sau khi gộp và union trường.
            - key_mapping: Từ điển ánh xạ `old_citation_key -> canonical_citation_key`.
            - merged_details: Chi tiết các cụm có từ 2 entries trở lên được gộp (phục vụ báo cáo).
            - stats: Thống kê số lượng entries trước/sau dedup, số key được gộp, số trường được bổ sung.
        """
        if not raw_entries:
            return {
                "arxiv_id": arxiv_id,
                "bibtex": [],
                "key_mapping": {},
                "merged_details": [],
                "stats": {
                    "total_raw_entries": 0,
                    "unique_raw_keys": 0,
                    "dedup_entries_count": 0,
                    "removed_duplicates_count": 0,
                    "reduction_percentage": 0.0,
                    "renamed_keys_count": 0,
                    "fields_enriched_count": 0,
                },
            }

        # Chuẩn bị các thuộc tính chuẩn hóa tạm thời để so khớp nhanh
        prepared: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_entries):
            e = dict(item)
            e["_idx"] = idx
            e["_norm_doi"] = cls.normalize_doi(e.get("doi"))
            e["_norm_title"] = cls.normalize_title_key(e.get("title"))
            e["_surnames"] = cls.extract_author_surnames(e.get("authors") or [])
            prepared.append(e)

        n = len(prepared)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

        # Dùng chỉ mục băm (hash index) cho DOI, Title, và Citation Key để tăng tốc O(N)
        doi_buckets: Dict[str, List[int]] = {}
        title_buckets: Dict[str, List[int]] = {}
        key_buckets: Dict[str, List[int]] = {}

        for i, e in enumerate(prepared):
            if e["_norm_doi"]:
                doi_buckets.setdefault(e["_norm_doi"], []).append(i)
            if e["_norm_title"] and len(e["_norm_title"]) >= 10:
                title_buckets.setdefault(e["_norm_title"], []).append(i)
            key_buckets.setdefault(e["citation_key"], []).append(i)

        # 1. Gom theo DOI trùng khớp
        for indices in doi_buckets.values():
            first = indices[0]
            for other in indices[1:]:
                union(first, other)

        # 2. Gom theo Normalized Title + kiểm tra an toàn
        for indices in title_buckets.values():
            for i_idx in range(len(indices)):
                for j_idx in range(i_idx + 1, len(indices)):
                    u, v = indices[i_idx], indices[j_idx]
                    if cls._are_entries_same_reference(prepared[u], prepared[v]):
                        union(u, v)

        # 3. Gom theo Citation Key giữa các version (nếu nội dung tương thích)
        for indices in key_buckets.values():
            for i_idx in range(len(indices)):
                for j_idx in range(i_idx + 1, len(indices)):
                    u, v = indices[i_idx], indices[j_idx]
                    if cls._are_entries_same_reference(prepared[u], prepared[v]):
                        union(u, v)

        # Gom các phần tử theo gốc Union-Find
        clusters_map: Dict[int, List[Dict[str, Any]]] = {}
        for i, e in enumerate(prepared):
            root = find(i)
            clusters_map.setdefault(root, []).append(e)

        unified_bibtex: List[Dict[str, Any]] = []
        key_mapping: Dict[str, str] = {}
        used_canonical_keys: Set[str] = set()
        merged_details: List[Dict[str, Any]] = []
        fields_enriched_count = 0

        for root_idx in sorted(clusters_map.keys()):
            cluster = clusters_map[root_idx]
            # Chọn canonical_key là key của phần tử xuất hiện đầu tiên trong cụm
            base_key = cluster[0]["citation_key"]
            canonical_key = base_key
            if canonical_key in used_canonical_keys:
                # Phòng trường hợp 2 bài hoàn toàn khác nhau nhưng trùng key ở 2 version khác nhau
                v_tag = cluster[0].get("version", "v2")
                canonical_key = f"{base_key}_{v_tag}"
            used_canonical_keys.add(canonical_key)

            distinct_keys_in_cluster: List[str] = []
            for e in cluster:
                k = e["citation_key"]
                if k not in distinct_keys_in_cluster:
                    distinct_keys_in_cluster.append(k)
                # Ánh xạ old_key -> canonical_key
                if k not in key_mapping:
                    key_mapping[k] = canonical_key

            unified_entry = cls.unionize_cluster(arxiv_id, cluster, canonical_key)
            unified_bibtex.append(unified_entry)

            # Kiểm tra xem quá trình unionize có bổ sung thêm trường bị khuyết so với bản đầu tiên không
            first_e = cluster[0]
            enriched_fields: List[str] = []
            for f_check in ("title", "year", "doi", "journal", "url"):
                if not first_e.get(f_check) and unified_entry.get(f_check):
                    enriched_fields.append(f_check)
            if len(unified_entry.get("authors") or []) > len(first_e.get("authors") or []):
                enriched_fields.append("authors")
            if enriched_fields:
                fields_enriched_count += 1

            if len(distinct_keys_in_cluster) > 1 or enriched_fields:
                merged_details.append({
                    "canonical_key": canonical_key,
                    "all_keys": distinct_keys_in_cluster,
                    "cluster_size": len(cluster),
                    "enriched_fields": enriched_fields,
                    "title": unified_entry["title"],
                    "doi": unified_entry["doi"],
                    "year": unified_entry["year"],
                })

        total_raw = len(raw_entries)
        unique_raw_keys = len({e["citation_key"] for e in raw_entries})
        dedup_count = len(unified_bibtex)
        removed_count = total_raw - dedup_count
        reduction_pct = round((removed_count / total_raw * 100.0), 2) if total_raw > 0 else 0.0
        renamed_keys_count = sum(1 for k, v in key_mapping.items() if k != v)

        return {
            "arxiv_id": arxiv_id,
            "bibtex": unified_bibtex,
            "key_mapping": key_mapping,
            "merged_details": merged_details,
            "stats": {
                "total_raw_entries": total_raw,
                "unique_raw_keys": unique_raw_keys,
                "dedup_entries_count": dedup_count,
                "removed_duplicates_count": removed_count,
                "reduction_percentage": reduction_pct,
                "renamed_keys_count": renamed_keys_count,
                "fields_enriched_count": fields_enriched_count,
            },
        }

    @classmethod
    def rewrite_citations_in_text(
        cls, text: str, key_mapping: Dict[str, str]
    ) -> Tuple[str, int, int]:
        """
        Đổi tên các citation_key bên trong mọi lệnh \\cite{...} của một chuỗi văn bản
        sang canonical_key và khử trùng lặp danh sách key nằm chung trong cùng 1 lệnh \\cite.
        Trả về: (chuỗi_đã_cập_nhật, số_lệnh_cite_thay_đổi, số_key_được_đổi_tên).
        """
        if not text or not key_mapping or "\\" not in text:
            return text, 0, 0

        commands_changed = 0
        keys_renamed = 0

        def _replace_cite(match: re.Match) -> str:
            nonlocal commands_changed, keys_renamed
            prefix = match.group(1)
            keys_body = match.group(2)
            suffix = match.group(3)

            raw_keys = [k.strip() for k in keys_body.split(",") if k.strip()]
            if not raw_keys:
                return match.group(0)

            new_keys: List[str] = []
            changed_this_cmd = False

            for rk in raw_keys:
                mapped = key_mapping.get(rk, rk)
                if mapped != rk:
                    keys_renamed += 1
                    changed_this_cmd = True
                if mapped not in new_keys:
                    new_keys.append(mapped)
                else:
                    # Hai key khác nhau trong cùng 1 lệnh \cite{k1, k2} gộp về cùng 1 canonical_key
                    changed_this_cmd = True

            new_body = ", ".join(new_keys)
            if changed_this_cmd:
                commands_changed += 1
                return f"{prefix}{new_body}{suffix}"
            return match.group(0)

        updated_text = cls.CITE_COMMAND_PATTERN.sub(_replace_cite, text)
        return updated_text, commands_changed, keys_renamed

    @classmethod
    def rewrite_citations_in_elements(
        cls,
        version_parsed_data: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
        key_mapping: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Quét và cập nhật toàn bộ lệnh \\cite{...} trong danh sách elements của mọi version
        TRƯỚC khi thực hiện khử trùng lặp Full-text content.
        """
        total_cmds_changed = 0
        total_keys_renamed = 0
        sample_rewrites: List[Dict[str, str]] = []

        # Kiểm tra xem có key nào thực sự thay đổi không
        has_active_mapping = any(k != v for k, v in key_mapping.items())
        if not has_active_mapping and not key_mapping:
            return {
                "cite_commands_updated": 0,
                "cite_keys_renamed": 0,
                "samples": [],
            }

        for v_name, (elems, _) in version_parsed_data.items():
            for elem in elems:
                old_content = elem.get("content", "")
                new_content, cmds_ch, keys_ren = cls.rewrite_citations_in_text(
                    old_content, key_mapping
                )
                if cmds_ch > 0:
                    elem["content"] = new_content
                    total_cmds_changed += cmds_ch
                    total_keys_renamed += keys_ren
                    if len(sample_rewrites) < 5:
                        sample_rewrites.append({
                            "version": v_name,
                            "before": old_content,
                            "after": new_content,
                        })

        return {
            "cite_commands_updated": total_cmds_changed,
            "cite_keys_renamed": total_keys_renamed,
            "samples": sample_rewrites,
        }


class ElementDeduplicator:
    """
    Khử trùng lặp phần tử (Full-text Content) giữa các version của bài báo
    và ánh xạ lại bảng cạnh hierarchy cũng như cây phân cấp của từng version.
    """

    HEADING_KINDS = {
        "part",
        "chapter",
        "section",
        "subsection",
        "subsubsection",
        "paragraph",
        "subparagraph",
    }

    @classmethod
    def normalize_content_key(cls, content: str) -> str:
        """Chuẩn hóa khoảng trắng của chuỗi nội dung để so khớp trùng lặp chính xác."""
        return re.sub(r"\s+", " ", content).strip()

    @classmethod
    def ensure_cleaned_content(cls, kind: str, content: str) -> str:
        """
        Đảm bảo nội dung phần tử đã được chuẩn hóa toán và làm sạch lệnh LaTeX trình bày
        trước khi so khớp full-text (tuân thủ quy định của Milestone 2:
        'full-text matching should be performed after cleanup').
        """
        if not content or kind in ("document", "itemize"):
            return content
        text = HierarchyExtractor.standardize_math(content)
        text = HierarchyExtractor.clean_latex(text)
        return cls.normalize_content_key(text)

    @classmethod
    def deduplicate_publication(
        cls,
        arxiv_id: str,
        version_parsed_data: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
        key_mapping: Optional[Dict[str, str]] = None,
        clean_before_match: bool = False,
    ) -> Dict[str, Any]:
        """
        Khử trùng lặp phần tử trên tất cả các version của 1 bài báo.
        - Nếu có `key_mapping`, tự động đồng bộ các lệnh \\cite{...} trước khi so khớp.
        - Giữ riêng nút gốc Document (<arxiv_id>#root1, #root2...) cho từng version.
        - Bảo vệ cấu trúc cây: các khối `itemize` được phân biệt bằng chữ ký nội dung các `item`
          con trực thuộc để không gộp nhầm các danh sách khác nhau trong cùng 1 version, nhưng
          vẫn gộp chính xác nếu danh sách giữ nguyên qua các version.
        - Các tiêu đề mục (heading) trùng tên trong cùng 1 version được phân biệt theo thứ tự
          xuất hiện trong version đó để không làm chập nhánh cây phân cấp.
        """
        cite_rewrite_stats = {
            "cite_commands_updated": 0,
            "cite_keys_renamed": 0,
            "samples": [],
        }
        if key_mapping:
            cite_rewrite_stats = ReferenceDeduplicator.rewrite_citations_in_elements(
                version_parsed_data, key_mapping
            )

        unique_elements: List[Dict[str, Any]] = []
        # Ánh xạ khóa chữ ký phần tử -> canonical element_id
        signature_to_id: Dict[Tuple[str, ...], str] = {}

        # Bộ đếm theo từng loại phần tử để đánh mã ID chuẩn (ví dụ: #s1, #sec1, #fig1)
        kind_counters: Dict[str, int] = {}

        def get_kind_prefix(k: str) -> str:
            if k == "sentence":
                return "s"
            if k in ("section", "chapter", "part"):
                return "sec"
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

        total_before = 0
        by_kind_before: Dict[str, int] = {}
        unified_hierarchy: List[Dict[str, Any]] = []
        per_version_dedup_elements: Dict[str, List[Dict[str, Any]]] = {}
        per_version_dedup_edges: Dict[str, List[Dict[str, Any]]] = {}

        # Duyệt các version theo thứ tự tăng dần (v1, v2, ...)
        sorted_versions = sorted(
            version_parsed_data.keys(),
            key=lambda x: int(x[1:]) if x.startswith("v") and x[1:].isdigit() else x,
        )

        for v_name in sorted_versions:
            raw_elements, raw_hierarchy = version_parsed_data[v_name]
            raw_id_to_final_id: Dict[str, str] = {}

            # Đánh chỉ mục nội dung của từng raw_id trong version hiện tại
            # để tính chữ ký con cho các khối container `itemize`
            elem_by_raw_id: Dict[str, Dict[str, Any]] = {}
            for elem in raw_elements:
                r_id = elem.get("raw_id") or elem.get("element_id")
                if clean_before_match:
                    elem["content"] = cls.ensure_cleaned_content(elem["kind"], elem["content"])
                elem_by_raw_id[r_id] = elem

            children_by_raw_parent: Dict[str, List[str]] = {}
            for edge in raw_hierarchy:
                p_raw = edge.get("parent_raw_id") or edge.get("parent_id")
                c_raw = edge.get("element_raw_id") or edge.get("element_id")
                if p_raw and c_raw:
                    children_by_raw_parent.setdefault(p_raw, []).append(c_raw)

            # Đếm số lần xuất hiện của cùng (kind, norm_content) đối với các node trung gian
            # ngay trong cùng 1 version để chống chập 2 nhánh khác nhau trong cùng 1 version
            non_leaf_seen_in_version: Dict[Tuple[str, str], int] = {}

            v_num = (
                int(v_name[1:])
                if v_name.startswith("v") and v_name[1:].isdigit()
                else int(v_name)
            )

            for elem in raw_elements:
                total_before += 1
                kind = elem["kind"]
                content = elem["content"]
                raw_id = elem.get("raw_id") or elem.get("element_id")

                by_kind_before[kind] = by_kind_before.get(kind, 0) + 1

                # Quy tắc Milestone 2: Mỗi version giữ 1 nút gốc Document riêng (<arxiv_id>#root1, #root2)
                if kind == "document":
                    final_id = f"{arxiv_id}#root{v_num}"
                    raw_id_to_final_id[raw_id] = final_id
                    unique_elements.append({
                        "element_id": final_id,
                        "arxiv_id": arxiv_id,
                        "kind": kind,
                        "content": f"Version {v_num} root" if not content else content,
                    })
                    continue

                norm_text = cls.normalize_content_key(content)

                # Xây dựng khóa chữ ký (signature) để khử trùng lặp chính xác
                if kind == "itemize":
                    # Chữ ký của khối danh sách dựa trên nội dung các mục \item con trực tiếp của nó
                    child_ids = children_by_raw_parent.get(raw_id, [])
                    child_texts = [
                        cls.normalize_content_key(elem_by_raw_id[cid]["content"])
                        for cid in child_ids
                        if cid in elem_by_raw_id
                    ]
                    list_body_sig = " || ".join(child_texts) if child_texts else norm_text
                    occ = non_leaf_seen_in_version.get((kind, list_body_sig), 0) + 1
                    non_leaf_seen_in_version[(kind, list_body_sig)] = occ
                    sig_key: Tuple[str, ...] = (kind, list_body_sig, str(occ))
                elif kind in cls.HEADING_KINDS:
                    # Nếu trong cùng 1 version có 2 mục con trùng tên (ví dụ 2 subsection "Experiments"),
                    # phân biệt bằng thứ tự xuất hiện `occ` trong version đó để không làm hỏng cây
                    occ = non_leaf_seen_in_version.get((kind, norm_text), 0) + 1
                    non_leaf_seen_in_version[(kind, norm_text)] = occ
                    sig_key = (kind, norm_text, str(occ))
                else:
                    # Các nút lá (sentence, block_formula, figure, item): khớp exact match theo (kind, norm_text)
                    sig_key = (kind, norm_text)

                if sig_key in signature_to_id:
                    final_id = signature_to_id[sig_key]
                else:
                    prefix = get_kind_prefix(kind)
                    count = kind_counters.get(prefix, 0) + 1
                    kind_counters[prefix] = count
                    final_id = f"{arxiv_id}#{prefix}{count}"

                    signature_to_id[sig_key] = final_id
                    unique_elements.append({
                        "element_id": final_id,
                        "arxiv_id": arxiv_id,
                        "kind": kind,
                        "content": content,
                    })

                raw_id_to_final_id[raw_id] = final_id

            # Ánh xạ lại các cạnh cha-con cho version hiện tại
            seen_edges_in_version: Set[Tuple[int, str, str]] = set()
            v_edges: List[Dict[str, Any]] = []
            for edge in raw_hierarchy:
                raw_child = edge.get("element_raw_id") or edge.get("element_id")
                raw_parent = edge.get("parent_raw_id") or edge.get("parent_id")

                if raw_parent is None:
                    continue

                final_child = raw_id_to_final_id.get(raw_child)
                final_parent = raw_id_to_final_id.get(raw_parent)

                if final_child and final_parent is not None and final_child != final_parent:
                    edge_key = (v_num, final_child, final_parent)
                    if edge_key not in seen_edges_in_version:
                        seen_edges_in_version.add(edge_key)
                        edge_obj = {
                            "arxiv_id": arxiv_id,
                            "version": v_num,
                            "element_id": final_child,
                            "parent_id": final_parent,
                        }
                        unified_hierarchy.append(edge_obj)
                        v_edges.append(edge_obj)

            # Tập các phần tử xuất hiện trong version hiện tại (với ID đã deduplicate)
            elem_map = {e["element_id"]: e for e in unique_elements}
            v_elem_ids: List[str] = []
            seen_v_eids: Set[str] = set()
            for elem in raw_elements:
                r_id = elem.get("raw_id") or elem.get("element_id")
                f_id = raw_id_to_final_id.get(r_id)
                if f_id and f_id not in seen_v_eids and f_id in elem_map:
                    seen_v_eids.add(f_id)
                    v_elem_ids.append(f_id)

            per_version_dedup_elements[v_name] = [elem_map[eid] for eid in v_elem_ids]
            per_version_dedup_edges[v_name] = v_edges

        total_after = len(unique_elements)
        reduction_count = total_before - total_after
        reduction_pct = (
            (reduction_count / total_before * 100.0) if total_before > 0 else 0.0
        )

        by_kind_after: Dict[str, int] = {}
        for elem in unique_elements:
            k = elem["kind"]
            by_kind_after[k] = by_kind_after.get(k, 0) + 1

        return {
            "arxiv_id": arxiv_id,
            "elements": unique_elements,
            "hierarchy": unified_hierarchy,
            "per_version_elements": per_version_dedup_elements,
            "per_version_hierarchy": per_version_dedup_edges,
            "cite_rewrite_stats": cite_rewrite_stats,
            "stats": {
                "total_before": total_before,
                "total_after": total_after,
                "reduction_count": reduction_count,
                "reduction_percentage": round(reduction_pct, 2),
                "by_kind_before": by_kind_before,
                "by_kind_after": by_kind_after,
                "cite_commands_updated": cite_rewrite_stats["cite_commands_updated"],
                "cite_keys_renamed": cite_rewrite_stats["cite_keys_renamed"],
            },
        }


class PublicationDeduplicator:
    """
    Điều phối trọn vẹn nhiệm vụ Deduplication (của Kiệt) cho từng bài báo:
    1. Tải dữ liệu elements/hierarchy của từng version (từ phần của Tân & Khoa) và dữ liệu references.
    2. Thực hiện Reference Deduplication & Field Unionization -> xuất bảng `bibtex` vào `<pub_dir>/bibtex/`
       và `<pub_dir>/elements/`.
    3. Đổi tên lệnh `\\cite{...}` trong toàn bộ `elements` theo `key_mapping`.
    4. Thực hiện Full-text Element Deduplication & cập nhật lại bảng `elements`, `hierarchy`
       cùng các file cây phân cấp `hierarchy_tree.json` của từng version.
    """

    @classmethod
    def load_or_build_version_data(
        cls, pub_dir: Path
    ) -> Tuple[List[str], Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]], Set[str]]:
        """
        Đọc dữ liệu `elements` và `hierarchy` theo từng version từ `<pub_dir>/elements/`.
        Nếu bài báo chưa được chạy bước Hierarchy Construction của Tân, tự động gọi
        `HierarchyBuilder.parse_version` để đảm bảo pipeline hoạt động trơn tru.
        """
        pub_dir = Path(pub_dir).resolve()
        arxiv_id = pub_dir.name
        elements_dir = pub_dir / "elements"
        elements_dir.mkdir(parents=True, exist_ok=True)

        version_dirs = sorted(
            [
                d
                for d in pub_dir.iterdir()
                if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
            ],
            key=lambda x: int(x.name[1:]),
        )
        if version_dirs:
            version_names = [vd.name for vd in version_dirs]
        else:
            inlined_files = sorted(elements_dir.glob("full_inlined_v*.tex"))
            version_names = [f.stem.replace("full_inlined_", "") for f in inlined_files]

        version_parsed_data: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
        empty_versions: Set[str] = set()
        preserved_references_by_version: Dict[str, Any] = {}

        raw_refs_path = elements_dir / "raw_references.json"
        has_raw_refs = raw_refs_path.is_file()

        for v_name in version_names:
            pre_elems_path = elements_dir / f"pre_dedup_elements_{v_name}.json"
            pre_hier_path = elements_dir / f"pre_dedup_hierarchy_{v_name}.json"
            elems_v_path = elements_dir / f"elements_{v_name}.json"
            hier_v_path = elements_dir / f"hierarchy_{v_name}.json"
            tree_v_path = elements_dir / f"hierarchy_tree_{v_name}.json"

            # 1. Nếu có bản lưu trước dedup (đảm bảo tính lũy đẳng idempotent khi chạy lại)
            if pre_elems_path.is_file() and pre_hier_path.is_file():
                try:
                    # Kiểm tra nếu Tân/Khoa vừa chạy lại và tạo elements_{v_name}.json mới hơn có mã #vX_
                    use_fresh_v = False
                    if elems_v_path.is_file() and hier_v_path.is_file():
                        v_elems_check = json.loads(elems_v_path.read_text(encoding="utf-8"))
                        if any(f"#{v_name}_" in str(e.get("element_id", "")) for e in v_elems_check[:5]):
                            use_fresh_v = True
                    if not use_fresh_v:
                        raw_elems = json.loads(pre_elems_path.read_text(encoding="utf-8"))
                        raw_hier = json.loads(pre_hier_path.read_text(encoding="utf-8"))
                        if raw_elems:
                            version_parsed_data[v_name] = (raw_elems, raw_hier)
                        else:
                            empty_versions.add(v_name)
                        continue
                except Exception:
                    pass

            # 2. Nếu có sẵn file elements_vX.json chưa dedup của Tân/Khoa (chứa mã #vX_ hoặc raw_id)
            if elems_v_path.is_file() and hier_v_path.is_file():
                try:
                    raw_elems = json.loads(elems_v_path.read_text(encoding="utf-8"))
                    raw_hier = json.loads(hier_v_path.read_text(encoding="utf-8"))
                    if raw_elems:
                        is_pre_dedup = any(
                            "raw_id" in e or f"#{v_name}_" in str(e.get("element_id", ""))
                            for e in raw_elems[:5]
                        )
                        if is_pre_dedup:
                            pre_elems_path.write_text(
                                json.dumps(raw_elems, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                            pre_hier_path.write_text(
                                json.dumps(raw_hier, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                            version_parsed_data[v_name] = (raw_elems, raw_hier)
                            continue
                    else:
                        empty_versions.add(v_name)
                        continue
                except Exception:
                    pass
            elif tree_v_path.is_file():
                try:
                    t_obj = json.loads(tree_v_path.read_text(encoding="utf-8"))
                    if t_obj.get("is_empty"):
                        empty_versions.add(v_name)
                        continue
                except Exception:
                    pass

            # 3. Nếu chưa có bản pre_dedup, gọi HierarchyBuilder của Tân để tạo từ full_inlined_vX.tex hoặc vX/
            v_dir = pub_dir / v_name
            meta = None
            meta_p = v_dir / "metadata.json"
            if meta_p.is_file():
                try:
                    meta = json.loads(meta_p.read_text(encoding="utf-8"))
                except Exception:
                    pass

            inlined_tex, gather_res = HierarchyBuilder.load_version_tex_from_elements_or_source(
                pub_dir=pub_dir,
                version_name=v_name,
                metadata=meta,
            )
            if inlined_tex and gather_res:
                inlined_p = elements_dir / f"full_inlined_{v_name}.tex"
                if not inlined_p.is_file():
                    inlined_p.write_text(inlined_tex, encoding="utf-8")

            if not inlined_tex:
                empty_versions.add(v_name)
                err_msg = (
                    gather_res.get("error")
                    if gather_res
                    else "Version không có mã nguồn TeX (empty record)"
                )
                v_out = HierarchyBuilder.build_version_hierarchy_outputs(
                    arxiv_id=arxiv_id,
                    version_name=v_name,
                    raw_elements=[],
                    raw_hierarchy=[],
                    is_empty=True,
                    error_msg=err_msg,
                )
                if v_dir.is_dir():
                    (v_dir / "hierarchy_tree.json").write_text(
                        json.dumps(v_out["tree"], ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                (elements_dir / f"hierarchy_tree_{v_name}.json").write_text(
                    json.dumps(v_out["tree"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                preserved_references_by_version[v_name] = {
                    "status": "empty_version",
                    "thebibliography_blocks": [],
                    "bibitem_entries": [],
                    "bibliography_commands": [],
                    "reference_sections": [],
                }
                continue

            raw_elems, raw_hier, preserved_refs = HierarchyBuilder.parse_version(
                arxiv_id=arxiv_id,
                version=v_name,
                inlined_tex=inlined_tex,
            )
            pre_elems_path.write_text(
                json.dumps(raw_elems, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            pre_hier_path.write_text(
                json.dumps(raw_hier, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            version_parsed_data[v_name] = (raw_elems, raw_hier)
            preserved_references_by_version[v_name] = preserved_refs

        if preserved_references_by_version and not has_raw_refs:
            raw_refs_path.write_text(
                json.dumps(preserved_references_by_version, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return version_names, version_parsed_data, empty_versions

    @classmethod
    def deduplicate_and_save_publication(
        cls, pub_dir: Path, clean_before_match: bool = True
    ) -> Dict[str, Any]:
        """
        Thực thi toàn bộ quy trình Deduplication cho 1 bài báo và lưu kết quả:
        - Lưu bảng `bibtex` đã khử trùng lặp vào `<pub_dir>/bibtex/bibtex.{json,csv}`
          và `<pub_dir>/elements/bibtex.{json,csv}`.
        - Cập nhật bảng `elements` và `hierarchy` đã khử trùng lặp vào `<pub_dir>/elements/`.
        - Cập nhật trực tiếp cây phân cấp `hierarchy_tree.json` của từng version (trong `<pub_dir>/<v_name>/`
          và `<pub_dir>/elements/hierarchy_tree_<v_name>.json`) với các `element_id` dùng chung sau dedup.
        """
        pub_dir = Path(pub_dir).resolve()
        arxiv_id = pub_dir.name
        elements_dir = pub_dir / "elements"
        bibtex_dir = pub_dir / "bibtex"
        elements_dir.mkdir(parents=True, exist_ok=True)
        bibtex_dir.mkdir(parents=True, exist_ok=True)

        # Bước 1: Đọc dữ liệu từng version
        version_names, version_parsed_data, empty_versions = cls.load_or_build_version_data(pub_dir)

        # Bước 2: Đọc và khử trùng lặp Reference Entries (BibTeX)
        raw_bib_entries = ReferenceDeduplicator.load_publication_references(pub_dir, version_names)
        bib_dedup_res = ReferenceDeduplicator.deduplicate_bibtex(arxiv_id, raw_bib_entries)

        # Lưu kết quả BibTeX đã khử trùng lặp vào folder `bibtex/` và `elements/`
        bibtex_json_str = json.dumps(bib_dedup_res["bibtex"], ensure_ascii=False, indent=2)
        (bibtex_dir / "bibtex.json").write_text(bibtex_json_str, encoding="utf-8")
        (elements_dir / "bibtex.json").write_text(bibtex_json_str, encoding="utf-8")
        (bibtex_dir / "citation_key_mapping.json").write_text(
            json.dumps(bib_dedup_res["key_mapping"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Bước 3: Đổi tên lệnh \cite{...} trong elements & Khử trùng lặp Full-text Content
        elem_dedup_res = ElementDeduplicator.deduplicate_publication(
            arxiv_id=arxiv_id,
            version_parsed_data=version_parsed_data,
            key_mapping=bib_dedup_res["key_mapping"],
            clean_before_match=clean_before_match,
        )

        # Bước 4: Cập nhật ngược lại vào cây `hierarchy_tree.json` của từng version (phần của Tân)
        for v_name in version_names:
            if v_name in empty_versions or v_name not in elem_dedup_res["per_version_elements"]:
                continue

            v_elems = elem_dedup_res["per_version_elements"][v_name]
            v_edges = elem_dedup_res["per_version_hierarchy"][v_name]

            updated_tree = HierarchyBuilder.build_nested_tree(
                arxiv_id=arxiv_id,
                version_name=v_name,
                elements=v_elems,
                hierarchy_edges=v_edges,
            )

            v_dir = pub_dir / v_name
            if v_dir.is_dir():
                (v_dir / "hierarchy_tree.json").write_text(
                    json.dumps(updated_tree, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            (elements_dir / f"hierarchy_tree_{v_name}.json").write_text(
                json.dumps(updated_tree, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (elements_dir / f"elements_{v_name}.json").write_text(
                json.dumps(v_elems, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (elements_dir / f"hierarchy_{v_name}.json").write_text(
                json.dumps(v_edges, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # Bước 5: Lưu 3 bảng chuẩn (elements, hierarchy, bibtex) ở định dạng JSON, CSV, Parquet
        (elements_dir / "elements.json").write_text(
            json.dumps(elem_dedup_res["elements"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (elements_dir / "hierarchy.json").write_text(
            json.dumps(elem_dedup_res["hierarchy"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        try:
            import pandas as pd

            df_elements = pd.DataFrame(elem_dedup_res["elements"])
            df_elements.to_csv(elements_dir / "elements.csv", index=False, encoding="utf-8")

            df_hierarchy = pd.DataFrame(elem_dedup_res["hierarchy"])
            df_hierarchy.to_csv(elements_dir / "hierarchy.csv", index=False, encoding="utf-8")

            df_bibtex = pd.DataFrame(
                bib_dedup_res["bibtex"],
                columns=[
                    "arxiv_id",
                    "citation_key",
                    "entry_type",
                    "title",
                    "authors",
                    "year",
                    "doi",
                    "journal",
                    "url",
                ],
            )
            df_bibtex.to_csv(bibtex_dir / "bibtex.csv", index=False, encoding="utf-8")
            df_bibtex.to_csv(elements_dir / "bibtex.csv", index=False, encoding="utf-8")

            try:
                df_elements.to_parquet(elements_dir / "elements.parquet", index=False)
                df_hierarchy.to_parquet(elements_dir / "hierarchy.parquet", index=False)
                df_bibtex.to_parquet(bibtex_dir / "bibtex.parquet", index=False)
                df_bibtex.to_parquet(elements_dir / "bibtex.parquet", index=False)
            except Exception:
                pass
        except Exception:
            pass

        # Lưu báo cáo thống kê deduplication chi tiết của bài báo
        combined_stats = {
            "arxiv_id": arxiv_id,
            "total_versions": len(version_names),
            "tex_versions": len(version_parsed_data),
            "empty_versions": len(empty_versions),
            "elements_dedup": elem_dedup_res["stats"],
            "bibtex_dedup": bib_dedup_res["stats"],
            "merged_bibtex_details": bib_dedup_res["merged_details"],
            "cite_rewrite_samples": elem_dedup_res["cite_rewrite_stats"]["samples"],
        }
        (elements_dir / "dedup_stats.json").write_text(
            json.dumps(combined_stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "arxiv_id": arxiv_id,
            "total_versions": len(version_names),
            "tex_versions": len(version_parsed_data),
            "empty_versions": len(empty_versions),
            "elements_count": len(elem_dedup_res["elements"]),
            "hierarchy_count": len(elem_dedup_res["hierarchy"]),
            "bibtex_count": len(bib_dedup_res["bibtex"]),
            "elements_dedup": elem_dedup_res["stats"],
            "bibtex_dedup": bib_dedup_res["stats"],
            "merged_bibtex_details": bib_dedup_res["merged_details"],
            "cite_rewrite_samples": elem_dedup_res["cite_rewrite_stats"]["samples"],
        }
