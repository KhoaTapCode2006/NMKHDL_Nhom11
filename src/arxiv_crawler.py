import arxiv
import json
import gzip
import io
import tarfile
import time
import re
import os
import requests
from pathlib import Path, PurePosixPath
import urllib.request
from urllib.error import HTTPError
from dotenv import load_dotenv
import http.client

load_dotenv()
API_KEY = os.getenv("API_KEY")
YEAR_MONTH = "2510"  
START_SEQ = 10788
END_SEQ = 10837
INTERVAL = 3.1

all_target_ids = [f"{YEAR_MONTH}.{seq:05d}" for seq in range(START_SEQ, END_SEQ + 1)]

def chunk_list(data, chunk_size=100):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

client = arxiv.Client(
    page_size=100,
    delay_seconds=3.0,
    num_retries=3
)


def save_json_file(data: dict | list, file_path: Path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def build_paper_metadata(paper, versioned_arxiv_id: str) -> dict:
    return {
        "arxiv_id": versioned_arxiv_id,
        "title": paper.title.strip().replace("\n", " "),
        "authors": [author.name for author in paper.authors],
        "version": versioned_arxiv_id.rsplit("v", 1)[-1],
        "submission_date": paper.published.isoformat() if paper.published else None,
        "updated_date": paper.updated.isoformat() if paper.updated else None,
        "journal_ref": paper.journal_ref,
        "primary_category": paper.primary_category,
        "categories": paper.categories
    }

def get_paper_references(arxiv_id: str) -> list[dict]:
    if not API_KEY:
        raise RuntimeError("API_KEY is missing. Set it in the .env file.")

    base_id = re.sub(r"v\d+$", "", arxiv_id)
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{base_id}"
    params = {
        "fields": (
            "references.paperId,references.externalIds,references.title,"
            "references.authors,references.year"
        )
    }
    api_key = API_KEY
    headers = {"x-api-key": api_key}
    
    max_retries = 10
    
    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code == 429:
            wait_time = INTERVAL * (attempt + 2)
            print(f"Lỗi 429 cho {arxiv_id}. Đang chờ {wait_time}s để thử lại ({attempt + 1}/{max_retries})...")
            time.sleep(wait_time)
            continue
            
        response.raise_for_status()
        
        time.sleep(INTERVAL)
        return response.json().get("references") or []
        
    print(f"Skipping {arxiv_id} persistent 429.")
    return None


def get_source(
    arxiv_id: str,
    save_dir: str | Path = "./sourcesArxiv",
    paper=None,
    references: list | None = None,
) -> dict:
    """Download one version and save only its TeX/Bib files and JSON metadata."""
    base_id, separator, version = arxiv_id.rpartition("v")
    if not separator or not version.isdigit():
        base_id, version = arxiv_id, "1"
        versioned_arxiv_id = f"{base_id}v{version}"
    else:
        versioned_arxiv_id = arxiv_id

    if paper is None:
        paper = next(client.results(arxiv.Search(id_list=[base_id])))

    paper_dir = Path(save_dir) / base_id.replace("/", "-")
    version_dir = paper_dir / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = version_dir / "metadata.json"

    # Resume support: skip versions that were fully downloaded earlier
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                saved_meta = json.load(f)
            if saved_meta.get("source_status") in {"downloaded", "no_tex_or_bib"}:
                print(f"  {versioned_arxiv_id}: skipped (da tai truoc do)")
                return {
                    "paper_dir": str(paper_dir),
                    "version_dir": str(version_dir),
                    "source_status": "skipped",
                }
        except (json.JSONDecodeError, OSError):
            pass

    metadata = build_paper_metadata(paper, versioned_arxiv_id)
    metadata["source_status"] = "downloading"
    save_json_file(metadata, metadata_path)
    if references is not None:
        save_json_file(references, version_dir / "references.json")

    request = urllib.request.Request(
        f"https://export.arxiv.org/src/{versioned_arxiv_id}",
        headers={"User-Agent": "NMKHDL-Nhom11 arXiv source crawler"},
    )
    archive_bytes = b""
    max_download_retries = 3
    
    for attempt in range(max_download_retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                archive_bytes = response.read()
            time.sleep(3)
            break  # Nếu tải thành công, thoát khỏi vòng lặp
            
        except http.client.IncompleteRead as e:
            print(f"Đứt kết nối giữa chừng (IncompleteRead) tải {versioned_arxiv_id}. Thử lại ({attempt + 1}/{max_download_retries})...")
            time.sleep(3)
            
        except urllib.error.HTTPError as error:
            if error.code == 404:
                archive_bytes = b""
                time.sleep(3)
                break  # Bỏ qua retry nếu file thực sự không tồn tại
            elif error.code in (429, 500, 502, 503, 504):
                print(f"Lỗi server {error.code} khi tải {versioned_arxiv_id}. Thử lại ({attempt + 1}/{max_download_retries})...")
                time.sleep(5)
            else:
                raise  # Văng lỗi nếu là các mã HTTP khác
                
        except Exception as e:
            print(f"Lỗi mạng ({type(e).__name__}) tải {versioned_arxiv_id}. Thử lại ({attempt + 1}/{max_download_retries})...")
            time.sleep(3)

    tex_files = []
    bib_files = []
    total_uncompressed_bytes = 0
    filtered_uncompressed_bytes = 0
    raw_archive_bytes = len(archive_bytes) if archive_bytes else 0

    if archive_bytes:
        try:
            archive = tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*")
        except tarfile.ReadError:
            archive = None

        if archive is not None:
            with archive:
                for member in archive.getmembers():
                    if member.isfile():
                        total_uncompressed_bytes += member.size
                    member_path = PurePosixPath(member.name)
                    if (
                        not member.isfile()
                        or member_path.is_absolute()
                        or ".." in member_path.parts
                        or member_path.suffix.lower() not in {".tex", ".bib"}
                    ):
                        continue

                    source_file = archive.extractfile(member)
                    if source_file is None:
                        continue

                    content = source_file.read()
                    filtered_uncompressed_bytes += len(content)
                    output_path = version_dir.joinpath(*member_path.parts)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(content)
                    if member_path.suffix.lower() == ".tex":
                        tex_files.append(str(output_path.relative_to(version_dir)))
                    else:
                        bib_files.append(str(output_path.relative_to(version_dir)))
        else:
            try:
                data = gzip.decompress(archive_bytes)
                total_uncompressed_bytes = len(data)
                if b"\documentclass" in data or b"\begin{document}" in data:
                    output_path = version_dir / "main.tex"
                    output_path.write_bytes(data)
                    tex_files.append("main.tex")
                    filtered_uncompressed_bytes = len(data)
            except Exception:
                pass

    metadata["raw_archive_bytes"] = raw_archive_bytes
    metadata["total_uncompressed_bytes"] = total_uncompressed_bytes
    metadata["filtered_uncompressed_bytes"] = filtered_uncompressed_bytes
    metadata["references_status"] = "success" if isinstance(references, list) else "failed"
    metadata["references_count"] = len(references) if isinstance(references, list) else 0
    metadata["source_status"] = "downloaded" if tex_files or bib_files else "no_tex_or_bib"
    save_json_file(metadata, metadata_path)
    return {
        "paper_dir": str(paper_dir),
        "version_dir": str(version_dir),
        "tex_files": tex_files,
        "bib_files": bib_files,
        "raw_archive_bytes": raw_archive_bytes,
        "total_uncompressed_bytes": total_uncompressed_bytes,
        "filtered_uncompressed_bytes": filtered_uncompressed_bytes,
        "source_status": metadata["source_status"],
    }


if __name__ == "__main__":
    for batch in chunk_list(all_target_ids, chunk_size=50):
        search = arxiv.Search(id_list=batch)

        for paper in client.results(search):
            latest = paper.entry_id.rsplit("/", 1)[-1]
            base_id, _, latest_version = latest.rpartition("v")
            print(f"[{base_id}] (Latest: v{latest_version})")

            paper_dir = Path("./sourcesArxiv") / base_id.replace("/", "-")
            all_versions_exist = True
            for v_num in range(1, int(latest_version) + 1):
                v_meta = paper_dir / f"v{v_num}" / "metadata.json"
                if not v_meta.exists():
                    all_versions_exist = False
                    break
                try:
                    with open(v_meta, "r", encoding="utf-8") as f_m:
                        if json.load(f_m).get("source_status") not in {"downloaded", "no_tex_or_bib"}:
                            all_versions_exist = False
                            break
                except Exception:
                    all_versions_exist = False
                    break

            if all_versions_exist:
                print(f"  Bo qua toan bo {base_id} (da tai du {latest_version} phien ban)")
                continue

            ref_file = paper_dir / "v1" / "references.json"
            references = None
            if ref_file.exists():
                try:
                    with open(ref_file, "r", encoding="utf-8") as f_r:
                        references = json.load(f_r)
                except Exception:
                    references = None

            if references is None:
                references = get_paper_references(base_id)

            for version_number in range(1, int(latest_version) + 1):
                get_source(
                    f"{base_id}v{version_number}",
                    paper=paper,
                    references=references,
                )




