import io
import json
import tarfile
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src import arxiv_crawler


def make_tar(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in entries:
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class ArxivCrawlerTests(unittest.TestCase):
    def setUp(self):
        arxiv_crawler._reference_next_request_at = 0.0
        self.paper = SimpleNamespace(
            title="A paper\n title ",
            authors=[SimpleNamespace(name="Author One"), SimpleNamespace(name="Author Two")],
            published=datetime(2025, 1, 2),
            updated=datetime(2025, 2, 3),
            journal_ref="Journal",
            primary_category="cs.IR",
            categories=["cs.IR", "cs.AI"],
        )

    def test_chunk_list_keeps_order_and_short_final_chunk(self):
        self.assertEqual(
            list(arxiv_crawler.chunk_list(list(range(5)), 2)),
            [[0, 1], [2, 3], [4]],
        )

    def test_build_paper_metadata_normalizes_title_and_records_version(self):
        result = arxiv_crawler.build_paper_metadata(self.paper, "2501.12345v3")

        self.assertEqual(result["arxiv_id"], "2501.12345v3")
        self.assertEqual(result["title"], "A paper title")
        self.assertEqual(result["version"], "3")
        self.assertEqual(result["authors"], ["Author One", "Author Two"])
        self.assertEqual(result["primary_category"], "cs.IR")

    @patch.object(arxiv_crawler.requests, "get")
    def test_get_paper_references_returns_api_references(self, get):
        get.return_value.status_code = 200
        get.return_value.json.return_value = {"references": [{"title": "Reference"}]}

        result = arxiv_crawler.get_paper_references("2501.12345v2")

        self.assertEqual(result, [{"title": "Reference"}])
        self.assertIn("2501.12345", get.call_args.args[0])

    @patch.object(arxiv_crawler.requests, "get")
    def test_get_paper_references_returns_failure_signal_after_repeated_rate_limits(
        self, get
    ):
        response = Mock(status_code=429, headers={"Retry-After": ""})
        get.return_value = response
        clock = [100.0]
        sleep_durations = []

        def fake_sleep(duration):
            sleep_durations.append(duration)
            clock[0] += duration

        with patch.object(arxiv_crawler, "API_KEY", "test-key"), patch(
            "builtins.print"
        ), patch.object(
            arxiv_crawler.time, "monotonic", side_effect=lambda: clock[0]
        ), patch.object(arxiv_crawler.time, "sleep", side_effect=fake_sleep):
            result = arxiv_crawler.get_paper_references("2501.12345")

        self.assertIsNone(result)
        self.assertEqual(get.call_count, arxiv_crawler.REFERENCE_MAX_RETRIES)
        self.assertGreaterEqual(sum(sleep_durations), 500.0)

    @patch.object(arxiv_crawler.requests, "get")
    def test_reference_requests_are_spaced_using_monotonic_deadlines(self, get):
        clock = [100.0]
        request_times = []
        sleep_durations = []

        def make_response(*_args, **_kwargs):
            request_times.append(clock[0])
            response = Mock(status_code=200)
            response.json.return_value = {"references": []}
            return response

        def wake_early_once(duration):
            sleep_durations.append(duration)
            clock[0] += duration - 0.1 if len(sleep_durations) == 1 else duration

        get.side_effect = make_response
        with patch.object(arxiv_crawler, "API_KEY", "test-key"), patch.object(
            arxiv_crawler.time, "monotonic", side_effect=lambda: clock[0]
        ), patch.object(arxiv_crawler.time, "sleep", side_effect=wake_early_once):
            arxiv_crawler.get_paper_references("2501.12345")
            arxiv_crawler.get_paper_references("2501.12346")

        self.assertGreaterEqual(
            request_times[1] - request_times[0],
            arxiv_crawler.REFERENCE_REQUEST_INTERVAL,
        )
        self.assertGreaterEqual(sum(sleep_durations), arxiv_crawler.REFERENCE_REQUEST_INTERVAL)
        self.assertGreater(len(sleep_durations), 1)

    @patch.object(arxiv_crawler.requests, "get")
    def test_retry_after_is_honored_in_addition_to_rate_limit(self, get):
        clock = [100.0]
        request_times = []

        def response_for_attempt(*_args, **_kwargs):
            request_times.append(clock[0])
            response = Mock()
            if len(request_times) == 1:
                response.status_code = 429
                response.headers = {"Retry-After": "15"}
            else:
                response.status_code = 200
                response.json.return_value = {"references": [{"title": "Reference"}]}
            return response

        def fake_sleep(duration):
            clock[0] += duration

        get.side_effect = response_for_attempt
        with patch.object(arxiv_crawler, "API_KEY", "test-key"), patch(
            "builtins.print"
        ), patch.object(
            arxiv_crawler.time, "monotonic", side_effect=lambda: clock[0]
        ), patch.object(arxiv_crawler.time, "sleep", side_effect=fake_sleep):
            result = arxiv_crawler.get_paper_references("2501.12345")

        self.assertEqual(result, [{"title": "Reference"}])
        self.assertGreaterEqual(request_times[1] - request_times[0], 15.0)

    def test_retry_after_http_date_is_parsed(self):
        response = Mock(
            headers={
                "Retry-After": format_datetime(
                    datetime.now(timezone.utc) + timedelta(seconds=30),
                    usegmt=True,
                )
            }
        )

        self.assertGreaterEqual(arxiv_crawler._retry_after_seconds(response), 29.0)

    @patch.object(arxiv_crawler.time, "sleep")
    @patch.object(arxiv_crawler.urllib.request, "urlopen")
    def test_get_source_extracts_tex_and_bib_from_archive(self, urlopen, _sleep):
        archive_bytes = make_tar(
            [
                ("main.tex", "\\documentclass{article}\\begin{document}Text.\\end{document}"),
                ("refs/paper.bib", "@article{key}"),
                ("notes.txt", "ignored"),
                ("../outside.tex", "must not escape the version directory"),
            ]
        )
        urlopen.return_value.__enter__.return_value.read.return_value = archive_bytes

        with tempfile.TemporaryDirectory() as temp_dir:
            result = arxiv_crawler.get_source(
                "2501.12345v2",
                save_dir=temp_dir,
                paper=self.paper,
                references=[],
            )
            version_dir = Path(result["version_dir"])

            self.assertEqual(result["source_status"], "downloaded")
            self.assertEqual(result["tex_files"], ["main.tex"])
            self.assertEqual(
                [path.replace("\\", "/") for path in result["bib_files"]],
                ["refs/paper.bib"],
            )
            self.assertTrue((version_dir / "main.tex").is_file())
            self.assertTrue((version_dir / "refs" / "paper.bib").is_file())
            self.assertFalse((Path(temp_dir) / "outside.tex").exists())
            metadata = json.loads((version_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["references_status"], "success")
            self.assertEqual(metadata["references_count"], 0)

    @patch.object(arxiv_crawler.time, "sleep")
    @patch.object(arxiv_crawler.urllib.request, "urlopen")
    def test_get_source_does_not_mark_download_failure_as_empty_version(
        self, urlopen, _sleep
    ):
        urlopen.side_effect = OSError("network unavailable")

        with tempfile.TemporaryDirectory() as temp_dir, patch("builtins.print"):
            result = arxiv_crawler.get_source(
                "2501.12345v1", save_dir=temp_dir, paper=self.paper
            )

            self.assertEqual(result["source_status"], "download_failed")
            self.assertIn("network unavailable", result["error"])
            metadata_path = Path(result["version_dir"]) / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_status"], "download_failed")

    @patch.object(arxiv_crawler.time, "sleep")
    @patch.object(arxiv_crawler.urllib.request, "urlopen")
    def test_get_source_marks_invalid_archive_as_download_failure(
        self, urlopen, _sleep
    ):
        urlopen.return_value.__enter__.return_value.read.return_value = b"not an archive"

        with tempfile.TemporaryDirectory() as temp_dir:
            result = arxiv_crawler.get_source(
                "2501.12345v1", save_dir=temp_dir, paper=self.paper
            )

        self.assertEqual(result["source_status"], "download_failed")
        self.assertIn("Invalid source archive", result["error"])

    def test_get_source_resume_skips_previously_completed_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            version_dir = Path(temp_dir) / "2501.12345" / "v1"
            version_dir.mkdir(parents=True)
            (version_dir / "metadata.json").write_text(
                json.dumps({"source_status": "downloaded"}), encoding="utf-8"
            )

            with patch.object(arxiv_crawler.urllib.request, "urlopen") as urlopen:
                result = arxiv_crawler.get_source(
                    "2501.12345v1", save_dir=temp_dir, paper=self.paper
                )

            self.assertEqual(result["source_status"], "skipped")
            urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
