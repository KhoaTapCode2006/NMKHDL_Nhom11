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

from src import arxiv_crawler, crawler_common


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
        crawler_common._reference_next_request_at = 0.0
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

    @patch.object(crawler_common.requests, "get")
    def test_get_paper_references_returns_api_references(self, get):
        get.return_value.status_code = 200
        get.return_value.json.return_value = {"references": [{"title": "Reference"}]}

        result = arxiv_crawler.get_paper_references("2501.12345v2")

        self.assertEqual(result, [{"title": "Reference"}])
        self.assertIn("2501.12345", get.call_args.args[0])

    @patch.object(crawler_common.requests, "get")
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

        with patch.object(crawler_common, "API_KEY", "test-key"), patch(
            "builtins.print"
        ), patch.object(
            arxiv_crawler.time, "monotonic", side_effect=lambda: clock[0]
        ), patch.object(arxiv_crawler.time, "sleep", side_effect=fake_sleep):
            result = arxiv_crawler.get_paper_references("2501.12345")

        self.assertIsNone(result)
        self.assertEqual(get.call_count, crawler_common.REFERENCE_MAX_RETRIES)
        expected = sum(
            crawler_common._reference_backoff(a)
            for a in range(crawler_common.REFERENCE_MAX_RETRIES - 1)
        )
        self.assertGreaterEqual(sum(sleep_durations), 0.9 * expected)

    @patch.object(crawler_common.requests, "get")
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
        with patch.object(crawler_common, "API_KEY", "test-key"), patch.object(
            arxiv_crawler.time, "monotonic", side_effect=lambda: clock[0]
        ), patch.object(arxiv_crawler.time, "sleep", side_effect=wake_early_once):
            arxiv_crawler.get_paper_references("2501.12345")
            arxiv_crawler.get_paper_references("2501.12346")

        self.assertGreaterEqual(
            request_times[1] - request_times[0],
            crawler_common.REFERENCE_REQUEST_INTERVAL,
        )
        self.assertGreaterEqual(sum(sleep_durations), crawler_common.REFERENCE_REQUEST_INTERVAL)
        self.assertGreater(len(sleep_durations), 1)

    @patch.object(crawler_common.requests, "get")
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
        with patch.object(crawler_common, "API_KEY", "test-key"), patch(
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

        self.assertGreaterEqual(crawler_common._retry_after_seconds(response), 29.0)

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

    @patch.object(arxiv_crawler.time, "sleep")
    @patch.object(arxiv_crawler.urllib.request, "urlopen")
    def test_get_source_caches_direct_pdf_as_version_without_tex_or_bib(
        self, urlopen, _sleep
    ):
        urlopen.return_value.__enter__.return_value.read.return_value = (
            b"%PDF-1.7\nbinary PDF response"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = arxiv_crawler.get_source(
                "2501.12345v1",
                save_dir=temp_dir,
                paper=self.paper,
            )
            version_dir = Path(result["version_dir"])
            metadata = json.loads(
                (version_dir / "metadata.json").read_text(encoding="utf-8")
            )

            self.assertEqual(result["source_status"], "no_tex_or_bib")
            self.assertEqual(metadata["source_status"], "no_tex_or_bib")
            self.assertEqual(result["tex_files"], [])
            self.assertEqual(result["bib_files"], [])

            skipped = arxiv_crawler.get_source(
                "2501.12345v1",
                save_dir=temp_dir,
                paper=self.paper,
            )

        self.assertEqual(skipped["source_status"], "skipped")
        urlopen.assert_called_once()

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

    # ------------------------------------------------------------------ #
    # Regression tests for the review fixes
    # ------------------------------------------------------------------ #
    @patch.object(crawler_common.requests, "get")
    def test_semantic_scholar_404_is_not_indexed_instead_of_crash(self, get):
        get.return_value = Mock(status_code=404, headers={})

        references, status = crawler_common.fetch_references("2510.99999")

        self.assertEqual(references, [])
        self.assertEqual(status, "not_indexed")

    @patch.object(crawler_common.requests, "get")
    def test_reference_request_asks_for_publication_date_and_venue(self, get):
        get.return_value.status_code = 200
        get.return_value.json.return_value = {"references": []}

        crawler_common.fetch_references("2510.10788")

        fields = get.call_args.kwargs["params"]["fields"]
        self.assertIn("references.publicationDate", fields)
        self.assertIn("references.venue", fields)

    @patch.object(crawler_common.requests, "get")
    def test_api_key_is_optional(self, get):
        get.return_value.status_code = 200
        get.return_value.json.return_value = {"references": [{"title": "R"}]}

        with patch.object(crawler_common, "API_KEY", None):
            references, status = crawler_common.fetch_references("2510.10788")

        self.assertEqual(status, "success")
        self.assertEqual(references, [{"title": "R"}])
        self.assertNotIn("x-api-key", get.call_args.kwargs["headers"])

    def test_metadata_uses_the_date_of_each_version(self):
        paper = SimpleNamespace(
            **{**vars(self.paper), "entry_id": "http://arxiv.org/abs/2501.12345v3"}
        )
        dates = {1: "2025-01-02T00:00:00", 2: "2025-01-20T00:00:00", 3: "2025-02-03T00:00:00"}

        v1 = arxiv_crawler.build_paper_metadata(paper, "2501.12345v1", dates)
        v2 = arxiv_crawler.build_paper_metadata(paper, "2501.12345v2", dates)

        self.assertEqual(v1["submission_date"], "2025-01-02T00:00:00")
        self.assertEqual(v1["version_date"], "2025-01-02T00:00:00")
        self.assertEqual(v2["version_date"], "2025-01-20T00:00:00")
        self.assertEqual(v2["revised_dates"], ["2025-01-20T00:00:00", "2025-02-03T00:00:00"])
        self.assertEqual(v2["venue"], "Journal")

    def test_discover_papers_reports_ids_the_api_did_not_return(self):
        found = SimpleNamespace(entry_id="http://arxiv.org/abs/2510.10788v2")
        with patch.object(arxiv_crawler.client, "results", return_value=[found]), patch(
            "builtins.print"
        ):
            papers, missing = arxiv_crawler.discover_papers(["2510.10788", "2510.10789"])

        self.assertEqual(list(papers), ["2510.10788"])
        self.assertEqual(missing, ["2510.10789"])

    def test_fetch_version_dates_queries_only_middle_versions(self):
        paper = SimpleNamespace(
            entry_id="http://arxiv.org/abs/2510.10788v3",
            published=datetime(2025, 10, 1),
            updated=datetime(2025, 10, 30),
        )
        middle = SimpleNamespace(
            entry_id="http://arxiv.org/abs/2510.10788v2",
            updated=datetime(2025, 10, 15),
        )
        with patch.object(arxiv_crawler.client, "results", return_value=[middle]) as results:
            dates = arxiv_crawler.fetch_version_dates({"2510.10788": paper})

        self.assertEqual(results.call_args.args[0].id_list, ["2510.10788v2"])
        self.assertEqual(
            dates["2510.10788"],
            {1: "2025-10-01T00:00:00", 2: "2025-10-15T00:00:00", 3: "2025-10-30T00:00:00"},
        )

    @patch.object(arxiv_crawler.time, "sleep")
    @patch.object(arxiv_crawler.urllib.request, "urlopen")
    def test_gzipped_non_tex_file_is_an_empty_record_not_a_failure(self, urlopen, _sleep):
        import gzip

        urlopen.return_value.__enter__.return_value.read.return_value = gzip.compress(
            b"<html>not a TeX submission</html>"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = arxiv_crawler.get_source("2501.12345v1", save_dir=temp_dir, paper=self.paper)

        self.assertEqual(result["source_status"], "no_tex_or_bib")

    # ------------------------------------------------------------------ #
    # Pacing: 3 s counted from the START of the previous request
    # ------------------------------------------------------------------ #
    def _run_downloads(self, durations):
        """Download len(durations) versions on a fake clock; return the request start times."""
        clock = [1000.0]
        starts, active, overlap = [], [0], [False]

        def fake_urlopen(*_args, **_kwargs):
            if active[0]:
                overlap[0] = True
            active[0] += 1
            starts.append(clock[0])
            clock[0] += durations[len(starts) - 1]  # time spent downloading
            active[0] -= 1
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.read.return_value = b"%PDF-1.7 pdf"
            return response

        crawler_common.reset_arxiv_pacer()
        with patch.object(
            crawler_common.time, "monotonic", side_effect=lambda: clock[0]
        ), patch.object(
            crawler_common.time, "sleep", side_effect=lambda s: clock.__setitem__(0, clock[0] + s)
        ), patch.object(crawler_common.urllib.request, "urlopen", side_effect=fake_urlopen):
            for index in range(len(durations)):
                crawler_common.download_source(f"2510.00001v{index + 1}")
        crawler_common.reset_arxiv_pacer()
        return starts, overlap[0]

    def test_three_seconds_are_counted_from_the_start_of_the_previous_request(self):
        # each download takes 1.5 s: the old "download, then sleep 3 s" rhythm gave 4.5 s
        starts, _ = self._run_downloads([1.5, 1.5, 1.5, 1.5])

        gaps = [b - a for a, b in zip(starts, starts[1:])]
        self.assertEqual(gaps, [3.0, 3.0, 3.0])

    def test_slow_download_is_followed_immediately_and_never_overlaps(self):
        starts, overlap = self._run_downloads([1.0, 5.0, 1.0])

        gaps = [b - a for a, b in zip(starts, starts[1:])]
        self.assertEqual(gaps, [3.0, 5.0])  # after the 5 s download no extra wait
        self.assertGreaterEqual(min(gaps), 3.0)
        self.assertFalse(overlap)

    def test_note_arxiv_request_makes_the_next_request_wait(self):
        clock = [50.0]
        crawler_common.reset_arxiv_pacer()
        with patch.object(
            crawler_common.time, "monotonic", side_effect=lambda: clock[0]
        ), patch.object(
            crawler_common.time, "sleep", side_effect=lambda s: clock.__setitem__(0, clock[0] + s)
        ):
            crawler_common.note_arxiv_request()  # e.g. the last arxiv.py API call
            crawler_common.wait_for_arxiv_slot()
        crawler_common.reset_arxiv_pacer()

        self.assertEqual(clock[0], 53.0)

    # ------------------------------------------------------------------ #
    # Background reference prefetching
    # ------------------------------------------------------------------ #
    def test_prefetcher_runs_ahead_but_only_by_the_lookahead(self):
        import threading
        import time as real_time

        fetched, lock = [], threading.Lock()

        def fetch(paper_id):
            with lock:
                fetched.append(paper_id)
            return [{"title": paper_id}], "success"

        prefetcher = crawler_common.ReferencePrefetcher(
            ["a", "b", "c", "d", "e"], fetch=fetch, lookahead=2
        ).start()
        try:
            deadline = real_time.time() + 5
            while len(fetched) < 2 and real_time.time() < deadline:
                real_time.sleep(0.01)
            real_time.sleep(0.15)
            self.assertEqual(fetched, ["a", "b"])  # stops at the lookahead

            self.assertEqual(prefetcher.get("a"), ([{"title": "a"}], "success"))
            deadline = real_time.time() + 5
            while len(fetched) < 3 and real_time.time() < deadline:
                real_time.sleep(0.01)
            self.assertEqual(fetched[:3], ["a", "b", "c"])
            self.assertTrue(prefetcher.has("e"))
            self.assertFalse(prefetcher.has("zzz"))
        finally:
            prefetcher.stop()

    def test_prefetcher_turns_a_failing_fetch_into_a_failed_result(self):
        def fetch(paper_id):
            if paper_id == "bad":
                raise RuntimeError("boom")
            return [], "success"

        prefetcher = crawler_common.ReferencePrefetcher(["bad", "good"], fetch=fetch).start()
        try:
            with patch("builtins.print"):
                self.assertEqual(prefetcher.get("bad"), (None, "failed"))
            self.assertEqual(prefetcher.get("good"), ([], "success"))
        finally:
            prefetcher.stop()

    def test_crawl_paper_uses_prefetched_references_without_asking_again(self):
        paper = SimpleNamespace(entry_id="http://arxiv.org/abs/2510.10788v1")

        class FakePrefetcher:
            def has(self, paper_id):
                return paper_id == "2510.10788"

            def get(self, paper_id):
                return [{"title": "prefetched"}], "success"

        captured = {}

        def fake_get_source(arxiv_id, **kwargs):
            captured.update(kwargs)
            return {"source_status": "downloaded"}

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            arxiv_crawler, "fetch_references", side_effect=AssertionError("must not fetch inline")
        ), patch.object(arxiv_crawler, "get_source", side_effect=fake_get_source), patch(
            "builtins.print"
        ):
            result = arxiv_crawler.crawl_paper(paper, temp_dir, True, None, FakePrefetcher())

        self.assertEqual(captured["references"], [{"title": "prefetched"}])
        self.assertEqual(result["references_status"], "success")

    def test_backoff_is_short_at_first_and_capped(self):
        delays = [crawler_common._reference_backoff(a) for a in range(crawler_common.REFERENCE_MAX_RETRIES)]

        self.assertLessEqual(delays[0], 2.0)  # a short retry is as good as a long wait
        self.assertLessEqual(delays[2], 5.0)
        self.assertEqual(max(delays), crawler_common.REFERENCE_BACKOFF_MAX)
        self.assertEqual(delays, sorted(delays))
        self.assertGreaterEqual(crawler_common.REFERENCE_MAX_RETRIES, 20)

    def test_papers_missing_references_are_prefetched_even_when_sources_are_done(self):
        done_with_refs = SimpleNamespace(entry_id="http://arxiv.org/abs/2510.00001v1")
        done_no_refs = SimpleNamespace(entry_id="http://arxiv.org/abs/2510.00002v1")
        fresh = SimpleNamespace(entry_id="http://arxiv.org/abs/2510.00003v1")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for pub in ("2510.00001", "2510.00002"):
                (root / pub / "v1").mkdir(parents=True)
                (root / pub / "v1" / "metadata.json").write_text(
                    json.dumps({"source_status": "downloaded"}), encoding="utf-8"
                )
            (root / "2510.00001" / "v1" / "references.json").write_text("[]", encoding="utf-8")

            self.assertFalse(arxiv_crawler.needs_references(done_with_refs, root))
            self.assertTrue(arxiv_crawler.needs_references(done_no_refs, root))
            self.assertTrue(arxiv_crawler.needs_references(fresh, root))
            self.assertIsNone(
                arxiv_crawler.start_reference_prefetcher([done_with_refs], root, with_refs=True)
            )
            self.assertIsNone(arxiv_crawler.start_reference_prefetcher([fresh], root, with_refs=False))

    def test_rerun_attaches_missing_references_without_downloading_sources(self):
        paper = SimpleNamespace(entry_id="http://arxiv.org/abs/2510.00002v2")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for version in (1, 2):
                version_dir = root / "2510.00002" / f"v{version}"
                version_dir.mkdir(parents=True)
                (version_dir / "metadata.json").write_text(
                    json.dumps({"source_status": "downloaded", "references_status": "failed",
                                "references_count": 0}),
                    encoding="utf-8",
                )
            refs = [{"title": "R", "externalIds": {"ArXiv": "2301.01234"}}, {"title": "S"}]
            with patch.object(arxiv_crawler, "fetch_references", return_value=(refs, "success")),                     patch.object(arxiv_crawler, "get_source", side_effect=AssertionError("no download")),                     patch("builtins.print"):
                result = arxiv_crawler.crawl_paper(paper, root, True)

            for version in (1, 2):
                version_dir = root / "2510.00002" / f"v{version}"
                meta = json.loads((version_dir / "metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(json.loads((version_dir / "references.json").read_text("utf-8")), refs)
                self.assertEqual(meta["references_status"], "success")
                self.assertEqual(meta["references_count"], 2)
                self.assertEqual(meta["arxiv_references_count"], 1)
                self.assertEqual(meta["source_status"], "downloaded")
        self.assertEqual(result["references_status"], "success")
        self.assertFalse(result["skipped"])

    def test_rerun_with_no_refs_skips_a_paper_whose_sources_are_done(self):
        paper = SimpleNamespace(entry_id="http://arxiv.org/abs/2510.00002v1")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "2510.00002" / "v1").mkdir(parents=True)
            (root / "2510.00002" / "v1" / "metadata.json").write_text(
                json.dumps({"source_status": "downloaded"}), encoding="utf-8"
            )
            with patch.object(arxiv_crawler, "fetch_references", side_effect=AssertionError("no fetch")),                     patch("builtins.print"):
                result = arxiv_crawler.crawl_paper(paper, root, with_refs=False)
        self.assertTrue(result["skipped"])

    def test_member_ids_are_read_from_members_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            members = Path(temp_dir) / "members.json"
            members.write_text(
                json.dumps({
                    "a": {"year_month": "2510", "start": 1, "end": 2},
                    "b": {"ids": ["2510.00009v2"]},
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                crawler_common.load_member_ids(members, "a"),
                ["2510.00001", "2510.00002"],
            )
            self.assertEqual(
                crawler_common.load_member_ids(members),
                ["2510.00001", "2510.00002", "2510.00009"],
            )


if __name__ == "__main__":
    unittest.main()
