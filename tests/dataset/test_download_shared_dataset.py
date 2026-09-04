from __future__ import annotations

from http.client import IncompleteRead
from pathlib import Path
from urllib.parse import quote

import download_shared_dataset as downloader


def test_plan_no_7_to_12_as_0807(tmp_path: Path) -> None:
    direct = (
        "https://dts.midea.com:8066/download.html?lang=en"
        "#/home/fronthome/download/info/share-key/stamp-value"
    )
    safe_link = f"https://example.safelinks.protection.outlook.com/?url={quote(direct)}"
    assert downloader.share_identity(safe_link) == ("share-key", "stamp-value")

    names = [
        "0806S.zip",
        "192.168.1.1_1.zip",
        "192.168.1.2_1.zip",
        "192.168.1.11_1.zip",
        "192.168.1.12_1.zip",
        "192.168.1.14_1.zip",
        "192.168.1.2_1.zip",
        "192.168.1.11_1.zip",
        "192.168.1.12_1.zip",
        "192.168.1.14_1.zip",
        "192.168.1.15_1.zip",
        "0807S.zip",
    ]
    files = [{"filename": name, "filesize": index} for index, name in enumerate(names, 1)]

    jobs = downloader.plan_downloads(files, tmp_path, start_no=7, end_no=12)

    assert [job["number"] for job in jobs] == list(range(7, 13))
    assert [job["destination"].parent.name for job in jobs] == ["0807"] * 6
    assert jobs[0]["destination"] == tmp_path / "0807" / "192.168.1.2_1.zip"
    assert jobs[-1]["destination"] == tmp_path / "0807" / "0807S.zip"


def test_download_resumes_part_file(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "0807" / "file.zip"
    ranges: list[str | None] = []

    class Response:
        def __init__(self, chunks: list[bytes | Exception], status: int) -> None:
            self.chunks = iter(chunks)
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, _size: int) -> bytes:
            chunk = next(self.chunks, b"")
            if isinstance(chunk, Exception):
                raise chunk
            return chunk

    responses = iter(
        [
            Response([b"abc", IncompleteRead(b"", 3)], 200),
            Response([b"def", b""], 206),
        ]
    )

    def fake_urlopen(request, timeout):
        del timeout
        ranges.append(request.get_header("Range"))
        return next(responses)

    monkeypatch.setattr(downloader, "request_download_url", lambda _file: "https://file")
    monkeypatch.setattr(downloader, "urlopen", fake_urlopen)

    downloader.download_file(
        {"number": 7, "filename": "file.zip", "filesize": 6},
        destination,
        retry_delay=0,
    )

    assert destination.read_bytes() == b"abcdef"
    assert not destination.with_suffix(".zip.part").exists()
    assert ranges == [None, "bytes=3-"]


def test_two_downloads_is_the_default() -> None:
    args = downloader.build_parser().parse_args(
        ["https://example.com/#/info/share/stamp", "--password", "secret"]
    )
    assert args.workers == 2
