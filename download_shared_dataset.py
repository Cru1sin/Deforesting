"""Download password-protected Midea DTS ZIP files with automatic resume."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import IncompleteRead
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

LIST_URL = "https://dts.midea.com:8066/v1/dtp/service/share/file/list"
DOWNLOAD_URL = "https://dts.midea.com:8066/v1/dtp/service/share/down"
USER_AGENT = "Mozilla/5.0"
CHUNK_BYTES = 8 * 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="DTS download page or Outlook Safe Links URL")
    parser.add_argument("--password", help="omit to enter it without saving it in shell history")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--start-no", type=int, default=1)
    parser.add_argument("--end-no", type=int)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--retry-delay", type=float, default=5)
    return parser


def share_identity(url: str) -> tuple[str, str]:
    parsed = urlsplit(url)
    wrapped = parse_qs(parsed.query).get("url")
    if wrapped:
        parsed = urlsplit(wrapped[0])
    parts = parsed.fragment.strip("/").split("/")
    if "info" not in parts or len(parts) < parts.index("info") + 3:
        raise ValueError("URL does not contain a DTS share key and stamp")
    position = parts.index("info")
    return parts[position + 1], parts[position + 2]


def post_json(url: str, payload: dict[str, object]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=60) as response:
        body = json.load(response)
    if str(body.get("errorCode")) != "200":
        raise RuntimeError(str(body.get("msg") or "DTS request failed"))
    return body["result"]


def list_shared_files(url: str, password: str) -> list[dict[str, Any]]:
    share_key, stamp = share_identity(url)
    result = post_json(
        f"{LIST_URL}?sharekey={share_key}&stamp={stamp}",
        {"password": password},
    )
    return list(result["fileList"])


def plan_downloads(
    files: list[dict[str, Any]],
    data_dir: Path,
    *,
    start_no: int,
    end_no: int | None,
) -> list[dict[str, Any]]:
    end_no = len(files) if end_no is None else end_no
    if not 1 <= start_no <= end_no <= len(files):
        raise ValueError(f"No. range must be within 1-{len(files)}")

    jobs: list[dict[str, Any]] = []
    for group_start in range(0, len(files), 6):
        group = files[group_start : group_start + 6]
        marker = next(
            (
                match.group(1)
                for file in group
                if (match := re.fullmatch(r"(\d{4})S\.zip", str(file["filename"]), re.I))
            ),
            f"batch_{group_start + 1:03d}",
        )
        for offset, file in enumerate(group):
            number = group_start + offset + 1
            if start_no <= number <= end_no:
                filename = Path(str(file["filename"])).name
                jobs.append(
                    {
                        **file,
                        "number": number,
                        "destination": data_dir / marker / filename,
                    }
                )
    return jobs


def request_download_url(file: dict[str, Any]) -> str:
    result = post_json(
        DOWNLOAD_URL,
        {
            "spId": file["spId"],
            "key": file["md5"],
            "cephId": file["cephId"],
            "trackId": file["trackId"],
            "fileName": file["filename"],
            "recUserNum": file["uid"],
            "fileSize": file["filesize"],
        },
    )
    return str(result["realshareurl"]).replace(
        "https://dts.midea.com/", "https://dts.midea.com:8066/", 1
    )


def download_file(file: dict[str, Any], destination: Path, retry_delay: float) -> Path:
    expected = int(file["filesize"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(f"{destination.name}.part")
    label = f"No.{file['number']} {destination.parent.name}/{destination.name}"
    if destination.is_file() and destination.stat().st_size == expected:
        print(f"[skip] {label}", flush=True)
        return destination

    while True:
        offset = part.stat().st_size if part.exists() else 0
        try:
            headers = {"User-Agent": USER_AGENT}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = Request(request_download_url(file), headers=headers)
            with urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    part.unlink()
                    raise RuntimeError("server ignored Range; restarting this ZIP")
                mode = "ab" if offset else "wb"
                written = offset
                next_report = written + 256 * 1024 * 1024
                with part.open(mode) as output:
                    while chunk := response.read(CHUNK_BYTES):
                        output.write(chunk)
                        written += len(chunk)
                        if written >= next_report:
                            print(
                                f"[data] {label}: {written / 1024**3:.2f} / "
                                f"{expected / 1024**3:.2f} GiB",
                                flush=True,
                            )
                            next_report = written + 256 * 1024 * 1024
            if written != expected:
                raise IncompleteRead(b"", expected - written)
            part.replace(destination)
            print(f"[done] {label}", flush=True)
            return destination
        except (HTTPError, URLError, TimeoutError, OSError, IncompleteRead, RuntimeError) as error:
            print(
                f"[retry] {label}: {type(error).__name__}; resume at "
                f"{(part.stat().st_size if part.exists() else 0) / 1024**3:.2f} GiB",
                flush=True,
            )
            time.sleep(retry_delay)


def run(args: argparse.Namespace) -> int:
    password = args.password or os.environ.get("MIDEA_DTS_PASSWORD") or getpass.getpass(
        "Download password: "
    )
    files = list_shared_files(args.url, password)
    jobs = plan_downloads(files, args.data, start_no=args.start_no, end_no=args.end_no)
    total_gib = sum(int(job["filesize"]) for job in jobs) / 1024**3
    print(f"[plan] {len(jobs)} ZIPs, {total_gib:.2f} GiB, workers={args.workers}", flush=True)
    for job in jobs:
        print(f"[plan] No.{job['number']} -> {job['destination']}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_file, job, job["destination"], args.retry_delay)
            for job in jobs
        ]
        for future in futures:
            future.result()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
