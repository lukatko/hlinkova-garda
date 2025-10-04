#!/usr/bin/env python3
"""Submit a submission JSON file to the hackathon server.

Usage examples:
  python scripts/submit_submission.py --file data/sample_submission.json
  python scripts/submit_submission.py --file my_answers.json --endpoint submit
  python scripts/submit_submission.py --file out.json --dry-run

The script POSTs the file under the multipart form field `file` which matches
the API expectations of the server.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import requests


DEFAULT_BASE_URL = (
    "https://hackathon-server.ashysand-de33d6c5.westeurope.azurecontainerapps.io"
)


def submit_file(
    file_path: str,
    base_url: str = DEFAULT_BASE_URL,
    endpoint: str = "check-submission",
    timeout: Optional[float] = 30.0,
) -> int:
    """Submit a JSON file to the hackathon server.

    Returns 0 on success, non-zero on failure.
    """
    if not os.path.isfile(file_path):
        print(f"ERROR: file not found: {file_path}", file=sys.stderr)
        return 2

    url = base_url.rstrip("/") + "/" + endpoint.lstrip("/")

    print(f"Submitting '{file_path}' -> {url}")

    with open(file_path, "rb") as fh:
        files = {"file": (os.path.basename(file_path), fh, "application/json")}
        headers = {"accept": "application/json"}
        try:
            resp = requests.post(url, headers=headers, files=files, timeout=timeout)
        except requests.RequestException as exc:
            print(f"Request failed: {exc}", file=sys.stderr)
            return 3

    status = resp.status_code
    print(f"HTTP {status}")
    try:
        data = resp.json()
        # pretty print JSON if possible
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except ValueError:
        # not JSON
        print(resp.text)

    if 200 <= status < 300:
        return 0
    return 4


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Upload a submission JSON to the hackathon API")
    parser.add_argument("--file", "-f", required=True, help="Path to the submission JSON file")
    parser.add_argument(
        "--base-url",
        "-b",
        default=DEFAULT_BASE_URL,
        help="Base URL of the hackathon server (default uses the public server)",
    )
    parser.add_argument(
        "--endpoint",
        "-e",
        default="check-submission",
        choices=("check-submission", "submit"),
        help="Submission endpoint to call (default: check-submission)",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually send the request; show what would be done")

    args = parser.parse_args(argv)

    if args.dry_run:
        if not os.path.isfile(args.file):
            print(f"ERROR: file not found: {args.file}", file=sys.stderr)
            return 2
        size = os.path.getsize(args.file)
        print("DRY-RUN: would POST file")
        print(f"  file: {args.file} ({size} bytes)")
        print(f"  url: {args.base_url.rstrip('/')}/{args.endpoint}")
        return 0

    return submit_file(args.file, base_url=args.base_url, endpoint=args.endpoint, timeout=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
