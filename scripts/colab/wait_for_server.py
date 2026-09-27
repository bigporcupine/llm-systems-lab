"""Wait until an HTTP health endpoint is ready or fail with a timeout."""

import argparse
import time
from urllib import error, request


parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True)
parser.add_argument("--timeout-seconds", type=float, default=600)
args = parser.parse_args()

deadline = time.monotonic() + args.timeout_seconds
last_error = "not attempted"
while time.monotonic() < deadline:
    try:
        with request.urlopen(args.url, timeout=5) as response:
            if response.status == 200:
                print(f"Ready: {args.url}")
                raise SystemExit(0)
    except (error.URLError, TimeoutError) as exc:
        last_error = str(exc)
    time.sleep(2)

raise SystemExit(f"Server did not become ready: {last_error}")

