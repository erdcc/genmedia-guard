"""Tiny load generator for the demo (stdlib only).

    python -m genmedia_guard.loadgen --url http://localhost:8000 --rps 20 --duration 300
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from collections import Counter


def _post(url: str, payload: dict) -> int:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except OSError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rps", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=300.0, help="seconds")
    args = parser.parse_args()

    codes: Counter = Counter()
    lock = threading.Lock()

    def fire(i: int) -> None:
        code = _post(f"{args.url}/generate", {"prompt": "a lighthouse at dawn", "seed": i})
        with lock:
            codes[code] += 1

    start, i = time.monotonic(), 0
    threads = []
    while time.monotonic() - start < args.duration:
        t = threading.Thread(target=fire, args=(i,), daemon=True)
        t.start()
        threads.append(t)
        i += 1
        time.sleep(1.0 / args.rps)
    for t in threads:
        t.join(timeout=20)
    print(f"sent {i} requests; status codes: {dict(codes)}")


if __name__ == "__main__":
    main()
