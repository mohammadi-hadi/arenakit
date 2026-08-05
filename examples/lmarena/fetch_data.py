"""Download the public LMArena preference sample, reading only what is needed.

The published file is a 380 MB parquet, and 99% of it is conversation text this
audit never looks at. Parquet stores columns separately and puts a footer at the
end describing where each one lives, so with HTTP range requests the six columns
that matter can be pulled without the rest: about 7 MB instead of 380.

Run this first; :mod:`audit_lmarena` reads the CSV it writes.

Needs ``pyarrow``, which arenakit itself does not depend on:

    pip install pyarrow
"""

from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

DATASET = "lmarena-ai/arena-human-preference-100k"
URL = (
    "https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k/"
    "resolve/main/data/arena-explorer-preference-100k.parquet"
)
COLUMNS = ["model_a", "model_b", "winner", "anony", "judge_hash", "question_id"]
OUT = Path(__file__).parent / "data" / "battles.csv"


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file backed by HTTP range requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.pos = 0
        self.downloaded = 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        if size is None or size < 0:
            size = self.size - self.pos
        if size == 0:
            return b""
        last = min(self.pos + size, self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{last}"})
        with urllib.request.urlopen(request) as response:
            chunk = response.read()
        self.pos += len(chunk)
        self.downloaded += len(chunk)
        return chunk


def main() -> None:
    import pyarrow.parquet as pq

    OUT.parent.mkdir(parents=True, exist_ok=True)
    handle = HttpRangeFile(URL)
    table = pq.ParquetFile(handle).read(columns=COLUMNS)
    print(
        f"read {table.num_rows:,} rows of {len(COLUMNS)} columns from {DATASET} "
        f"({handle.downloaded / 1e6:.1f} MB downloaded of {handle.size / 1e6:.0f} MB)"
    )
    data = table.to_pydict()
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model_a", "model_b", "winner", "voter_id", "item_id", "anonymous"])
        for row in zip(*(data[column] for column in COLUMNS), strict=True):
            model_a, model_b, winner, anony, judge, question = row
            writer.writerow([model_a, model_b, winner, judge, question, bool(anony)])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
