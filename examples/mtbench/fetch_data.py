"""Download the MT-Bench judgments: the same pairs judged by people and by GPT-4.

Two splits of ``lmsys/mt_bench_human_judgments``:

* ``human`` - 3,355 votes from 65 annotators.
* ``gpt4_pair`` - 2,400 verdicts from GPT-4 as a judge. This split was collected
  by asking GPT-4 both ways round; where the two orders disagreed the verdict is
  recorded as ``tie (inconsistent)``, which makes its rate of position
  inconsistency directly readable.

Needs ``pyarrow``, which arenakit itself does not depend on.
"""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/api/datasets/lmsys/mt_bench_human_judgments/parquet/default"
SPLITS = ("human", "gpt4_pair")
OUT = Path(__file__).parent / "data"


def main() -> None:
    import pyarrow.parquet as pq

    OUT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        raw = OUT / f"{split}.parquet"
        urllib.request.urlretrieve(f"{BASE}/{split}/0.parquet", raw)
        table = pq.read_table(
            raw, columns=["question_id", "model_a", "model_b", "winner", "judge", "turn"]
        ).to_pydict()
        path = OUT / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["model_a", "model_b", "winner", "voter_id", "item_id"])
            for row in zip(*(table[c] for c in table), strict=True):
                question, model_a, model_b, winner, judge, turn = row
                writer.writerow([model_a, model_b, winner, judge, f"{question}-{turn}"])
        raw.unlink()
        print(f"wrote {path} ({len(table['winner']):,} votes)")


if __name__ == "__main__":
    main()
