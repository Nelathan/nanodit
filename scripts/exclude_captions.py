"""The exclusion list of a training set: its captions checked for the defects seen in its longest rows: a run of one character or of a short
unit repeated (a captioner loop, ASCII art), CJK in a caption (quoted text to render and code-switching alike), UTF-8 written as escaped bytes (`\\xNN`),
a leaked chat template, and an empty caption. Counts per defect and source and examples of each go to OUT.json; the
sorted keys of every flagged row go to OUT.json and to the set's exclusion list, `SET_DIR/excluded_keys.txt`
(`nanodit.data.EXCLUDED`, SPEC §1), which it replaces. Run it after every shuffle.

usage: exclude_captions.py SET_DIR OUT.json
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from nanodit.data import EXCLUDED

EXAMPLES = 40
DEFECTS = {
    "repeat": re.compile(r"(.{1,8}?)\1{11,}", re.S),  # a unit of up to 8 characters, 12 times or more in a row
    "cjk": re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]"),
    "escaped_bytes": re.compile(r"\\x[0-9a-fA-F]{2}"),
    "chat_template": re.compile(r"(?:^|\n)(?:assistant|user|system)\n|<\|im_(?:start|end)\|>|<start_of_turn>"),
}


def main(set_dir: str, out: str) -> None:
    counts, examples = defaultdict(Counter), defaultdict(list)
    rows, flagged = 0, set()
    for path in sorted(Path(set_dir).glob("*.parquet")):
        f = pq.ParquetFile(path)
        for group in range(f.metadata.num_row_groups):
            table = f.read_row_group(group, columns=["key", "source", "caption"])
            for key, source, caption in zip(*(table[c].to_pylist() for c in ("key", "source", "caption"))):
                rows += 1
                found = [name for name, pattern in DEFECTS.items() if pattern.search(caption)]
                if not caption.strip():
                    found.append("empty")
                if found:
                    flagged.add(key)
                for name in found:
                    counts[name][source] += 1
                    if len(examples[name]) < EXAMPLES:
                        examples[name].append(dict(key=key, source=source, file=path.name, chars=len(caption), caption=caption[:600]))
        print(path.name, flush=True)
    report = dict(rows=rows, counts={k: dict(v) | dict(total=sum(v.values())) for k, v in counts.items()}, flagged=len(flagged), examples=examples, keys=sorted(flagged))
    Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (Path(set_dir) / EXCLUDED).write_text("".join(f"{k}\n" for k in report["keys"]))
    print(json.dumps(report["counts"], indent=1), report["flagged"], "rows flagged")


if __name__ == "__main__":
    main(*sys.argv[1:])
