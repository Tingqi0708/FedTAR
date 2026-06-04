import json
import random
from pathlib import Path
from typing import List

import torch

# =========================================================
# Basic utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_seconds(sec: float) -> str:
    sec = max(float(sec), 0.0)
    if sec < 60:
        return f"{sec:.0f}s"
    mins = sec / 60.0
    if mins < 60:
        return f"{mins:.1f}m"
    hours = mins / 60.0
    return f"{hours:.2f}h"


def safe_norm(x: torch.Tensor) -> float:
    return float(torch.norm(x.float()).item())

