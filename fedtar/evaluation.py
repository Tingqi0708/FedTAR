import multiprocessing as mp
import re
from pathlib import Path
from typing import List

import torch

from fedtar.utils import load_jsonl

# =========================================================
# Eval helpers
# =========================================================
def _normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_numeric_string(s: str) -> str:
    s = str(s).strip()
    s = s.replace(",", "")
    s = s.rstrip(".")
    if s == "":
        return ""
    try:
        x = float(s)
        if abs(x - int(x)) < 1e-9:
            return str(int(x))
        return str(x)
    except Exception:
        return s


def _extract_gsm8k_final_answer(s: str) -> str:
    s = str(s).strip()

    m = re.findall(r"####\s*([-+]?\d[\d,\.]*)", s)
    if m:
        return _normalize_numeric_string(m[-1])

    nums = re.findall(r"[-+]?\d[\d,\.]*", s)
    if nums:
        return _normalize_numeric_string(nums[-1])

    return _normalize_text(s)


def rouge_l_score(pred: str, ref: str) -> float:
    pred_toks = _normalize_text(pred).split()
    ref_toks = _normalize_text(ref).split()
    if len(pred_toks) == 0 or len(ref_toks) == 0:
        return 0.0

    dp = [[0] * (len(ref_toks) + 1) for _ in range(len(pred_toks) + 1)]
    for i in range(1, len(pred_toks) + 1):
        for j in range(1, len(ref_toks) + 1):
            if pred_toks[i - 1] == ref_toks[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    prec = lcs / max(len(pred_toks), 1)
    rec = lcs / max(len(ref_toks), 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def generate_text(tokenizer, model, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0) -> str:
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    if temperature <= 1e-8:
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()


@torch.no_grad()
def gsm8k_em(tokenizer, model, rows: List[dict], n_samples: int = 0, max_new_tokens: int = 256) -> float:
    if n_samples and n_samples > 0:
        rows = rows[:n_samples]
    if len(rows) == 0:
        return 0.0

    correct = 0
    for ex in rows:
        pred = generate_text(tokenizer, model, ex["prompt"], max_new_tokens=max_new_tokens)
        pred_ans = _extract_gsm8k_final_answer(pred)
        gold_ans = _extract_gsm8k_final_answer(ex["answer"])
        correct += int(pred_ans == gold_ans)
    return correct / len(rows)


@torch.no_grad()
def text_rougeL(tokenizer, model, rows: List[dict], n_samples: int = 0, max_new_tokens: int = 128) -> float:
    if n_samples and n_samples > 0:
        rows = rows[:n_samples]
    if len(rows) == 0:
        return 0.0

    vals = []
    for ex in rows:
        pred = generate_text(tokenizer, model, ex["prompt"], max_new_tokens=max_new_tokens)
        vals.append(rouge_l_score(pred, ex["answer"]))
    return sum(vals) / len(vals)


def _extract_mmlu_letter(s: str) -> str:
    s = str(s).strip().upper()

    patterns = [
        r"ANSWER\s*[:：]?\s*([ABCD])\b",
        r"THE ANSWER IS\s*([ABCD])\b",
        r"OPTION\s*([ABCD])\b",
        r"\b([ABCD])\b",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            return m.group(1)

    if len(s) > 0 and s[0] in "ABCD":
        return s[0]
    return ""


@torch.no_grad()
def mmlu_acc(tokenizer, model, rows: List[dict], n_samples: int = 0, max_new_tokens: int = 32) -> float:
    if n_samples and n_samples > 0:
        rows = rows[:n_samples]
    if len(rows) == 0:
        return 0.0

    correct = 0
    for ex in rows:
        pred = generate_text(tokenizer, model, ex["prompt"], max_new_tokens=max_new_tokens)
        p = _extract_mmlu_letter(pred)
        g = _extract_mmlu_letter(ex["answer"])
        correct += int(p == g)
    return correct / len(rows)


def _normalize_bbh_answer(s: str) -> str:
    s = _normalize_text(s)
    s = s.strip(" .,:;!?\"'`()[]{}")
    return s


@torch.no_grad()
def bbh_em(tokenizer, model, rows: List[dict], n_samples: int = 0, max_new_tokens: int = 128) -> float:
    if n_samples and n_samples > 0:
        rows = rows[:n_samples]
    if len(rows) == 0:
        return 0.0

    correct = 0
    for ex in rows:
        pred = generate_text(tokenizer, model, ex["prompt"], max_new_tokens=max_new_tokens)
        p = _normalize_bbh_answer(pred)
        g = _normalize_bbh_answer(ex["answer"])

        ok = (p == g)
        if not ok and len(g) > 0:
            ok = (g in p)

        correct += int(ok)
    return correct / len(rows)


SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "callable": callable,
    "chr": chr, "complex": complex, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float, "hash": hash,
    "hex": hex, "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map, "max": max, "min": min,
    "next": next, "ord": ord, "pow": pow, "print": print, "range": range,
    "repr": repr, "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip,
}


def extract_python_code(text: str) -> str:
    fenced = re.findall(r"```python\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        return fenced[0].strip()
    fenced2 = re.findall(r"```\s*(.*?)```", text, flags=re.S)
    if fenced2:
        return fenced2[0].strip()
    return text.strip()


def _run_mbpp_case_worker(code: str, tests: List[str], queue: mp.Queue):
    try:
        import collections
        import functools
        import itertools
        import math as math_mod
        import re as re_mod

        safe_globals = {
            "__builtins__": SAFE_BUILTINS,
            "math": math_mod,
            "re": re_mod,
            "itertools": itertools,
            "functools": functools,
            "collections": collections,
        }
        safe_locals = {}

        exec(code, safe_globals, safe_locals)
        for t in tests:
            exec(t, safe_globals, safe_locals)

        queue.put((True, "ok"))
    except Exception as e:
        queue.put((False, repr(e)))


def run_mbpp_tests(code: str, tests: List[str], timeout_sec: int = 3) -> bool:
    if not code.strip() or tests is None or len(tests) == 0:
        return False

    q = mp.Queue()
    p = mp.Process(target=_run_mbpp_case_worker, args=(code, tests, q))
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()
        return False

    if q.empty():
        return False

    ok, _ = q.get()
    return bool(ok)


@torch.no_grad()
def mbpp_pass_at_1(
    tokenizer,
    model,
    rows: List[dict],
    n_samples: int = 0,
    max_new_tokens: int = 256,
    timeout_sec: int = 3,
) -> float:
    if n_samples and n_samples > 0:
        rows = rows[:n_samples]
    if len(rows) == 0:
        return 0.0

    passed = 0
    total = 0
    for ex in rows:
        tests = ex.get("tests", ex.get("test_list", None))
        if tests is None:
            continue

        pred = generate_text(tokenizer, model, ex["prompt"], max_new_tokens=max_new_tokens)
        code = extract_python_code(pred)
        ok = run_mbpp_tests(code, tests, timeout_sec=timeout_sec)
        passed += int(ok)
        total += 1

    if total == 0:
        raise RuntimeError("MBPP eval rows do not contain tests/test_list; pass@1 cannot be computed.")

    return passed / total


def eval_single_domain(
    domain: str,
    tokenizer,
    model,
    fixed_eval_dir: str | Path,
    n_samples: int = 0,
    split: str = "test",
):
    fixed_eval_dir = Path(fixed_eval_dir)

    def choose_file(test_name: str, val_name: str) -> Path:
        test_path = fixed_eval_dir / test_name
        val_path = fixed_eval_dir / val_name

        if split == "test" and test_path.exists():
            return test_path
        if split == "val" and val_path.exists():
            return val_path
        if test_path.exists():
            return test_path
        if val_path.exists():
            return val_path

        raise FileNotFoundError(
            f"Neither {test_name} nor {val_name} exists under {fixed_eval_dir}"
        )

    if domain == "gsm8k":
        rows = load_jsonl(choose_file("gsm8k_test.jsonl", "gsm8k_val.jsonl"))
        return "EM", gsm8k_em(tokenizer, model, rows, n_samples=n_samples, max_new_tokens=256)

    if domain == "dolly":
        rows = load_jsonl(choose_file("dolly_test.jsonl", "dolly_val.jsonl"))
        return "ROUGE-L", text_rougeL(tokenizer, model, rows, n_samples=n_samples, max_new_tokens=128)

    if domain == "mbpp":
        rows = load_jsonl(choose_file("mbpp_test.jsonl", "mbpp_val.jsonl"))
        return "pass@1", mbpp_pass_at_1(tokenizer, model, rows, n_samples=n_samples, max_new_tokens=256, timeout_sec=3)

    if domain == "mmlu":
        rows = load_jsonl(choose_file("mmlu_test.jsonl", "mmlu_val.jsonl"))
        return "ACC", mmlu_acc(tokenizer, model, rows, n_samples=n_samples, max_new_tokens=32)

    if domain == "bbh":
        rows = load_jsonl(choose_file("bbh_test.jsonl", "bbh_val.jsonl"))
        return "EM", bbh_em(tokenizer, model, rows, n_samples=n_samples, max_new_tokens=128)

    raise ValueError(domain)

