#!/usr/bin/env python3
import argparse
import json
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_TAG = "topk_eval/dense-vs-topk-20260308"


LOG_PREFIX_RE = re.compile(r"^\[[A-Z0-9 _\-]+\]")
EOS_RE = re.compile(
    r"(\[\s*end\s+of\s+text\s*\]|<\|endoftext\|>|<\|im_end\|>|<eos>|</s>)",
    re.IGNORECASE,
)
INLINE_LOG_RE = re.compile(r"\[[A-Z0-9 _\-]+\][^\[]*")
ENV_DUMP_LINE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
WS_RE = re.compile(r"\s+")


@dataclass
class CaseResult:
    prompt_id: str
    backend: str
    output: str
    eos_hit: bool
    token_count: int
    rep3: float


def parse_kv(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"invalid KEY=VALUE: {it}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def load_prompts(path: Path) -> List[Tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[Tuple[str, str]] = []
    for i, raw in enumerate(lines, start=1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append((f"p{i:03d}", s))
    if not out:
        raise ValueError(f"no prompts found in {path}")
    return out


def extract_generated_text(raw: str) -> str:
    kept: List[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("__PS_RC__:"):
            continue
        if ENV_DUMP_LINE_RE.match(s):
            continue
        if LOG_PREFIX_RE.match(s):
            continue
        s = INLINE_LOG_RE.sub(" ", s).strip()
        if not s:
            continue
        kept.append(s)
    out = "\n".join(kept)
    if out.strip():
        return out

    # Safety fallback: never return empty when raw has content.
    # Keep minimal cleanup only, so metrics do not collapse to all-zero.
    raw_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return "\n".join(raw_lines)


def repetition_3gram_ratio(text: str) -> float:
    toks = text.split()
    if len(toks) < 6:
        return 0.0
    grams = [" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)]
    uniq = set(grams)
    return 1.0 - (len(uniq) / len(grams))


def normalize_for_similarity(text: str) -> str:
    # For backend alignment, line break / spacing noise should not dominate score.
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    return WS_RE.sub(" ", s).strip()


def detect_eos(raw: str, extracted: str) -> bool:
    # Only use extracted text to avoid false positives from startup/debug logs
    # such as special token prints (<|endoftext|>, etc.).
    return bool(EOS_RE.search(extracted))


def run_once(
    adb: str,
    serial: str,
    remote_bin: str,
    work_folder: str,
    prompt: str,
    n_predicts: int,
    env: Dict[str, str],
) -> str:
    # Build one remote shell command with strict single-quote escaping.
    # `adb shell` still goes through `/system/bin/sh`, so prompt text like
    # "(LLMs)" must be quoted or it can trigger syntax errors.
    def _sq(s: str) -> str:
        return "'" + s.replace("'", "'\"'\"'") + "'"

    def _build_cmd_inline_env() -> List[str]:
        env_assign = " ".join(f"{k}={_sq(v)}" for k, v in env.items())
        inner = (
            f"{env_assign} "
            f"{_sq(remote_bin)} "
            f"--work-folder {_sq(work_folder)} "
            f"--prompt {_sq(prompt)} "
            f"--n-predicts {int(n_predicts)} 2>&1"
        )
        return [adb, "-s", serial, "shell", "sh", "-c", inner]

    def _build_cmd_export_env() -> List[str]:
        exports = " ; ".join(f"export {k}={_sq(v)}" for k, v in env.items())
        run_line = (
            f"{_sq(remote_bin)} "
            f"--work-folder {_sq(work_folder)} "
            f"--prompt {_sq(prompt)} "
            f"--n-predicts {int(n_predicts)} 2>&1"
        )
        inner = f"{exports} ; {run_line}"
        return [adb, "-s", serial, "shell", "sh", "-c", inner]

    def _clip(s: str, n: int = 2000) -> str:
        if len(s) <= n:
            return s
        return s[:n] + f"\n...<trimmed {len(s) - n} chars>"

    def _run(cmd: List[str]) -> Tuple[int, str, str, str]:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        stdout = p.stdout or ""
        stderr = p.stderr or ""
        out = stdout + ("\n" + stderr if stderr else "")
        return p.returncode, out, stdout, stderr

    attempts = [
        ("inline_env", _build_cmd_inline_env()),
        ("export_env", _build_cmd_export_env()),
    ]
    errors: List[str] = []

    for idx, (name, cmd) in enumerate(attempts, start=1):
        rc, out, stdout, stderr = _run(cmd)
        if rc != 0:
            errors.append(
                f"attempt#{idx} strategy={name} rc={rc}\n"
                f"cmd={cmd}\n"
                f"stdout_len={len(stdout)} stderr_len={len(stderr)}\n"
                f"out:\n{_clip(out)}"
            )
            continue

        if not out.strip():
            errors.append(
                f"attempt#{idx} strategy={name} empty output\n"
                f"cmd={cmd}\n"
                f"stdout_len={len(stdout)} stderr_len={len(stderr)}"
            )
            continue

        if n_predicts >= 32 and len(out.strip()) < 80:
            errors.append(
                f"attempt#{idx} strategy={name} suspiciously short output\n"
                f"cmd={cmd}\n"
                f"stdout_len={len(stdout)} stderr_len={len(stderr)}\n"
                f"out:\n{_clip(out)}"
            )
            continue

        if idx > 1:
            print(f"[topk_eval] run_once recovered with strategy={name}")
        return out

    raise RuntimeError(
        "run_once failed after trying all command strategies\n"
        + "\n\n".join(errors)
    )


def summarize(results: List[CaseResult], name: str) -> Dict[str, float]:
    xs = [r for r in results if r.backend == name]
    return {
        "count": float(len(xs)),
        "eos_rate": sum(1 for r in xs if r.eos_hit) / max(1, len(xs)),
        "avg_tokens": statistics.mean([r.token_count for r in xs]) if xs else 0.0,
        "avg_rep3": statistics.mean([r.rep3 for r in xs]) if xs else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adb", default="adb")
    ap.add_argument("--serial", required=True)
    ap.add_argument("--remote-bin", required=True)
    ap.add_argument("--work-folder", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n-predicts", type=int, default=128)
    ap.add_argument("--base-env", action="append", default=[])
    ap.add_argument("--dense-env", action="append", default=[])
    ap.add_argument("--topk-env", action="append", default=[])
    ap.add_argument("--topk-k", type=int, required=True)
    ap.add_argument("--sim-threshold", type=float, default=0.95)
    ap.add_argument("--out-json", default="topk_eval_report.json")
    ap.add_argument("--out-text", default="topk_eval_outputs.txt")
    args = ap.parse_args()
    print(f"[topk_eval] {SCRIPT_TAG} script={Path(__file__).resolve()}")

    if args.topk_k <= 0:
        raise ValueError("--topk-k must be > 0")

    prompts = load_prompts(Path(args.prompts))
    base_env = parse_kv(args.base_env)
    dense_env = parse_kv(args.dense_env)
    topk_env = parse_kv(args.topk_env)

    if "POWERSERVE_USE_OPENCL" in base_env:
        print("[topk_eval] warning: --base-env POWERSERVE_USE_OPENCL is ignored (force ggml)")
    if "POWERSERVE_USE_OPENCL" in dense_env:
        print("[topk_eval] warning: --dense-env POWERSERVE_USE_OPENCL is ignored (force ggml)")
    if "POWERSERVE_USE_OPENCL" in topk_env:
        print("[topk_eval] warning: --topk-env POWERSERVE_USE_OPENCL is ignored (force ggml)")
    if "POWERSERVE_GGML_TOPK" in dense_env and dense_env["POWERSERVE_GGML_TOPK"] not in ("", "0"):
        print("[topk_eval] warning: forcing dense mode to POWERSERVE_GGML_TOPK=0")
    if "POWERSERVE_GGML_TOPK" in topk_env and topk_env["POWERSERVE_GGML_TOPK"] != str(args.topk_k):
        print(f"[topk_eval] warning: overriding --topk-env POWERSERVE_GGML_TOPK to {args.topk_k}")

    env_dense = dict(base_env)
    env_dense.update(dense_env)
    env_dense.pop("POWERSERVE_USE_OPENCL", None)
    env_dense["POWERSERVE_GGML_TOPK"] = "0"

    env_topk = dict(base_env)
    env_topk.update(topk_env)
    env_topk.pop("POWERSERVE_USE_OPENCL", None)
    env_topk["POWERSERVE_GGML_TOPK"] = str(args.topk_k)

    all_results: List[CaseResult] = []
    pair_scores = []
    out_text_path = Path(args.out_text)
    out_text_path.write_text("", encoding="utf-8")

    for pid, prompt in prompts:
        raw_dense = run_once(
            args.adb, args.serial, args.remote_bin, args.work_folder, prompt, args.n_predicts, env_dense
        )
        out_dense = extract_generated_text(raw_dense)
        res_dense = CaseResult(
            prompt_id=pid,
            backend="dense",
            output=out_dense,
            eos_hit=detect_eos(raw_dense, out_dense),
            token_count=len(out_dense.split()),
            rep3=repetition_3gram_ratio(out_dense),
        )
        all_results.append(res_dense)

        raw_topk = run_once(
            args.adb, args.serial, args.remote_bin, args.work_folder, prompt, args.n_predicts, env_topk
        )
        out_topk = extract_generated_text(raw_topk)
        res_topk = CaseResult(
            prompt_id=pid,
            backend="topk",
            output=out_topk,
            eos_hit=detect_eos(raw_topk, out_topk),
            token_count=len(out_topk.split()),
            rep3=repetition_3gram_ratio(out_topk),
        )
        all_results.append(res_topk)

        out_dense_norm = normalize_for_similarity(out_dense)
        out_topk_norm = normalize_for_similarity(out_topk)
        sim_raw = SequenceMatcher(a=out_dense, b=out_topk, autojunk=False).ratio()
        sim = SequenceMatcher(a=out_dense_norm, b=out_topk_norm, autojunk=False).ratio()
        pair_scores.append({
            "prompt_id": pid,
            "char_similarity": sim,
            "char_similarity_raw": sim_raw,
            "dense_tokens": res_dense.token_count,
            "topk_tokens": res_topk.token_count,
            "dense_eos": res_dense.eos_hit,
            "topk_eos": res_topk.eos_hit,
            "dense_rep3": res_dense.rep3,
            "topk_rep3": res_topk.rep3,
            "sim_pass": sim >= args.sim_threshold,
        })
        with out_text_path.open("a", encoding="utf-8") as f:
            f.write(f"{pid}:\n")
            f.write("dense:\n")
            f.write(out_dense.strip() + "\n")
            f.write("topk:\n")
            f.write(out_topk.strip() + "\n")
            f.write("-" * 80 + "\n")
        print(
            f"{pid}: sim={sim:.4f} "
            f"sim_raw={sim_raw:.4f} "
            f"eos(d/t)={int(res_dense.eos_hit)}/{int(res_topk.eos_hit)} "
            f"rep3(d/t)={res_dense.rep3:.3f}/{res_topk.rep3:.3f} "
            f"tok(d/t)={res_dense.token_count}/{res_topk.token_count} "
            f"len_raw(d/t)={len(raw_dense)}/{len(raw_topk)} "
            f"len_txt(d/t)={len(out_dense)}/{len(out_topk)}"
        )

    s_dense = summarize(all_results, "dense")
    s_topk = summarize(all_results, "topk")
    report = {
        "summary": {
            "dense": s_dense,
            "topk": s_topk,
            "topk_k": args.topk_k,
            "avg_char_similarity": statistics.mean([x["char_similarity"] for x in pair_scores]) if pair_scores else 0.0,
            "avg_char_similarity_raw": statistics.mean([x["char_similarity_raw"] for x in pair_scores]) if pair_scores else 0.0,
            "sim_threshold": args.sim_threshold,
            "sim_pass_rate": (
                sum(1 for x in pair_scores if x["sim_pass"]) / len(pair_scores)
                if pair_scores else 0.0
            ),
            "sim_fail_count": sum(1 for x in pair_scores if not x["sim_pass"]),
        },
        "pairs": pair_scores,
    }

    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Summary ===")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report saved: {args.out_json}")
    print(f"text outputs saved: {args.out_text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
