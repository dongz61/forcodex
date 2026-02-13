#!/usr/bin/env python3
import argparse
import json
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple


LOG_PREFIX_RE = re.compile(r"^\[[A-Z0-9 _\-]+\]")
EOS_PATTERNS = ("[end of text]", "<|endoftext|>", "<eos>", "</s>")


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
        if LOG_PREFIX_RE.match(s):
            continue
        kept.append(s)
    if not kept:
        return ""
    return "\n".join(kept)


def repetition_3gram_ratio(text: str) -> float:
    toks = text.split()
    if len(toks) < 6:
        return 0.0
    grams = [" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)]
    uniq = set(grams)
    return 1.0 - (len(uniq) / len(grams))


def run_once(
    adb: str,
    serial: str,
    remote_bin: str,
    work_folder: str,
    prompt: str,
    n_predicts: int,
    env: Dict[str, str],
) -> str:
    exports = " ; ".join([f"export {k}={shlex.quote(v)}" for k, v in env.items()])
    inner = (
        f"{exports} ; "
        f"{shlex.quote(remote_bin)} "
        f"--work-folder {shlex.quote(work_folder)} "
        f"--prompt {shlex.quote(prompt)} "
        f"--n-predicts {n_predicts}"
    )
    cmd = [adb, "-s", serial, "shell", "sh", "-lc", inner]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}\n{out}")
    return out


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
    ap.add_argument("--opencl-env", action="append", default=[])
    ap.add_argument("--ggml-env", action="append", default=[])
    ap.add_argument("--out-json", default="backend_eval_report.json")
    args = ap.parse_args()

    prompts = load_prompts(Path(args.prompts))
    base_env = parse_kv(args.base_env)
    opencl_env = parse_kv(args.opencl_env)
    ggml_env = parse_kv(args.ggml_env)

    env_opencl = dict(base_env)
    env_opencl["POWERSERVE_USE_OPENCL"] = "1"
    env_opencl.update(opencl_env)

    env_ggml = dict(base_env)
    env_ggml["POWERSERVE_USE_OPENCL"] = "0"
    env_ggml.update(ggml_env)

    all_results: List[CaseResult] = []
    pair_scores = []

    for pid, prompt in prompts:
        raw_g = run_once(
            args.adb, args.serial, args.remote_bin, args.work_folder, prompt, args.n_predicts, env_ggml
        )
        out_g = extract_generated_text(raw_g)
        res_g = CaseResult(
            prompt_id=pid,
            backend="ggml",
            output=out_g,
            eos_hit=any(p in out_g.lower() for p in EOS_PATTERNS),
            token_count=len(out_g.split()),
            rep3=repetition_3gram_ratio(out_g),
        )
        all_results.append(res_g)

        raw_o = run_once(
            args.adb, args.serial, args.remote_bin, args.work_folder, prompt, args.n_predicts, env_opencl
        )
        out_o = extract_generated_text(raw_o)
        res_o = CaseResult(
            prompt_id=pid,
            backend="opencl",
            output=out_o,
            eos_hit=any(p in out_o.lower() for p in EOS_PATTERNS),
            token_count=len(out_o.split()),
            rep3=repetition_3gram_ratio(out_o),
        )
        all_results.append(res_o)

        sim = SequenceMatcher(a=out_g, b=out_o).ratio()
        pair_scores.append({
            "prompt_id": pid,
            "char_similarity": sim,
            "ggml_tokens": res_g.token_count,
            "opencl_tokens": res_o.token_count,
            "ggml_eos": res_g.eos_hit,
            "opencl_eos": res_o.eos_hit,
            "ggml_rep3": res_g.rep3,
            "opencl_rep3": res_o.rep3,
        })
        print(
            f"{pid}: sim={sim:.4f} "
            f"eos(g/o)={int(res_g.eos_hit)}/{int(res_o.eos_hit)} "
            f"rep3(g/o)={res_g.rep3:.3f}/{res_o.rep3:.3f} "
            f"tok(g/o)={res_g.token_count}/{res_o.token_count}"
        )

    s_ggml = summarize(all_results, "ggml")
    s_opencl = summarize(all_results, "opencl")
    report = {
        "summary": {
            "ggml": s_ggml,
            "opencl": s_opencl,
            "avg_char_similarity": statistics.mean([x["char_similarity"] for x in pair_scores]) if pair_scores else 0.0,
        },
        "pairs": pair_scores,
    }

    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Summary ===")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report saved: {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

