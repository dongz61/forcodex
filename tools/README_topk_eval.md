# TopK Text Quality Eval (GGML Dense vs TopK)

## 1) Prepare deterministic sampler in model `config.json`
Set deterministic sampler settings for stable comparison:

```json
{
  "sampler": {
    "seed": 12345,
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 1.0,
    "min_keep": 1,
    "penalty_last_n": 0,
    "penalty_repeat": 1.0,
    "penalty_freq": 0.0,
    "penalty_present": 0.0,
    "ignore_eos": false
  }
}
```

`top_k=1` is the key for stable text alignment checks.

## 2) Run comparison

```powershell
python forcodex/tools/topk_eval.py `
  --serial 3B15CR0014H00000 `
  --remote-bin /data/local/tmp/ziqian/models/qwen2-0.5b-work/bin/powerserve-run `
  --work-folder /data/local/tmp/ziqian/models/qwen2-0.5b-work `
  --prompts forcodex/tools/prompts_eval.txt `
  --n-predicts 128 `
  --topk-k 64 `
  --base-env LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 `
  --out-json topk_eval_report.json `
  --out-text topk_eval_outputs.txt
```

The script runs each prompt twice on GGML backend:
- dense: `POWERSERVE_GGML_TOPK=0`
- topk: `POWERSERVE_GGML_TOPK=<--topk-k>`

It writes extracted outputs incrementally to `--out-text`:
- `pXXX:`
- `dense:`
- `topk:`

## 3) Notes

- This tool forces GGML mode by removing `POWERSERVE_USE_OPENCL` from both runs.
- You can pass extra per-mode envs:
  - `--dense-env KEY=VALUE`
  - `--topk-env KEY=VALUE`
- If `--topk-env POWERSERVE_GGML_TOPK=...` conflicts with `--topk-k`, script overrides to `--topk-k`.

## 4) Acceptance suggestions

- `topk eos_rate` should not be significantly lower than `dense eos_rate`.
- `topk avg_rep3` should not be significantly higher than `dense avg_rep3`.
- `avg_char_similarity` is informative only; do not use it as sole gate.
