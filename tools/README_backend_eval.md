# Backend Text Quality Eval

## 1) Prepare deterministic sampler in model `config.json`
Set the model folder `config.json` sampler section to deterministic settings:

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

`top_k=1` is the key for stable backend comparison.

## 2) Run comparison

```powershell
python forcodex/tools/backend_eval.py `
  --serial 3B15CR0014H00000 `
  --remote-bin /data/local/tmp/ziqian/models/qwen2-0.5b-work/bin/powerserve-run `
  --work-folder /data/local/tmp/ziqian/models/qwen2-0.5b-work `
  --prompts forcodex/tools/prompts_eval.txt `
  --n-predicts 128 `
  --base-env LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 `
  --opencl-env POWERSERVE_Q8_FAST_ALIGN_X=1 `
  --out-json backend_eval_report.json `
  --out-text backend_eval_outputs.txt
```

The script runs each prompt twice:
- ggml: `POWERSERVE_USE_OPENCL=0`
- opencl: `POWERSERVE_USE_OPENCL=1`

It also writes extracted text outputs incrementally (after each prompt) to `--out-text`:
- format:
  - `pXXX:`
  - `ggml:`
  - `opencl:`

## 3) Acceptance suggestions

- `opencl eos_rate` should not be significantly lower than `ggml eos_rate`.
- `opencl avg_rep3` should not be significantly higher than `ggml avg_rep3`.
- `avg_char_similarity` is informative only; do not use it as sole gate.

Prefer trend comparison over single-prompt judgment.
