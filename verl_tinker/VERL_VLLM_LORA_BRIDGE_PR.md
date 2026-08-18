# VERL/vLLM in-memory LoRA loader compatibility fix

## Problem

VERL replaces `LRUCacheWorkerLoRAManager._load_adapter` so rollout workers can
load LoRA weights received as tensors instead of writing an adapter checkpoint
to disk. The replacement copied vLLM's complete path-based loader and drifted
from vLLM 0.26.0.

For Qwen3, VERL passed the full `hf_to_vllm_mapper` into
`LoRAModel.from_lora_tensors`. Its stacked mappings convert `q_proj`, `k_proj`,
and `v_proj` to the same `qkv_proj` key. As the tensor dictionary is parsed,
the three adapters overwrite one another. The last V projection survives as a
single tensor, but `MergedQKVParallelLinearWithLoRA` expects a packed list of
three tensors. Activation then fails in `slice_lora_b` with:

```text
IndexError: too many indices for tensor of dimension 1
```

Diagnostics confirmed that all 504 matrices were two-dimensional both when
exported by VERL/FSDP and after IPC receipt. The malformed representation was
introduced only while vLLM parsed and packed the adapter.

## Proposed upstream change

Keep `TensorLoRARequest`, because vLLM's public worker request still accepts a
path rather than an in-memory tensor dictionary. Narrow the monkey patch:

1. Save vLLM's original `_load_adapter` and delegate all ordinary
   `LoRARequest` objects to it.
2. Implement only the `TensorLoRARequest` branch in VERL.
3. Convert the model mapper with `get_unstacked_mapper()` before calling
   `LoRAModel.from_lora_tensors`. This preserves constituent adapter names so
   vLLM's adapter manager can pack them once.
4. Forward current loader behavior relevant to tensor adapters, including
   `lora_skip_prefixes`, model vocabulary size, dtype, and the 3-D MoE layout
   marker.
5. Make patch installation idempotent.

This follows vLLM 0.26.0's native path loader, which already calls
`get_unstacked_mapper()` before parsing LoRA checkpoint tensors. Delegating
path-backed requests prevents future vLLM loader changes from being shadowed by
another stale copy in VERL.

## Suggested tests

- A Qwen3 mapper test proving q/k/v LoRA names remain distinct before packing.
- A delegation test proving an ordinary `LoRARequest` invokes vLLM's original
  loader.
- An idempotency test that calls `VLLMHijack.hijack()` twice.
- A TP=2 integration test that transfers an adapter and completes one rollout.

## Versions reproduced

- VERL: `0.9.0.dev0`, commit `535c47799b537a0e3602b9839344ee53d2a47128`
- vLLM: `0.26.0`
