# Static batching: 12 requests on one T4

Run `20261122T234358Z-41c695f9`, from the saved outputs in [notebook 03](../notebooks/03-static-batching.ipynb). Values below are rounded medians displayed by the notebook; original per-request exports are not yet preserved in this repository.

## Configuration

- Model: `LiquidAI/LFM2.5-1.2B-Instruct`, revision `b9d816ab510f9a28dca37cb182e9da8b0226e13c`.
- Tesla T4, approximately 14.6 GiB device capacity; FP16, SDPA, Transformers 5.3.0.
- Twelve copies of one prompt, 66 input tokens each; greedy generation, maximum 64 new tokens, cache enabled, seed 42.
- Batch sizes 1/3/6; two warmup calls per size, then three measured rounds with rotating configuration order. All requests are ready at workload start; batches execute sequentially.
- Total time excludes loading, tokenization, input transfer, final text decoding and disk export. It includes generation and between-call Python/output-copy overhead.

## Observed results

| Batch size | Calls per 12 requests | Total seconds | Requests/second | Aggregate output tokens/second | Peak allocated GiB | All measured outputs match sequential reference |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 12 | 9.347 | 1.28 | 52.6 | 2.194 | Yes |
| 3 | 4 | 3.274 | 3.67 | 150.3 | 2.205 | Yes |
| 6 | 2 | 1.677 | 7.15 | 293.3 | 2.219 | Yes |

Batch sizes 3 and 6 reduced total time by approximately 65% and 82% (2.85× and 5.57× speedups). Batch size 6 added about 25.6 MiB of peak allocated memory. Shared model weights remain loaded once; batching allows more parallel work, with additional per-request state. Exact compute/memory bottlenecks were not profiled.

Throughput is aggregate across requests. It is not a single response's streaming speed. Request completion includes waiting for earlier batches; total time divided by 12 is amortized time, not individual latency.

## Limits and artifacts

One repeated prompt and three rounds do not establish real-traffic performance, tail latency or answer quality. Equal inputs avoid padding; matching outputs avoid output-length differences in this comparison. TTFT/TPOT and GPU compute utilization were not measured.

The notebook reports successful export of `results.json`, `summary.csv`, `workloads.csv`, `batches.csv`, and `requests.csv` to `/content/runs/static-batching/20261122T234358Z-41c695f9/`. Download these from Colab to retain exact measurements, full responses and complete hardware/software metadata. This report preserves the visible summary only.

Next: compare distinct prompts of similar lengths, then mixed-length prompts, holding the same 12 requests and their order fixed across batch sizes.
