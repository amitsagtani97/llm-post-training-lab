# CPU versus T4 GPU: LFM2.5 inference

Recorded November 22, 2026. This report compares batch-size-1, FP32 inference with `LiquidAI/LFM2.5-1.2B-Instruct` in Google Colab. Run IDs and filenames use UTC timestamps.

**Observed result:** the latest uninstrumented T4 run generated the same 56-token response in a median **1.352 seconds**, compared with **13.257 seconds** on CPU: **9.81× faster**, or **89.80% lower generation latency**. Separate T4 token diagnostics measured median **35.59 ms TTFT** and **24.23 ms TPOT**. CPU token timings have not been measured.

## Evidence and saved artifacts

| Run | Evidence retained | Artifacts |
| --- | --- | --- |
| CPU `20261122T093253Z-717dd005` | Summary transcribed from CPU notebook outputs reviewed earlier in this session. Original exported JSON/CSV and CPU token arrays are unavailable in this checkout. | [CPU summary](lfm2.5-cpu-fp32-20261122T093253Z-717dd005-summary.json) |
| T4 `20261122T102037Z-ac8eb56e` | Complete original baseline exports recovered from the notebook's printed file contents. | [JSON](lfm2.5-t4-fp32-20261122T102037Z-ac8eb56e.json), [CSV](lfm2.5-t4-fp32-20261122T102037Z-ac8eb56e.csv) |
| T4 `20261122T103323Z-4c863240` | Complete baseline and token-diagnostic exports recovered from the executed notebook's printed file contents. Main GPU result used below. | [JSON](lfm2.5-t4-fp32-20261122T103323Z-4c863240.json), [baseline CSV](lfm2.5-t4-fp32-20261122T103323Z-4c863240.csv), [token-timing CSV](lfm2.5-t4-fp32-20261122T103323Z-4c863240-token-timing.csv) |

The CPU summary explicitly distinguishes full-precision printed medians from rounded individual measurements. It is a provenance-labeled summary, not the original raw export. GPU JSON retains every response, token ID, generation setting, and diagnostic token-arrival timeline. Existing results are preserved as separate runs rather than overwritten or pooled.

## Workload and measurement configuration

| Setting | Configuration |
| --- | --- |
| Model | `LiquidAI/LFM2.5-1.2B-Instruct` |
| Model/tokenizer revision | `b9d816ab510f9a28dca37cb182e9da8b0226e13c` |
| Parameters | 1,170,340,608, printed during model loading |
| Batch size / concurrency | 1 request / 1 active request |
| Prompt | Explain the difference between a CPU and a GPU in three short sentences. |
| Input length | 24 tokens, including the chat template |
| Output limit | 64 new tokens |
| Observed output length | 56 token IDs per request; GPU arrays include final EOS ID `7` |
| Weight precision | `torch.float32` on both CPU and GPU |
| FP32 matrix precision policy | `torch.set_float32_matmul_precision("highest")` |
| Attention implementation requested | `sdpa`; exact dispatched kernel was not recorded |
| Decoding | Greedy: `do_sample=False`, `num_beams=1` |
| Request cache | `use_cache=True`; new request cache for every repetition |
| Seed | 42 |
| Model mode | `model.eval()` and `torch.inference_mode()` |
| Baseline sampling | One warmup, then three measured requests |
| Token-diagnostic sampling | Separate warmup, then three additional measured requests on GPU |
| Baseline version | `first-inference-v1` |
| Diagnostic version | `token-streamer-v1` |

Baseline latency starts immediately before `model.generate()` and ends after device synchronization. It includes prompt processing, autoregressive generation, and generation-loop overhead. It excludes model download/loading, tokenization, input transfer, conversion of output IDs to text, and printing. The output rate is actual generated token IDs divided by this generation time.

There is no HTTP service, network round trip, server queue, request batching, or concurrent load in these results. They describe in-process inference, rather than production service capacity.

## Recorded hardware configuration

| Property | CPU run | T4 GPU run |
| --- | --- | --- |
| Environment | Google Colab CPU runtime | Google Colab GPU runtime |
| Selected device | `cpu` | `cuda:0` |
| CPU identification recorded | `x86_64` architecture; exact processor model unknown | `x86_64` architecture; exact host processor model unknown |
| Logical CPU count reported by `os.cpu_count()` | 8 | 8 |
| PyTorch intra-operation CPU thread setting | 4 | 4 |
| Total system RAM reported | 50.990276 GiB | 50.990276 GiB |
| GPU model selected | None | Tesla T4 |
| GPU total memory reported by PyTorch | Not applicable | 14.563171 GiB |
| CUDA available to PyTorch | False | True |
| Apple MPS available | False | False |
| Forced CPU selection | False; automatic selection chose CPU | False; automatic selection chose CUDA |
| Model parameter device | `cpu` | `cuda:0` |
| Input tensor device | `cpu` | `cuda:0` |

The CPU thread setting is a host setting even in the GPU runtime; it is not the number of GPU execution threads. Logical CPU count does not establish the number of physical cores or dedicated CPU capacity. System RAM is total machine capacity, not measured process usage. GPU total memory is device capacity, not free memory or peak model allocation.

**Hardware details not captured:** exact CPU vendor/model, physical core count, CPU affinity/quota, inter-operation thread setting, CPU instruction extensions, NUMA layout, OpenMP/MKL environment variables, GPU driver version, GPU clocks/power limits, device utilization over time, and Colab plan/runtime tier. These values cannot be recovered from the saved metrics. The two runtimes have similar recorded host properties, but identical physical host hardware is not established.

## Software configuration

| Component | CPU run | T4 GPU run |
| --- | --- | --- |
| Python | 3.13.18 | 3.13.16 |
| PyTorch | `2.14.1+cpu` | `2.14.1+cu130` |
| CUDA version associated with PyTorch build | None | 13.0 |
| Transformers installed / loaded | 5.3.0 / 5.3.0 | 5.3.0 / 5.3.0 |
| huggingface-hub | 1.33.0 | 1.33.0 |
| tokenizers | 0.22.2 | 0.22.2 |
| safetensors | 0.8.0 | 0.8.0 |
| psutil | 5.9.5 | 5.9.5 |
| OS / architecture / libc | `Linux-6.6.159+-x86_64-with-glibc2.39` | `Linux-6.6.159+-x86_64-with-glibc2.39` |

`torch.version.cuda` describes the PyTorch build, not the installed NVIDIA driver or a separately installed CUDA toolkit. Optional kernel-package versions were not recorded. The full GPU generation configuration is retained in JSON; the CPU summary preserves the settings explicitly observed in the notebook.

An earlier attempt failed after a Transformers upgrade left old modules loaded. Restarting the session resolved it. Loaded and installed versions both reported 5.3.0 for the successful runs.

## CPU and GPU baseline results

| Metric | CPU | Latest T4 |
| --- | ---: | ---: |
| Measured requests, excluding warmup | 3 | 3 |
| Input tokens per request | 24 | 24 |
| Output tokens per request | 56 | 56 |
| Median generation latency | 13.257262 s | 1.351925 s |
| Median output rate | 4.224100 tokens/s | 41.422416 tokens/s |
| Warmup generation latency | 13.330 s, printed rounded | 1.376948 s |
| Download + load + device placement | 33.02 s, printed rounded | 3.031772 s |
| Peak PyTorch CUDA allocation | Not applicable; CPU peak RAM was not measured | 4.371658 GiB |

The latency ratio is `13.257262350000019 / 1.3519249969999692 = 9.806211`. With the same output length, the output-rate ratio gives the same speedup. This ratio uses uninstrumented baseline requests on both devices.

CPU requests below retain the rounding of the original printed output:

| Request | Generation seconds, rounded | Output tokens/s, rounded |
| --- | ---: | ---: |
| Measured 1 | 13.339 | 4.20 |
| Measured 2 | 13.257 | 4.22 |
| Measured 3 | 13.107 | 4.27 |

Latest GPU requests, displayed to six decimal places; full precision remains in JSON/CSV:

| Request | Input tokens | Output tokens | Generation seconds | Output tokens/s | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Measured 1 | 24 | 56 | 1.371443 | 40.832907 | 4.371658 |
| Measured 2 | 24 | 56 | 1.348940 | 41.514067 | 4.371658 |
| Measured 3 | 24 | 56 | 1.351925 | 41.422416 | 4.371658 |

The earlier preserved T4 rerun had median 1.385266 s and 40.425447 tokens/s. The current run is a new observation. Its values should not silently replace the earlier run or be mixed with its samples.

## Separate GPU token diagnostics

All values in this table are from the instrumented pass. The CPU baseline has no TTFT, TPOT, or ITL data; those cannot be calculated from total generation time alone.

| Median metric | CPU | T4 instrumented pass |
| --- | --- | ---: |
| TTFT | Not measured | 35.586 ms |
| TPOT | Not measured | 24.232 ms/token |
| Mean ITL per request | Not measured | 24.225 ms |
| Generation latency | Not measured in this mode | 1.368302 s |

| Request | TTFT ms | TPOT ms/token | Mean ITL ms | Min ITL ms | Max ITL ms | Generation seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Measured 1 | 35.586 | 24.128 | 24.121 | 23.799 | 26.114 | 1.362637 |
| Measured 2 | 35.527 | 24.232 | 24.225 | 23.839 | 25.423 | 1.368302 |
| Measured 3 | 35.946 | 24.244 | 24.236 | 23.818 | 24.906 | 1.369384 |

Definitions, with N output IDs and T seconds through `generate()` completion:

- **TTFT:** time from generation entry until the first generated token ID reaches the Python callback. It includes generation setup, prefill, first-token selection, and the CPU copy. It is neither API TTFT nor pure prefill-kernel time.
- **TPOT:** `(T - TTFT) / (N - 1)`. This definition includes cleanup after the final token. It is undefined for fewer than two output tokens.
- **ITL:** each difference between consecutive token arrival timestamps; N output tokens produce N-1 gaps.
- **Mean ITL:** the average of those gaps, excluding cleanup after the final arrival. Consequently, it can differ slightly from TPOT.
- Counts include EOS and other generated special IDs. The initial prompt callback is skipped. Time to a generated ID need not equal time to the first visible word.

For instrumented request 2, `35.527 ms + 55 × 24.232 ms ≈ 1,368.302 ms`. This shows how first-token wait and subsequent-token time contribute to completion. The calculation uses one request's values, rather than combining independently selected medians.

Transformers copies generated IDs to CPU before invoking the streamer. The copy and Python callbacks add overhead. Instrumented median latency was 1.21% longer than the uninstrumented median in this run. That is an observed difference, not an isolated estimate of instrumentation cost: the two passes ran sequentially with only three measured requests each.

## Findings and possible explanations

1. **GPU acceleration helped substantially at the same precision.** Both devices used FP32, the same revision, 24 input tokens, and 56 output tokens. The T4 result was 9.81× faster for this workload. A plausible explanation is faster tensor execution and data movement on GPU; this benchmark does not separate compute, memory, and software overhead contributions. Python patch versions and device-specific PyTorch builds also differed.
2. **The first token arrived well before the response finished.** GPU TTFT was around 36 ms while completion took around 1.37 s in the diagnostic pass. For this short prompt, the repeated generation steps account for most elapsed time after generation starts. Longer prompts could change that balance; no long-context experiment has been run.
3. **Output quality was unchanged for this example.** CPU and GPU returned the same decoded three-sentence response. This supports output consistency on one prompt, rather than general quality equivalence across devices. The response's description of CPU processing as sequential is oversimplified; CPUs can also execute work in parallel.
4. **Weights dominate the recorded allocation for this short request.** FP32 parameter storage is approximately 4.360 GiB; observed CUDA peak allocation was 4.372 GiB. This is PyTorch's peak live tensor allocation, not total device/process memory or GPU utilization. Longer contexts and batches need separate memory measurements.
5. **Setup timing changes across reruns.** CPU setup was 33.02 s while this GPU rerun's setup was about 3.03 s. Reusing downloaded files and an initialized environment can reduce setup work. These measurements combine network/cache/loading/transfer effects and cannot establish that GPU loading itself is eleven times faster. The warmup label likewise does not guarantee a fresh process or a cold GPU.
6. **The repeated answers were computed again.** Each request started a new attention/convolution cache. Identical responses are consistent with greedy decoding; there was no answer cache. The Hugging Face disk cache, the request's model cache, and warmed library state are separate mechanisms.
7. **The observed variation is small, but the evidence is limited.** Latest GPU baseline requests ranged from 1.349 to 1.371 s. Three repetitions and one prompt do not establish reliable p95/p99 latency, saturation, sustained throughput, or operating cost. Reported tokens/s is a single-request output rate, not concurrent server capacity.

The earlier GPU run reviewed in the session had a much slower first request (9.786 s) than its warmed requests (~1.35 s). Its original export is not retained here, so this remains a historical reviewed observation rather than a row in the raw-artifact comparison. First-use initialization is a possible cause, as illustrated in the [PyTorch benchmarking guide](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html); the exact cause was not profiled in this lab.

## Preserved response

The reviewed CPU output and every measured GPU response used in this report contained:

> A CPU (Central Processing Unit) handles general-purpose tasks and processes multiple instructions sequentially. A GPU (Graphics Processing Unit) specializes in parallel processing for graphics and complex calculations. While CPUs are versatile, GPUs excel at handling large datasets and parallel workloads efficiently.

GPU token arrays end in EOS ID `7`, explaining why generation stopped at 56 IDs before the 64-token cap. CPU token arrays are unavailable for a token-by-token cross-device equality check. All measured baseline and instrumented GPU responses and token arrays are retained in the latest JSON.

## Reproducing and extending the comparison

1. Open [the inference notebook](../notebooks/01-gpu-inference.ipynb), install its pinned dependencies, and restart after upgrading an already imported library.
2. Use the recorded model revision, prompt, FP32 precision, greedy settings, and seed. Confirm the selected device and dependency versions before loading.
3. Run CPU and GPU as separate recorded environments. Preserve the original baseline, and run the optional diagnostics as a separate measurement mode on both devices.
4. Export and preserve JSON, baseline CSV, and token-timing CSV before replacing a Colab runtime. Save the executed notebook's printed artifacts as well.
5. Capture missing hardware details in a future experiment. Test CPU thread counts 1/2/4/8, GPU FP16, prompt/output lengths, and batch sizes as separate controlled changes.
6. Extend the prompt set and sample count before drawing broad performance or quality conclusions. Serving, streaming API timing, concurrency, and cost remain later milestones.

The [main learning roadmap](../README.md) tracks those milestones. Earlier Qwen2.5 results remain in the [archived executed notebook](../notebooks/archive/01-qwen2.5-cpu.ipynb); they use another model/tokenizer and are excluded from this hardware comparison.

## Earlier observations and experiment history

The archived Qwen2.5 CPU run generated 47 tokens per request, with median generation latency 15.074 seconds and median output rate 3.118 tokens/second. Its displayed outputs remain in [the archived notebook](../notebooks/archive/01-qwen2.5-cpu.ipynb); its separate exports are unavailable in this checkout. These values describe a different model and tokenizer and are excluded from the LFM2.5 CPU/GPU comparison.

An earlier conversation also reported a Qwen CPU observation of approximately 1.6 tokens/second. Its code and result artifacts are unavailable, so it remains unverified and is not used as a benchmark baseline.

The entries below preserve the state of the work when each entry was written. References to pending work are historical; the [learning guide](../README.md) contains the current completion checklist.

| Date | Work completed | Evidence / conclusion | Next action |
| --- | --- | --- | --- |
| 2026-11-21 | Established the roadmap; removed two unused notebooks | Repository reset; no new model runs | Create and execute the first GPU inference notebook |
| 2026-11-21 | Created the CPU/GPU inference notebook; changed the first experiment to CPU before GPU | Automatic device selection, FP32 baseline, warmup and measured requests, JSON export; model execution pending | Run on CPU, preserve the result, then repeat on GPU |
| 2026-11-21 | Added explanatory code comments and a CSV response export | JSON retains every response and token IDs; CSV pairs each warmup/measured response with its metrics; each measured response is also printed | Run the CPU baseline and download both artifacts |
| 2026-11-22 | Reviewed the Qwen CPU run and switched the active lesson to LFM2.5-1.2B-Instruct | Qwen: 47 output tokens, median 15.074 s and 3.118 tokens/s; executed notebook archived. LFM2.5 has no measured results yet | Run a fresh LFM2.5 CPU baseline, then the same model/revision/settings on GPU |
| 2026-11-22 | Investigated the LFM2.5 import failure and added a loaded/installed version check | Traceback is consistent with mixed modules after upgrading Transformers; model inference remains pending | Restart the session/kernel and rerun from section 2 |
| 2026-11-22 | Added an inline preview of the complete benchmark JSON and CSV | Preview reads the exported files and prints their sizes, responses, and measurements without rerunning inference | Run the preview after export and save the executed notebook |
| 2026-11-22 | Reviewed successful LFM2.5 CPU/T4 runs, preserved the latest printed GPU artifacts, and added separate token diagnostics | Prior T4 rerun: 1.385 s and 40.425 tokens/s. Synthetic timing/export checks passed for prompt exclusion, TTFT/TPOT/ITL, short outputs, and diagnostics enabled/disabled; real token-timing execution pending | Rerun from section 2 and inspect baseline versus instrumented results |
| 2026-11-22 | Preserved the executed GPU token-diagnostic exports and wrote the CPU/GPU comparison report | Latest T4 baseline: 1.352 s, 41.422 tokens/s; diagnostic TTFT 35.586 ms, TPOT 24.232 ms/token. Hardware/software and evidence limitations recorded; CPU outputs retained as a labeled summary | Run CPU token diagnostics and preserve complete exports |
