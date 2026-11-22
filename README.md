# GPU ML Infrastructure Learning Guide

Learn GPU-based ML infrastructure through small experiments: run an open-weight model, measure its behavior, explain performance, serve it, and operate it reliably. Extend the same system with evaluated retrieval, model adaptation, and distributed execution as the foundations become clear.

This README contains the learning sequence and completion checklist. Check off a step when it has been executed and reviewed, with evidence in a notebook or report. Code that has only been written does not count as a completed experiment.

## Start here

- **Current lesson:** milestone 2 — trustworthy benchmarking and token-level latency.
- **Next practical step:** review the completed FP32/FP16 run one comparison at a time: generation time, first-token wait, then memory. Continue to batching once those measurements are clear.
- **Current notebook:** [02 — GPU workloads and precision](notebooks/02-gpu-benchmark.ipynb). Executed on a T4; 96 requests completed successfully. Interpretation is still being worked through. [Lesson 01](notebooks/01-gpu-inference.ipynb) remains available for reference.
- **Future scope:** GPU inference, memory, profiling, serving, and operations. Additional CPU benchmarks are skipped. Keep the existing CPU results as historical evidence, with their original limitations.
- **Benchmark results and hardware configurations:** [CPU versus T4 GPU report](results/cpu-vs-gpu.md) and [FP32 versus FP16 summary CSV](results/gpu-fp32-vs-fp16.csv). The CSV includes measurements, definitions, hardware/software, reasoning, and limitations; full request records remain in the executed notebook.

## Next lesson: repeatable GPU benchmarks and precision

**Notebook:** [02 — GPU workloads: FP32 versus FP16](notebooks/02-gpu-benchmark.ipynb).

**Status:** GPU execution and result recording are complete (run `20261122T214341Z-f97511de`). Understanding and interpretation remain in progress. See the [brief comparison CSV](results/gpu-fp32-vs-fp16.csv).

**Question:** how does FP16 change latency, memory, and response quality for the same model on the same GPU?

- [x] Require a CUDA GPU and print the actual device, driver, PyTorch/CUDA versions, and available memory before running.
- [x] Keep the existing model and pin its revision so this experiment changes precision rather than model identity.
- [x] Define a small fixed workload suite with stable prompt IDs and short, medium, and longer inputs; record actual token counts after chat formatting.
- [x] Run an FP32 baseline for each workload with explicit warmup and repeated measurements.
- [x] Run FP16 on the same GPU with the same prompts, generation settings, and output budgets; release the previous model before loading the next precision.
- [x] Record baseline generation latency/output rate and a separate instrumented TTFT/TPOT/ITL pass, plus allocated/reserved GPU memory.
- [x] Preserve actual output lengths and full responses; inspect output differences and any numerical failures.
- [x] Print and save per-request results, then write a separate comparison CSV.
- [ ] Explain the measured differences and timing boundaries comfortably before marking the lesson complete.

Finish the repeatable FP32 workload baseline before interpreting the precision comparison. Change one variable within each matched workload comparison. More input lengths provide workload coverage; they are not evidence of a precision effect by themselves.

Use a fresh GPU session so a previously loaded model cannot inflate memory measurements. Defaults run the three workloads in both precisions, with separate baseline and diagnostic passes. Every request is checkpointed to JSON outside the generation timer; final request/summary CSVs and the full JSON are printed, and one optional ZIP download preserves the run. Review full responses and stop reasons alongside performance before creating the results report.

## How to use this guide

1. Read the question for the current milestone and predict an experiment's outcome.
2. Work through a small notebook section and explain its inputs, operations, and outputs.
3. Execute it, inspect the response and measurements, and investigate failures.
4. Change one variable at a time and record the result in an experiment report.
5. Update the relevant checkboxes here and choose the next unfinished step.
6. Advance when the milestone's completion criterion is met. Use evidence rather than a fixed weekly deadline.

`[x]` means completed with reviewed evidence. `[ ]` means work remains. Tasks labeled **Skipped** are outside the current scope and do not count as completed experiments. Split partially completed tasks so completed work and remaining work are visible. Revisiting a topic with a new workload is a new experiment; it does not erase earlier results.

## Learning sequence

| Milestone | What to learn | Practical deliverable | Progress |
| --- | --- | --- | --- |
| 0. Establish the workspace | Runtimes, dependencies, files, and experiment records | Working notebook workflow and this guide | Complete |
| 1. Run inference | Tokenization, model loading, devices, and generation | Executed CPU/GPU inference notebook | Complete |
| 2. Measure correctly | Timing boundaries, warmup, token latency, and reproducibility | Comparable baseline and diagnostic reports | In progress |
| 3. Explain performance | GPU execution, precision, memory, caching, and batching | Controlled experiments with explanations | Not started |
| 4. Serve one model | Linux processes, containers, HTTP, streaming, and readiness | Reproducible inference server and client | Not started |
| 5. Handle demand and failures | Queueing, scheduling, capacity, backpressure, and recovery | Load tests and failure observations | Not started |
| 6. Operate a service | Deployment, storage, metrics, evaluation gates, and cost | Observable, reproducible service on a GPU host | Not started |
| 7. Build an evaluated application | Data ingestion, retrieval, reranking, citations, and freshness | Documentation RAG copilot | Later extension |
| 8. Adapt and manage models | Dataset lifecycle, fine-tuning, post-training, and release decisions | Evaluated adapted model and reproducible training run | Later extension |
| 9. Scale beyond one device | Parallelism, distributed execution, scheduling, and system design | Scaling experiment justified by measured limits | Advanced extension |

The initial focus is GPU LLM inference. New performance lessons must verify a CUDA GPU and stop with a clear message if one is unavailable. The first notebook retains its original CPU fallback for the completed introductory comparison. Host CPU activity still matters when it feeds GPU execution, tokenizes inputs, or handles requests; observing those interactions is part of GPU infrastructure, without adding standalone CPU optimization experiments.

The lessons in workload measurement, resource management, deployment, reliability, and reproducibility also apply to other ML systems. The advanced extensions are learning tracks selected when the preceding system and available compute justify them.

### 0. Establish the workspace

**Question:** where does code run, where do its files live, and how can an experiment be recovered?

- [x] Create a repository with a notebook-based learning workflow.
- [x] Establish a roadmap and a separate location for experiment results.
- [x] Run a notebook in Colab and inspect its environment.
- [x] Switch between CPU and GPU runtimes and verify the selected device.
- [x] Install dependencies, diagnose stale imports after an upgrade, and restart the kernel correctly.
- [x] Distinguish Colab runtime storage from local computer storage.
- [x] Export results and recover printed artifacts from an executed notebook.

**Completion criterion:** run a lesson from its setup cells, locate its artifacts, and preserve results before replacing a temporary runtime.

### 1. Run and understand inference

**Question:** what happens between a text prompt and a generated response?

- [x] Detect available accelerators and fall back to CPU when none is available.
- [x] Inspect the selected device, precision, model parameter count, and recorded environment.
- [x] Load an open-weight model and its matching tokenizer.
- [x] Inspect the chat template, special tokens, token IDs, input shape, and attention mask.
- [x] Explain evaluation mode and gradient-free inference.
- [x] Generate a response with greedy decoding and a bounded output length.
- [x] Separate prompt tokens from generated tokens and count the actual output.
- [x] Run a CPU baseline and repeat the same model/revision/workload on GPU in the same precision.
- [x] Compare the responses and discuss why output length and quality matter alongside speed.
- [x] Preserve the observed comparison and its evidence limitations in the results report.

**Completion criterion:** explain the path from messages to tokens to device tensors to generated IDs to text, and verify that inputs and model weights use the intended device.

### 2. Establish trustworthy measurements

**Question:** what exactly does each reported number measure, and can the procedure be reproduced?

- [x] Time model lookup/download/loading separately from generation.
- [x] State the generation timer's boundaries and exclude tokenization, input transfer, text decoding, and printing.
- [x] Synchronize GPU work around the wall-clock generation measurement.
- [x] Separate warmup requests from measured requests.
- [x] Record actual input/output lengths, generation latency, and output rate.
- [x] Record the model/tokenizer revision, generation settings, precision, seed, and basic hardware/software environment.
- [x] Preserve individual GPU measurements and responses before relying on aggregate statistics.
- [x] Measure GPU peak allocated memory and distinguish it from parameter storage and total device capacity.
- [x] Repeat the GPU baseline and document observed variation without pooling different runs silently.
- [x] Run a separate GPU token-diagnostic pass and retain TTFT, TPOT, ITL, and per-token arrival times.
- [x] Document callback overhead and distinguish instrumented measurements from the original baseline.
- [x] Write a comparison report that separates observations, hypotheses, and unmeasured quantities.
- [ ] Expand to a fixed prompt suite with stable IDs and representative input/output lengths.
- [ ] Increase repetitions and report sample counts and uncertainty before making tail-latency claims.
- [ ] Capture missing GPU environment details, including driver version and relevant kernel packages, plus host metadata needed to reproduce the GPU workload.
- [ ] Measure GPU reserved memory and device-wide usage with clearly stated definitions.

- **Skipped:** rerunning CPU inference to recover complete exports. The existing CPU record remains a reviewed summary.
- **Skipped:** CPU token diagnostics and a CPU/GPU token-level comparison. CPU TTFT/TPOT/ITL remain unmeasured.

**Completion criterion:** reproduce GPU measurements across representative workloads, preserve comparable per-request evidence, and explain which components each metric includes. TTFT requires token observations; total generation time alone cannot recover it. Skipped CPU work does not block this milestone.

### 3. Explain and optimize inference performance

**Question:** which workload or runtime change improves performance, and why?

Run separate experiments with a prediction, one changed variable, and a saved explanation.

- [ ] Inspect GPU utilization and memory behavior; observe host overhead when it affects GPU execution.
- [ ] Increase input length while keeping the output budget fixed; inspect actual output lengths.
- [ ] Increase the output budget while keeping the prompt fixed; inspect actual output lengths.
- [ ] Profile prefill and decode separately; distinguish host overhead, device execution, and memory movement.
- [ ] Compare request caching enabled/disabled and inspect output correctness as well as performance.
- [ ] Increase static batch size, handle padding correctly, and compare latency with aggregate throughput.
- [ ] Compare GPU FP32 and FP16 where supported; record speed, memory, and response differences.
- [ ] Test quantization, including AWQ when supported, against an unquantized quality/performance baseline.
- [ ] Explore compilation after measuring the eager baseline; separate compilation and steady-state costs.
- [ ] Investigate a bounded out-of-memory condition and recover the runtime.

- **Skipped:** standalone CPU thread sweeps, CPU-only utilization studies, and CPU-specific precision/quantization benchmarks.

**Completion criterion:** identify a useful GPU operating point for a defined workload and support the choice with measurements, including any quality or memory tradeoff. A newer GPU or lower precision is a hypothesis to test.

### 4. Serve one model over HTTP

**Question:** how does model execution become a service that another process can call?

- [ ] Practice the Linux/terminal essentials needed for deployment: files, permissions, environment variables, processes, logs, and signals.
- [ ] Create an isolated Python environment and reproduce the notebook's dependencies outside Colab.
- [ ] Connect to the serving host, inspect its GPU/driver/runtime compatibility, and manage the model cache.
- [ ] Start vLLM with an explicit model revision and configuration.
- [ ] Send a non-streaming request, then a streaming request, from a separate client process.
- [ ] Explain ports, binding addresses, client/server connections, request serialization, and response parsing.
- [ ] Trace the request through tokenization, scheduling, execution, and response delivery.
- [ ] Measure client latency and streaming metrics with explicitly defined timing boundaries.
- [ ] Distinguish API timing from in-process notebook timing.
- [ ] Exercise readiness checks, startup, shutdown, and server restart.
- [ ] Package the working service in a container and reproduce its GPU execution.

**Completion criterion:** recreate one working server/client exchange and explain where latency and failures can enter along the request path.

### 5. Handle demand and failures

**Question:** what happens when requests arrive faster than the service can complete them?

- [ ] Define a workload and initial service objectives for latency, throughput, and errors.
- [ ] Increase concurrency in controlled steps and record all completed and failed requests.
- [ ] Find saturation and explain queueing delay and the latency/throughput tradeoff.
- [ ] Compare continuous batching with the earlier static-batching experiment.
- [ ] Test mixed request lengths and a controlled arrival rate.
- [ ] Observe cache occupancy and scheduling behavior under load.
- [ ] Test invalid input, timeouts, client cancellation, overload, and restart during traffic.
- [ ] Add bounded retries, request limits, and backpressure; inspect their effects.
- [ ] Measure prefix caching with cold/warm cache states and an explicit hit rate.
- [ ] Compare vLLM with SGLang only after establishing comparable settings and a repeatable workload.

**Completion criterion:** describe the service's measured capacity and failure behavior and justify its operating limits. Include failures and sample counts alongside latency summaries.

### 6. Reproduce and operate the service

**Question:** can the service be recreated, observed, changed, and recovered reliably?

- [ ] Deploy the containerized service on a chosen GPU host and record the host configuration and launch procedure.
- [ ] Version dependency specifications, configuration, container images, and model revisions.
- [ ] Persist model caches and required artifacts outside disposable container/runtime storage.
- [ ] Manage credentials and configure the intended access boundary for the deployed endpoint.
- [ ] Automate startup, readiness checks, shutdown, and restart behavior.
- [ ] Collect request rate, latency, errors, queue indicators, cache usage, and GPU memory/utilization over time.
- [ ] Add structured logs and request identifiers; trace a request across service components.
- [ ] Build a small versioned quality set and run it before accepting a model or configuration change.
- [ ] Automate relevant build, smoke, quality, and performance checks in the release workflow.
- [ ] Compare a candidate release with the current version; exercise a canary and rollback.
- [ ] Calculate workload cost from recorded resource prices and measured useful throughput.
- [ ] Exercise a failure, investigate it using the recorded signals, and write recovery notes.
- [ ] Document how to stop billable resources and preserve results.

**Completion criterion:** recreate a known working service, detect an introduced problem, and recover it using documented procedures and evidence.

### 7. Extend to an evaluated RAG application

**Question:** how do data preparation and retrieval affect model quality, latency, and resource use?

Begin with a small documentation corpus after the serving foundations are established.

- [ ] Define representative questions, answer expectations, citation requirements, and abstention behavior.
- [ ] Build a versioned ingestion pipeline and record source provenance and document identifiers.
- [ ] Compare chunking strategies and preserve the configuration used to build each index.
- [ ] Serve embeddings, build retrieval, and evaluate retrieval separately from generation.
- [ ] Generate answers with citations and evaluate factual support, relevance, and abstention.
- [ ] Add a reranker after measuring the retrieval baseline.
- [ ] Measure ingestion/indexing cost and retrieval/reranking/generation latency separately and end to end.
- [ ] Refresh or rebuild the index and test data freshness and rollback behavior.
- [ ] Compare resource sharing and separate services for embeddings, reranking, and the LLM.

**Completion criterion:** demonstrate which components improve answer quality enough to justify their latency and resource cost. Citation formatting alone does not establish factual support.

### 8. Adapt models and manage training artifacts

**Question:** does model adaptation solve a measured quality problem, and can the training/release process be reproduced?

Select experiments based on an identified quality gap and available compute.

- [ ] Version a training dataset, create train/validation/test splits, and check for evaluation leakage.
- [ ] Establish the unmodified model's held-out quality and serving baseline.
- [ ] Run a small training experiment and inspect loss, validation behavior, throughput, and memory.
- [ ] Run SFT with LoRA/QLoRA and compare quality, training cost, and serving behavior.
- [ ] Save configuration, checkpoints, adapters, tokenizer, and evaluation evidence with stable identifiers.
- [ ] Resume an interrupted training run and reproduce inference from the saved artifacts.
- [ ] Define a model promotion decision and serve an accepted model version through the existing release workflow.
- [ ] Explore DPO when suitable preference data exists.
- [ ] Validate a verifiable reward, then explore RLVR/GRPO and test for reward exploitation.
- [ ] Explore reward-model training and PPO when the learning objective justifies the additional work.
- [ ] Compare selected methods on the same held-out evaluation set, including regressions and cost.

**Completion criterion:** demonstrate a defensible improvement on the intended task, reproduce the training artifacts, and release or reject the candidate based on evaluation evidence.

### 9. Scale beyond one device

**Question:** when is distribution necessary, and which new bottlenecks does it introduce?

Start after measuring single-device limits. Choose the experiments that answer an actual capacity or learning question.

- [ ] Compare larger batches, model replicas, and model parallelism as different scaling choices.
- [ ] Inspect host/device and device/device communication, topology, and model-loading behavior.
- [ ] Run multi-GPU inference with a justified parallelism strategy and measure communication overhead.
- [ ] Compare data-parallel and sharded training with DDP/FSDP when a training experiment requires them.
- [ ] Schedule multiple model workloads and measure resource contention and isolation.
- [ ] Explore an orchestrator and autoscaling after a repeatable deployment and load model exist.
- [ ] Exercise worker failure and recovery while preserving reproducibility and observability.
- [ ] Write a system design backed by the experiments: workload, quality goals, capacity, resource allocation, cost, failure handling, and scaling triggers.

**Completion criterion:** explain why a distributed configuration is appropriate for the measured workload and account for its communication, operating, and failure costs.

## Experiment and completion records

- Keep code minimal and readable, with one focused lesson per notebook and short explanatory comments.
- Keep run data, hardware/software configuration, findings, and experiment history in named reports under `results/`, linked from this guide.
- Preserve individual requests, responses, token counts, configuration, and errors. Label rounded summaries or missing raw evidence explicitly.
- Record each experiment's question, prediction, changed variable, artifact links, observations, explanation, limitations, and next action.
- Keep baseline and instrumented timing separate. Compare like workloads and report any differences in model, precision, software, or cache state.
- Update checkboxes after reviewing actual execution evidence. Keep a milestone in progress while its required tasks remain open.
- Preserve prior runs when a new model or configuration is introduced.
- Keep model weights, credentials, and bulky disposable outputs out of Git. Preserve selected reviewed artifacts before ending temporary runtimes.
- Review each diff before committing or pushing.

## Repository navigation

```text
README.md                         # Learning sequence, completion checklist, and next action
notebooks/01-gpu-inference.ipynb   # Completed introductory CPU/GPU lesson
notebooks/02-gpu-benchmark.ipynb   # Executed GPU workload/precision lesson; interpretation in progress
notebooks/archive/                # Earlier executed notebooks
results/cpu-vs-gpu.md              # Benchmark comparison, configurations, and findings
results/*.json                    # Preserved structured measurements or labeled summaries
results/*.csv                     # Preserved request tables and brief explanatory comparisons
runs/                             # Ignored runtime outputs; export before runtime replacement
src/                              # Shared helpers, introduced when needed
scripts/                          # Serving/load/deployment commands, introduced later
```

The lessons use PyTorch and Transformers for direct GPU inference, followed by a GPU serving engine. Colab provides experiment runtimes; a GPU host will provide the persistent serving environment. Model choices and exact run configurations are recorded in the benchmark report so this guide can continue as those choices evolve.
