---
layout: docs
title: Decision 3.0 (d3)
parent: Benchmarks
nav_order: 20
---

## ⚡ d3-lite on the NPU

Accuracy and latency of `d3:lite` (vllm-sr/d3-lite v3.1.3, package D3-Lite-NPU2) served by FastFlowLM 1.0.10.

> **Note:**
> - AMD Ryzen AI MAX+ PRO 395 (Strix Halo, XDNA 2 NPU), 128 GB, Ubuntu 24.04, XRT, FLM's default NPU power mode (Performance).
> - Latency is per request at the engine (no HTTP), median of 20 runs after warm-up; HTTP adds about a millisecond.

### Accuracy against the upstream BF16 model

| Set | Questions | Upstream BF16 | NPU (FLM) |
|---|---:|---:|---:|
| Decision Index public suite, stratified subset (gold accuracy) | 6,824 | 61.2% | 60.6% |
| Same questions, top-1 agreement with BF16 | 7,226 | -- | 90.5% |
| Hand-built contract set, top-1 agreement with BF16 | 311 | -- | 93.9% |

Paired gold accuracy, upstream BF16 minus NPU: +0.59 pt (95% CI [-0.00, +1.18] pt).
Prompts, token ids and answer objects are byte-identical to the upstream runtime.

### Latency

| Request | Median (ms) |
|---|---:|
| 1 question, 128 tokens | 112 |
| 1 question, 330 tokens | 159 |
| 1 question, 1,024 tokens | 518 |
| 1 question, 4,096 tokens | 3652 |
| 4 questions, 330 tokens each | 574 |
| 16 questions, 330 tokens each | 2198 |

First request after `flm serve` is ready: 166 ms (steady state 164 ms over HTTP).
