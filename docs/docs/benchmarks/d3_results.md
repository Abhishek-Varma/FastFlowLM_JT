---
layout: docs
title: Decision 3.0 (d3)
parent: Benchmarks
nav_order: 20
---

## ⚡ d3-lite on the NPU

Accuracy and latency of `d3:lite` (vllm-sr/d3-lite v3.1.3, package D3-Lite-NPU2) served by FastFlowLM 1.0.11.

> **Note:**
> - AMD Ryzen AI MAX+ PRO 395 (Strix Halo, XDNA 2 NPU), 128 GB, Ubuntu 24.04, XRT, FLM's default NPU power mode (Performance).
> - Latency is per request at the engine (no HTTP), median of 20 runs after warm-up; HTTP adds about a millisecond.

### Accuracy against the upstream BF16 model

| Set | Questions | Upstream BF16 | NPU (FLM) |
|---|---:|---:|---:|
| Decision Index public suite, stratified subset (gold accuracy) | 6,824 | 61.2% | 60.5% |
| Same questions, top-1 agreement with BF16 | 7,226 | -- | 90.5% |
| Hand-built contract set, top-1 agreement with BF16 | 311 | -- | 93.2% |

Paired gold accuracy, upstream BF16 minus NPU: +0.69 pt (95% CI [+0.10, +1.28] pt).
Prompts, token ids and answer objects are byte-identical to the upstream runtime.

### Latency

| Request | Median (ms) |
|---|---:|
| 1 question, 128 tokens | 109 |
| 1 question, 330 tokens | 151 |
| 1 question, 1,024 tokens | 430 |
| 1 question, 2,048 tokens | 765 |
| 1 question, 4,096 tokens | 1587 |
| 1 question, 8,192 tokens | 3690 |
| 1 question, 16,384 tokens | 8425 |
| 1 question, 32,768 tokens | 22696 |
| 4 questions, 330 tokens each | 536 |
| 16 questions, 330 tokens each | 2067 |

First request after `flm serve` is ready: 191 ms (steady state 168 ms over HTTP).

Rows of 2,048 tokens and more: median of 15 runs (32,768: 3). From 1,536 tokens attention runs on the NPU (below
that the CPU is faster); the first such request also loads the attention kernel (about 0.1 s) and from then on the
decision model holds two NPU hardware contexts instead of one.

### Images

Accuracy on the Decision Index public vision suite (the 10 benchmarks that can be rebuilt without gated data; 60
questions each, images capped at 1.6 MP as the suite's `decide` engine sends them):

| Benchmark | Upstream BF16 | NPU (FLM) | Same answer |
|---|---:|---:|---:|
| CV-Bench | 81.7% | 85.0% | 91.7% |
| BLINK | 51.7% | 58.3% | 78.3% |
| RealWorldQA | 68.3% | 71.7% | 88.3% |
| CharXiv | 65.0% | 70.0% | 90.0% |
| InfographicVQA | 78.3% | 80.0% | 90.0% |
| Mind2Web | 61.7% | 63.3% | 83.3% |
| KIE (CORD + FUNSD) | 91.7% | 93.3% | 98.3% |
| Moderation (Hateful Memes) | 55.0% | 53.3% | 95.0% |
| R-Bench-M | 31.7% | 26.7% | 71.7% |
| MMMU-Pro vision | 16.7% | 15.0% | 58.3% |
| **All 600** | **60.2%** | **61.7%** | **84.5%** |

Paired gold accuracy, NPU minus upstream BF16: +1.50 pt (95% CI [−0.92, +3.92] pt). Image grids and input token
counts equal upstream's on every question; the processor's pixels are within 2 grey levels of upstream's.

Latency with and without an image, at the same total input length (one question, over HTTP, median of 5; 8,192
tokens: 3 runs; 16,384 and 32,000: 1):

| Input tokens | Text only (ms) | Text + one image (ms) | Image |
|---:|---:|---:|---|
| 516 | 266 | 483 | 640 × 480 (300 tokens) |
| 1,028 | 448 | 658 | 640 × 480 (300 tokens) |
| 2,052 | 830 | 1,651 | 1.6 MP (1,600 tokens) |
| 4,100 | 1,803 | 2,641 | 1.6 MP |
| 8,196 | 3,798 | 4,641 | 1.6 MP |
| 16,386 | 8,733 | 9,560 | 1.6 MP |
| 31,994 | 21,466 | 22,477 | 1.6 MP |

An image costs a fixed amount on top of its tokens (decoding, resizing and the vision tower): about 0.2 s at
640 × 480 and 0.8 s at 1.6 MP. All questions of a request share one image encoding: a 1.6 MP image with 8 questions
takes 6.4 s, not 8 × 1.5 s. The first request with images loads the vision tower (about 40 ms) and two more NPU
hardware contexts.
