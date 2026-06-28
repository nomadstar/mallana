# What we are building

This repository is not simply a TurboQuant fork.

It is a research platform dedicated to one fundamental question:

> **How much useful intelligence can we run on the hardware people already own?**

Modern AI has advanced at an extraordinary pace, but much of that progress has come with an equally dramatic increase in computational requirements. Every new generation of models tends to demand more memory, more bandwidth, larger GPUs, and increasingly specialized hardware.

This project challenges the assumption that scaling intelligence must always scale hardware requirements.

Instead of asking:

> *"How can we make models larger?"*

we ask:

> *"How can we make existing hardware capable of running larger and better models?"*

The objective is not merely to compress tensors, accelerate kernels, or optimize attention. Those are individual research problems.

The real objective is to reduce the **cost of intelligence itself**.

That means making better use of every byte of memory, every cache line, every tensor operation, every synchronization point, and every watt consumed by the machine.

Every optimization should contribute to one overarching goal:

> **Increase the amount of useful intelligence that can execute per unit of hardware.**

To achieve this, the project investigates every layer of the inference pipeline.

---

## Quantization

Memory is often the first limiting resource during inference.

Research in this area focuses on representing model weights and KV cache using fewer bits while preserving numerical accuracy and model quality.

Representative topics include:

* TurboQuant
* TurboVec
* AWQ
* GPTQ
* SmoothQuant
* SpinQuant
* QuaRot
* BitNet
* FP8
* MXFP4
* Learned quantization
* Mixed precision
* Future quantization methods

---

## Attention

Attention is one of the largest computational bottlenecks in transformer inference.

Research focuses on reducing computation, reducing memory movement, improving locality, and enabling longer contexts without sacrificing correctness.

Representative topics include:

* FlashAttention
* FlashAttention-2
* FlashAttention-3
* FlashInfer
* PagedAttention
* TriAttention
* TurboAttention
* SparQ Attention
* MInference
* Sparse attention
* Block-sparse attention
* Linear attention
* Future attention algorithms

---

## KV Cache Systems

The KV cache has become one of the dominant consumers of memory during inference.

This project explores techniques for reducing its footprint while maintaining generation quality and enabling substantially longer contexts.

Representative topics include:

* TurboQuant KV compression
* SnapKV
* PyramidKV
* KIVI
* H2O
* Cache compression
* Cache eviction
* Cache scheduling
* Shared KV caches
* Hierarchical caches
* Memory paging
* Future KV cache systems

---

## Decoding

Inference speed depends not only on matrix multiplication but also on the decoding strategy.

Research investigates methods that reduce the amount of computation required to produce each token.

Representative topics include:

* Speculative Decoding
* Self-Speculative Decoding
* Assisted Decoding
* Parallel Decoding
* Multi-token prediction
* Early Exit
* Draft models
* Future decoding methods

---

## Execution Engine

Even mathematically optimal algorithms can lose performance because of inefficient execution.

Research in this area focuses on minimizing overhead throughout the execution pipeline.

Representative topics include:

* Better graph execution
* Graph optimization
* Operator fusion
* Kernel fusion
* Better scheduling
* Continuous batching
* CUDA Graphs
* Memory-efficient tensor layouts
* Asynchronous execution
* Layer-wise inference and weight streaming (AirLLM)
* Future execution optimizations

### Layer-Wise Inference (Weight Streaming)

When model weights exceed available VRAM, layer-by-layer execution is a practical alternative
to quantization: each transformer layer is loaded from disk or CPU RAM, computed on GPU, then
swapped out. This trades I/O bandwidth for VRAM headroom, enabling 70B models to run on 4 GB
cards without quantization. The key research question is how to overlap weight prefetch with
compute to minimize the I/O penalty.

Reference implementation: [AirLLM](https://github.com/lyogavin/airllm).

---

## Portability

Good ideas should not belong to a single hardware vendor.

Every optimization should be designed, whenever practical, to be portable across multiple backends.

Representative targets include:

* CUDA
* ROCm
* Vulkan
* CPU
* Metal
* SYCL
* Emerging accelerator APIs

Portability is not an afterthought.

It is one of the project's engineering goals.

---

## Validation and Compatibility

Correctness comes before optimization.

The first measure of success is whether the project continues to build and pass the supported CI matrix.

That includes CPU, CUDA, ROCm, Vulkan, Metal, SYCL, WebGPU, OpenVINO, Windows, macOS, Linux, ARM, x64, and other supported targets where applicable.

An optimization is not considered successful if it improves one backend while unnecessarily breaking another supported backend.

This project values performance, but not at the cost of making the system fragile, platform-specific, or unusable for real users.

The practical platform target is the hardware and operating systems officially supported by Ollama and llama.cpp.

The system should remain responsive on those platforms, not only fast on specialized hardware.

Responsiveness includes:

* reasonable time to first token
* stable interactive latency
* sustained generation without pathological stalls
* predictable memory use
* graceful behavior on constrained machines
* correctness across supported backends

---

## Integration Philosophy

This project prefers proven techniques over speculative invention.

New work should begin from methods that are already demonstrated in papers, production systems, upstream implementations, or reproducible experiments.

Original engineering is still necessary, but it should usually focus on adapting, composing, validating, and making known-good techniques work together across real hardware.

The goal is not isolated benchmark wins.

The goal is cumulative improvement: quantization, attention, KV cache systems, decoding, scheduling, memory management, and backend execution should reinforce each other rather than compete.

Brand new solutions are considered only when they are grounded in evidence, or when they combine proven ideas in a way that makes the whole system better.

---

## AI-Assisted Engineering

This project also investigates a second problem:

> **Can AI systems collaborate to accelerate systems research while remaining under human supervision?**

Rather than replacing engineering judgment, AI is treated as an engineering tool.

To make this collaboration repeatable and rigorous, this repository implements a structured **Research Automation Framework (Research OS)**. The framework separates responsibilities across specialized AI roles (Architect, Implementer, Validator, Reviewer) and anchors progress in immutable milestone logs under the `/research` directory.

Different AI systems specialize in their respective roles:
- **Architect**: Designs experiments, models physical/mathematical hypotheses, and sets explicit validation and rollback criteria.
- **Implementer**: Applies minimal, scoped code patches without ad-hoc optimization or unilateral design changes.
- **Validator**: Runs builds, benchmarks, and perplexity gates to verify improvements against the baseline, producing objective binary verdicts.
- **Reviewer**: Audits designs and implementations as the "devil's advocate," attempting to actively refute the architect's hypothesis.

This structured workflow is detailed in [RESEARCH.md](RESEARCH.md) and governed by the prompts in `/prompts` and automation tools in `/scripts`. Through this framework, the human researcher remains responsible for scientific direction, vetoes, and final approval, while AI systems execute the iterative lifecycle.

### Code Intelligence (Research OS Tooling)

Effective AI-assisted research requires agents to navigate a large codebase efficiently without
re-reading files repeatedly. A persistent knowledge graph — indexing functions, classes, call
chains, and cross-file references — reduces token consumption and tool-call overhead for all
participating agents.

Reference implementation: [codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp)
(tree-sitter AST analysis, 14 MCP tools, single binary, 120× fewer tokens vs file-by-file search).

---

## Future Research

No optimization technique is excluded a priori.

If a new idea increases capability density while respecting the project's engineering principles, it belongs here.

Every proposed technique is evaluated using the same questions:

* Does it pass the supported CI matrix?
* Does it improve correctness?
* Does it reduce memory?
* Does it improve portability?
* Does it reduce computation?
* Does it improve capability per unit of hardware?
* Does it preserve responsiveness on officially supported platforms?
* Does it compose cleanly with existing optimizations?
* Is it based on a proven method or a deliberate combination of proven methods?
* Does it avoid improving one backend by carelessly breaking another?

If the answer is yes, it deserves investigation.

The names of today's algorithms may change.

The mission of this project does not.
