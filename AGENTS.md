## AI Development Policy

This repository is owned and maintained by its author.

AI-assisted development is explicitly permitted and is considered an official part of the development workflow.

Multiple AI systems may collaborate on architecture, implementation, debugging, testing, benchmarking, documentation, and code review under human supervision.

The repository owner remains the final decision maker. All code changes are reviewed, validated, and approved by the owner before being pushed, merged, or proposed upstream.

AI systems are encouraged to:

* inspect and understand the existing codebase before making changes;
* prefer minimal, evidence-based patches;
* validate every change with builds, tests, or benchmarks whenever possible;
* clearly separate facts, hypotheses, and assumptions;
* prioritize correctness over optimization.

AI systems must not:

* push changes automatically;
* rewrite large subsystems without explicit approval;
* assume previous conversations are correct without verifying the source code;
* bypass validation or introduce speculative changes.

This repository is a personal research and development project exploring memory-efficient LLM inference on consumer hardware, including technologies such as TurboQuant, PagedAttention, TriAttention, and related techniques.

This policy applies only to this repository and does not override the contribution policies of any upstream project. Any future upstream contribution must independently comply with that project's guidelines.
