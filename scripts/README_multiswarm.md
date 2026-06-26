# Multiswarm Orchestration Engine

This folder contains the **Multiswarm Orchestrator** (`scripts/multiswarm.py`), a Python utility designed to coordinate multiple cloud-based AI assistants (`agy`, `claude`, and `opencode`) working together on a single development task or bug fix.

By utilizing different AI models/tools for specific roles, this swarm matches agent capabilities to their strengths while optimizing token usage.

---

## Architecture & Agent Roles

```
                      +-------------------+
                      |   User Command    |
                      +---------+---------+
                                |
                                v
                   +------------+------------+
                   |  scripts/multiswarm.py  |
                   +------------+------------+
                                |
        +-----------------------+-----------------------+
        |                       |                       |
        v                       v                       v
+---------------+       +---------------+       +---------------+
|   Architect   |       |  Implementer  |       |Verifier/Critic|
|     (agy)     |       |   (claude)    |       |  (opencode)   |
+---------------+       +---------------+       +---------------+
- Context-heavy         - Code modifications    - Validation log review
- Writes / Refines      - Guided by the plan    - Git diff critique
  .multiswarm_plan.md   - Updates codebase      - Emits 'SUCCESS' or
                        - Writes summary          detailed feedback
```

### 1. **The Architect (Gemini / Antigravity via `agy`)**
- **Strengths**: Large context windows, rapid planning, token-efficient architectural scanning.
- **Responsibilities**: Creates and maintains the master implementation plan in `.multiswarm_plan.md`. If a validation test fails, the Architect refines the plan based on the criticism.

### 2. **The Implementer (Claude via `claude`)**
- **Strengths**: Precise code refactoring, structural editing, and high-fidelity code generation.
- **Responsibilities**: Reads `.multiswarm_plan.md` and applies targeted edits directly to the codebase. Writes a concise summary to `.multiswarm_summary.md` on completion.

### 3. **The Verifier & Critic (OpenCode via `opencode`)**
- **Strengths**: Critical review, compiler output analysis, and code quality compliance checking.
- **Responsibilities**: Automatically runs local compilation and unit tests (via `scripts/validate.sh`), inspects the `git diff`, and analyzes the execution logs.
- If everything is perfectly correct, it writes the word `SUCCESS` to `.multiswarm_critique.md`. Otherwise, it outputs a detailed list of compiler/test errors or quality critiques, looping back to the Architect/Implementer.

---

## Safety and Security Compliance

The orchestrator enforces the repository's **AI Development Policy** defined in `AGENTS.md` and `.gitignore`:
1. **Zero Secret Exposure**: The swarm is instructed never to hardcode credentials, API keys, or security tokens.
2. **Gitignore Protection**: All temporary files (`.multiswarm_*`, `.env`, keys, tokens) are ignored by git to prevent accidental commits.
3. **User in Control (Interactive Mode)**: By default, the orchestrator stops after each agent's turn to present changes and ask: `Proceed to next step? (y/n)`. This ensures you review and approve every action.

---

## Installation & Setup

1. Verify that all CLI utilities are available in your path:
   ```bash
   which agy claude opencode
   ```
2. Make sure the script is executable:
   ```bash
   chmod +x scripts/multiswarm.py
   ```

---

## Command Usage

Run the script by passing the target task in the `--task` argument:

```bash
./scripts/multiswarm.py --task "Implement unit tests for TriAttention caching mechanisms"
```

### Options

| Flag | Description |
|------|-------------|
| `--task "..."` | **Required**. The development goal or bug description. |
| `--iterations N` | Maximum number of plan-implement-verify loops (default: 3). |
| `--skip-permissions` | Bypasses permission confirmations inside `agy` and `claude` (auto-approves changes). |
| `--no-interactive` | Automatically progresses through swarm steps without prompting the user. |
| `--continue-session` | Reuses the previous conversation session when calling `agy`, `claude`, or `opencode`. |
| `--model-agy` | Model override for `agy` (e.g., `gemini-1.5-pro`). |
| `--model-claude` | Model override for `claude` (e.g., `claude-3-5-sonnet`). |
| `--model-opencode` | Model override for `opencode`. |
| `--cleanup` | Deletes temporary workspace logs/plans upon a successful completion. |

---

## Typical Run Workflow

1. **Planning**: `agy` creates `.multiswarm_plan.md` showing target modifications.
2. **Interactive Halt**: The user reviews the plan and presses `y` to continue.
3. **Execution**: `claude` modifies the source files and creates `.multiswarm_summary.md`.
4. **Validation**: The script compiles the codebase and runs numerical tests.
5. **Critique**: `opencode` compares the diff and compile status. If compilation fails, `opencode` outputs the exact warning/error.
6. **Iterate**: The loop restarts. `agy` and `claude` receive the compiler errors, rewrite the affected lines, and re-run validation until `opencode` outputs `SUCCESS` or the max iterations limit is reached.
