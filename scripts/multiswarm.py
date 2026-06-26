#!/usr/bin/env python3
"""
Multi-Agent Orchestrator (Multiswarm) for llama-cpp-turboquant.
Orchestrates agy (Gemini/Antigravity), claude (Claude Code), and opencode
to iteratively plan, implement, validate, and critique codebase improvements.

Roles:
- Architect (agy): High context capacity, designs the plan.
- Implementer (claude): High precision code edits, writes the code.
- Verifier/Critic (opencode): Reviews the diff and test logs, provides critique.
"""

import argparse
import os
import sys
import subprocess
import shutil
import threading
import time
from datetime import datetime

# ANSI Color codes
BOLD = "\033[1m"
GREEN = "\033[38;5;82m"
BLUE = "\033[38;5;39m"
YELLOW = "\033[38;5;214m"
RED = "\033[38;5;196m"
CYAN = "\033[38;5;51m"
RESET = "\033[0m"

# Temp files (ignored by .gitignore)
PLAN_FILE = ".multiswarm_plan.md"
SUMMARY_FILE = ".multiswarm_summary.md"
CRITIQUE_FILE = ".multiswarm_critique.md"
VALIDATION_LOG = ".multiswarm_validation.log"
HISTORY_LOG = ".multiswarm_history.log"

HEARTBEAT_INTERVAL = 30  # seconds of silence before printing a status line

def check_cli_tool(name):
    """Check if a CLI tool is available in the system PATH."""
    return shutil.which(name) is not None

def format_cmd_display(cmd):
    """Return a human-readable command string, replacing long --prompt/--print values."""
    result = []
    skip_next = False
    for part in cmd:
        if skip_next:
            result.append("<prompt>")
            skip_next = False
        elif part in ("--prompt", "--print"):
            result.append(part)
            skip_next = True
        else:
            result.append(part)
    return " ".join(result)

def log_session(message):
    """Write log messages to history file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORY_LOG, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def run_with_output(cmd, prefix="", print_func=print, log_file=None):
    """Run a command streaming stdout/stderr in real time with a heartbeat.

    Prints a status line every HEARTBEAT_INTERVAL seconds when the subprocess
    produces no output, so the operator can tell it is still alive.
    Optionally tee's output to log_file (an open file object).
    Returns the Popen object (with .returncode set).
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        text=True,
    )
    prefix_str = f"{BOLD}[{prefix}]{RESET} " if prefix else ""
    start = time.monotonic()
    last_output = [start]
    stop_event = threading.Event()

    def heartbeat():
        while not stop_event.is_set():
            stop_event.wait(timeout=5)
            if stop_event.is_set():
                break
            silence = time.monotonic() - last_output[0]
            if silence >= HEARTBEAT_INTERVAL:
                elapsed = time.monotonic() - start
                print_func(
                    f"{prefix_str}{YELLOW}⟳ still running… "
                    f"{elapsed:.0f}s elapsed, last output {silence:.0f}s ago{RESET}",
                    flush=True,
                )

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()

    for line in iter(process.stdout.readline, ""):
        if line:
            last_output[0] = time.monotonic()
            print_func(f"{prefix_str}{line}", end="", flush=True)
            if log_file:
                log_file.write(line)
                log_file.flush()

    stop_event.set()
    hb.join()
    process.wait()

    elapsed = time.monotonic() - start
    rc_color = GREEN if process.returncode == 0 else RED
    rc_sym = "✔" if process.returncode == 0 else "✘"
    print_func(
        f"{prefix_str}{rc_color}{rc_sym} finished in {elapsed:.1f}s "
        f"(exit {process.returncode}){RESET}",
        flush=True,
    )
    return process

def run_planning(task, iteration, critique=None, model=None, skip_permissions=False, resume=False):
    """Phase 1: Architect (agy) designs or refines the plan."""
    if iteration == 1:
        prompt = (
            f"You are the ARCHITECT (Gemini / Antigravity CLI). The task is to: '{task}'.\n"
            f"The codebase is located in the current working directory. You can find key Flash Attention files "
            f"such as 'fattn-vec.cuh' and 'fattn-tile.cuh' in 'ggml/src/ggml-cuda/'. Do NOT run slow global searches "
            f"starting from '/' or '$HOME'. Analyze the codebase and write a precise, step-by-step implementation plan.\n"
            f"Your plan MUST begin with YAML frontmatter (delimited by ---) containing:\n"
            f"- task: {task}\n"
            f"- created_at: <current datetime>\n"
            f"- scope: <e.g. feature, bugfix, tests-only, refactor>\n"
            f"- forbidden_paths: <list of file/directory paths the plan must not modify>\n"
            f"\n"
            f"Your output MUST be written in markdown and saved to the file '{PLAN_FILE}'.\n"
            f"CRITICAL: Do NOT run background/asynchronous commands or end your turn to wait for tasks. "
            f"Run all commands synchronously or read files directly. You MUST write the '{PLAN_FILE}' file in the same turn "
            f"before completing your execution. Do not write the implementation code itself, only the files to modify, "
            f"changes required, and tests needed. Avoid conversational filler."
        )
    else:
        prompt = (
            f"You are the ARCHITECT (Gemini / Antigravity CLI). The implementation failed validation.\n"
            f"Here is the critique/error log:\n"
            f"```\n{critique}\n```\n"
            f"Please read the current plan in '{PLAN_FILE}', refine the plan to resolve these errors, "
            f"and update the file '{PLAN_FILE}' with the refined plan. Avoid conversational filler.\n"
            f"IMPORTANT: Preserve the YAML frontmatter at the top of the plan. Update 'created_at' "
            f"to the current datetime if desired.\n"
            f"CRITICAL: Do NOT run background/asynchronous commands or end your turn to wait for tasks. "
            f"Run all commands synchronously or read files directly. You MUST write the updated '{PLAN_FILE}' file in the same turn "
            f"before completing your execution."
        )

    cmd = ["agy"]
    if skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    if model:
        cmd.extend(["--model", model])
    if resume and iteration > 1:
        cmd.append("--continue")
    
    cmd.extend(["--prompt", prompt])

    print(f"\n{BOLD}{CYAN}=== Phase 1: Planning with Architect (agy) ==={RESET}")
    print(f"Executing: {format_cmd_display(cmd)}")
    log_session(f"Starting planning phase, iteration {iteration}")

    # Run with real-time streaming output
    res = run_with_output(cmd, prefix="AGY")
    
    if res.returncode == 0:
        if not os.path.exists(PLAN_FILE) or os.path.getsize(PLAN_FILE) == 0:
            print(f"{YELLOW}Warning: '{PLAN_FILE}' was not created or is empty. Please verify the agent wrote it.{RESET}")
            return False
        return True
    else:
        print(f"{RED}Architect planning failed with return code {res.returncode}.{RESET}")
        return False

def run_implementation(task, iteration, model=None, skip_permissions=False, resume=False):
    """Phase 2: Implementer (claude) writes code changes based on the plan."""
    if not os.path.exists(PLAN_FILE):
        print(f"{RED}Error: Plan file '{PLAN_FILE}' not found!{RESET}")
        return False

    with open(PLAN_FILE, "r") as f:
        plan_content = f.read()

    critique_content = ""
    if os.path.exists(CRITIQUE_FILE):
        with open(CRITIQUE_FILE, "r") as f:
            critique_content = f.read()

    prompt = (
        f"You are the IMPLEMENTER (Claude Code). Your task is to execute the implementation plan.\n"
        f"Here is the plan:\n"
        f"```markdown\n{plan_content}\n```\n"
    )
    if critique_content:
        prompt += (
            f"IMPORTANT: The previous attempt failed validation. Here is the feedback/errors:\n"
            f"```\n{critique_content}\n```\n"
            f"Please address these errors and modify the implementation accordingly.\n"
        )
    prompt += (
        f"Please perform the required file edits/creations to implement the plan. "
        f"When you are finished, write a short summary of the files modified and changes made "
        f"to '{SUMMARY_FILE}' (e.g. 'Modified src/main.cpp to support X')."
    )

    cmd = ["claude"]
    if skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    if model:
        cmd.extend(["--model", model])
    if resume and iteration > 1:
        cmd.append("--continue")

    cmd.extend(["--print", prompt])

    print(f"\n{BOLD}{CYAN}=== Phase 2: Implementation with Implementer (claude) ==={RESET}")
    print(f"Executing: {format_cmd_display(cmd)}")
    log_session(f"Starting implementation phase, iteration {iteration}")

    res = run_with_output(cmd, prefix="CLAUDE")
    if res.returncode != 0:
        return False
    if not os.path.exists(SUMMARY_FILE) or os.path.getsize(SUMMARY_FILE) == 0:
        print(f"{YELLOW}Warning: '{SUMMARY_FILE}' was not created or is empty.{RESET}")
        return False
    return True

def run_validation():
    """System runs local compilation & unit tests validation."""
    print(f"\n{BOLD}{CYAN}=== Phase 3: Verification (Local Validation) ==={RESET}")
    log_session("Running validation suite")

    validate_script = "scripts/validate.sh"
    if not os.path.exists(validate_script):
        print(f"{YELLOW}Warning: '{validate_script}' not found. Falling back to build tools...{RESET}")
        if os.path.exists("CMakeLists.txt") or os.path.exists("build"):
            cmd = ["cmake", "--build", "build"]
        else:
            return False, "No validation script or build system found."
    else:
        cmd = ["bash", validate_script]

    print(f"Executing validation: {' '.join(cmd)}")

    with open(VALIDATION_LOG, "w") as log_file:
        res = run_with_output(cmd, prefix="BUILD", log_file=log_file)

    log_content = ""
    if os.path.exists(VALIDATION_LOG):
        with open(VALIDATION_LOG, "r") as f:
            log_content = f.read()

    passed = (res.returncode == 0)
    if passed:
        print(f"{GREEN}✔ Local validation tests passed.{RESET}")
    else:
        print(f"{RED}✘ Local validation tests failed.{RESET}")

    return passed, log_content

def run_critique(task, iteration, validation_passed, validation_log, model=None, agent=None, resume=False):
    """Phase 4: Reviewer/Critic (opencode) reviews implementation and log."""
    diff_output = ""
    try:
        diff_res = subprocess.run(["git", "diff"], capture_output=True, text=True)
        diff_output = diff_res.stdout
    except Exception as e:
        diff_output = f"Could not run git diff: {str(e)}"

    prompt = (
        f"You are the VERIFIER & CRITIC (OpenCode). A set of changes has been implemented for task: '{task}'.\n"
        f"Here is the local validation pass status: {validation_passed}\n"
        f"Here is the local validation log:\n"
        f"```\n{validation_log[:4000]}\n```\n"
    )
    if diff_output:
        prompt += (
            f"Here is the git diff of the current implementation:\n"
            f"```diff\n{diff_output[:8000]}\n```\n"
        )
    prompt += (
        f"Analyze the implementation, diff, and validation log against requirements.\n"
        f"If the compilation succeeds, unit tests pass, and the changes are completely correct, "
        f"write EXACTLY the word 'SUCCESS' in the file '{CRITIQUE_FILE}' and nothing else.\n"
        f"If there are compiler errors, unit test regressions, or code quality issues, "
        f"write a detailed markdown critique list in '{CRITIQUE_FILE}' listing what needs to be fixed. "
        f"Do NOT write 'SUCCESS' if there is any issue."
    )

    cmd = ["opencode", "run", prompt]
    if model:
        cmd.extend(["--model", model])
    if agent:
        cmd.extend(["--agent", agent])
    if resume and iteration > 1:
        cmd.append("--continue")

    print(f"\n{BOLD}{CYAN}=== Phase 4: Critique with Reviewer (opencode) ==={RESET}")
    print(f"Executing: {format_cmd_display(cmd)}")
    log_session(f"Starting critique phase, iteration {iteration}")

    res = run_with_output(cmd, prefix="OPENCODE")
    if res.returncode != 0:
        return False
    if not os.path.exists(CRITIQUE_FILE) or os.path.getsize(CRITIQUE_FILE) == 0:
        print(f"{YELLOW}Warning: '{CRITIQUE_FILE}' was not created or is empty.{RESET}")
        return False
    return True

def check_success():
    """Verify if the critique step declared success."""
    if not os.path.exists(CRITIQUE_FILE):
        return False
    with open(CRITIQUE_FILE, "r") as f:
        content = f.read().strip().upper()
    return "SUCCESS" in content

def parse_plan_task():
    """Parse the 'task' field from an existing plan file's YAML frontmatter."""
    if not os.path.exists(PLAN_FILE) or os.path.getsize(PLAN_FILE) == 0:
        return None
    with open(PLAN_FILE, "r") as f:
        content = f.read()
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            frontmatter = content[3:end].strip()
            for line in frontmatter.split("\n"):
                line = line.strip()
                if line.startswith("task:"):
                    return line[len("task:"):].strip()
    return None

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Swarm Orchestrator (Multiswarm)")
    parser.add_argument("--task", required=True, help="The development task or bug to fix")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations/loops (default: 3)")
    parser.add_argument("--skip-permissions", action="store_true", help="Auto-approve tool permissions (skip prompts)")
    parser.add_argument("--no-interactive", action="store_true", help="Run without asking for confirmation between phases")
    parser.add_argument("--continue-session", action="store_true", help="Continue previous CLI sessions if possible")
    parser.add_argument("--use-plan", action="store_true", help="Use existing .multiswarm_plan.md without running Architect planning")
    parser.add_argument("--force-use-plan", action="store_true", help="Use existing plan even if its task field does not match --task")
    parser.add_argument("--model-agy", help="Model override for agy")
    parser.add_argument("--model-claude", help="Model override for claude")
    parser.add_argument("--model-opencode", help="Model override for opencode")
    parser.add_argument("--agent-opencode", help="Custom agent for opencode")
    parser.add_argument("--cleanup", action="store_true", help="Delete temporary multiswarm files on completion")
    args = parser.parse_args()

    # Validate that CLI tools are installed
    missing_tools = []
    for tool in ["agy", "claude", "opencode"]:
        if not check_cli_tool(tool):
            missing_tools.append(tool)

    if missing_tools:
        print(f"{RED}Error: The following required CLI tools are missing in the PATH: {', '.join(missing_tools)}{RESET}")
        print("Please ensure they are installed and in your environment PATH.")
        sys.exit(1)

    print(f"{BOLD}{GREEN}================================================================{RESET}")
    print(f"{BOLD}{GREEN}                 MULTISWARM ORCHESTRATION ENGINE                {RESET}")
    print(f"{BOLD}{GREEN}================================================================{RESET}")
    print(f"Task: {args.task}")
    print(f"Max Iterations: {args.iterations}")
    print(f"Skip Permissions: {args.skip_permissions}")
    print(f"Interactive Confirmation: {not args.no_interactive}")
    print(f"{BOLD}{GREEN}================================================================{RESET}\n")

    # Clear old temporary files from previous runs
    for f in [SUMMARY_FILE, CRITIQUE_FILE, VALIDATION_LOG]:
        if os.path.exists(f):
            os.remove(f)

    use_existing_plan = False
    if args.use_plan or args.force_use_plan:
        if not os.path.exists(PLAN_FILE) or os.path.getsize(PLAN_FILE) == 0:
            print(f"{RED}Error: --use-plan specified but '{PLAN_FILE}' does not exist or is empty!{RESET}")
            sys.exit(1)
        stored_task = parse_plan_task()
        print(f"\n{BOLD}Current task:{RESET}       {args.task}")
        print(f"{BOLD}Stored plan task:{RESET}  {stored_task or '(no frontmatter)'}")
        if stored_task and stored_task != args.task:
            if args.force_use_plan:
                print(f"{YELLOW}  Reuse allowed: yes (--force-use-plan overrides mismatch){RESET}")
                use_existing_plan = True
            else:
                print(f"{RED}  Reuse allowed: no (task mismatch){RESET}")
                print(f"{RED}Error: --use-plan requires the stored plan task to match --task.{RESET}")
                print(f"       Use --force-use-plan to override.{RESET}")
                sys.exit(1)
        else:
            print(f"{GREEN}  Reuse allowed: yes{RESET}")
            use_existing_plan = True
    else:
        if os.path.exists(PLAN_FILE) and os.path.getsize(PLAN_FILE) > 0:
            stored_task = parse_plan_task()
            print(f"\n{BOLD}Existing plan found in '{PLAN_FILE}':{RESET}")
            print(f"  {BOLD}Current task:{RESET}       {args.task}")
            print(f"  {BOLD}Stored plan task:{RESET}  {stored_task or '(no frontmatter)'}")
            if stored_task and stored_task != args.task:
                print(f"  {RED}Reuse allowed: no (task mismatch){RESET}")
                print(f"  {YELLOW}Auto-removing incompatible plan.{RESET}")
                os.remove(PLAN_FILE)
            else:
                print(f"  {GREEN}Reuse allowed: yes{RESET}")
                if not args.no_interactive:
                    choice = input(f"\n{BOLD}{YELLOW}Reuse existing plan? (y/n): {RESET}").strip().lower()
                    if choice == 'y':
                        use_existing_plan = True
                    else:
                        os.remove(PLAN_FILE)
                else:
                    use_existing_plan = True

    log_session(f"New multiswarm task initiated: {args.task}")

    success = False
    for iteration in range(1, args.iterations + 1):
        print(f"\n{BOLD}{YELLOW}>>> STARTING ITERATION {iteration} of {args.iterations} <<<{RESET}")
        
        # Read latest critique if present
        critique = None
        if iteration > 1 and os.path.exists(CRITIQUE_FILE):
            with open(CRITIQUE_FILE, "r") as f:
                critique = f.read()

        # Step 1: Planning
        if use_existing_plan and iteration == 1:
            print(f"\n{BOLD}{CYAN}=== Phase 1: Planning (Reusing Existing Plan) ==={RESET}")
            print(f"Reusing existing plan from '{PLAN_FILE}'")
        else:
            if not run_planning(args.task, iteration, critique, args.model_agy, args.skip_permissions, args.continue_session):
                print(f"{RED}Planning failed in iteration {iteration}. Aborting.{RESET}")
                break

        if not args.no_interactive:
            if use_existing_plan and iteration == 1:
                # Ask to proceed using the existing plan
                choice = input(f"\n{BOLD}{YELLOW}Proceed to Implementation with existing plan? (y/n): {RESET}").strip().lower()
            else:
                choice = input(f"\n{BOLD}{YELLOW}Plan updated in '{PLAN_FILE}'. Proceed to Implementation? (y/n): {RESET}").strip().lower()
            if choice != 'y':
                print("Swarm aborted by user.")
                break

        # Step 2: Implementation
        if not run_implementation(args.task, iteration, args.model_claude, args.skip_permissions, args.continue_session):
            print(f"{RED}Implementation failed in iteration {iteration}. Aborting.{RESET}")
            break

        if not args.no_interactive:
            choice = input(f"\n{BOLD}{YELLOW}Implementation completed. Proceed to Verification? (y/n): {RESET}").strip().lower()
            if choice != 'y':
                print("Swarm aborted by user.")
                break

        # Step 3: Local validation
        val_passed, val_log = run_validation()

        # Step 4: Critique and review
        if not run_critique(args.task, iteration, val_passed, val_log, args.model_opencode, args.agent_opencode, args.continue_session):
            print(f"{RED}Critique phase failed in iteration {iteration}. Aborting.{RESET}")
            break

        # Step 5: Check completion status
        if check_success():
            print(f"\n{BOLD}{GREEN}✔ Success! Verifier (opencode) has approved the changes.{RESET}")
            success = True
            break
        else:
            print(f"\n{BOLD}{YELLOW}⚠ Iteration {iteration} complete. Verification did not yield success.{RESET}")
            if os.path.exists(CRITIQUE_FILE):
                print(f"{BOLD}Critique:{RESET}")
                with open(CRITIQUE_FILE, "r") as f:
                    print(f.read())
            
            if iteration < args.iterations:
                if not args.no_interactive:
                    choice = input(f"\n{BOLD}{YELLOW}Proceed to Iteration {iteration + 1}? (y/n): {RESET}").strip().lower()
                    if choice != 'y':
                        print("Swarm aborted by user.")
                        break
            else:
                print(f"{RED}Reached max iterations ({args.iterations}) without achieving verified SUCCESS.{RESET}")

    # Cleanup temp files if requested
    if args.cleanup:
        print("\nCleaning up temporary multiswarm files...")
        for f in [PLAN_FILE, SUMMARY_FILE, CRITIQUE_FILE, VALIDATION_LOG]:
            if os.path.exists(f):
                os.remove(f)
        print("Cleanup done.")

    if success:
        print(f"\n{BOLD}{GREEN}================================================================{RESET}")
        print(f"{BOLD}{GREEN}             MULTISWARM TASK COMPLETED SUCCESSFULLY             {RESET}")
        print(f"{BOLD}{GREEN}================================================================{RESET}")
        sys.exit(0)
    else:
        print(f"\n{BOLD}{RED}================================================================{RESET}")
        print(f"{BOLD}{RED}                MULTISWARM TASK FAILED OR ABORTED               {RESET}")
        print(f"{BOLD}{RED}================================================================{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
