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

def check_cli_tool(name):
    """Check if a CLI tool is available in the system PATH."""
    return shutil.which(name) is not None

def log_session(message):
    """Write log messages to history file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORY_LOG, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def run_planning(task, iteration, critique=None, model=None, skip_permissions=False, resume=False):
    """Phase 1: Architect (agy) designs or refines the plan."""
    if iteration == 1:
        prompt = (
            f"You are the ARCHITECT (Gemini / Antigravity CLI). The task is to: '{task}'.\n"
            f"Analyze the codebase and write a precise, step-by-step implementation plan.\n"
            f"Your output MUST be written in markdown and saved to the file '{PLAN_FILE}'.\n"
            f"Do not write the implementation code itself, only the files to modify, changes required, "
            f"and tests needed. Avoid conversational filler."
        )
    else:
        prompt = (
            f"You are the ARCHITECT (Gemini / Antigravity CLI). The implementation failed validation.\n"
            f"Here is the critique/error log:\n"
            f"```\n{critique}\n```\n"
            f"Please read the current plan in '{PLAN_FILE}', refine the plan to resolve these errors, "
            f"and update the file '{PLAN_FILE}' with the refined plan. Avoid conversational filler."
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
    print(f"Executing: {' '.join(cmd)}")
    log_session(f"Starting planning phase, iteration {iteration}")

    # Run interactively so user can see prompts and output
    res = subprocess.run(cmd)
    
    # Fallback check
    if res.returncode == 0:
        if not os.path.exists(PLAN_FILE) or os.path.getsize(PLAN_FILE) == 0:
            print(f"{YELLOW}Warning: '{PLAN_FILE}' was not created or is empty. Please verify the agent wrote it.{RESET}")
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
    print(f"Executing: {' '.join(cmd)}")
    log_session(f"Starting implementation phase, iteration {iteration}")

    res = subprocess.run(cmd)
    return res.returncode == 0

def run_validation():
    """System runs local compilation & unit tests validation."""
    print(f"\n{BOLD}{CYAN}=== Phase 3: Verification (Local Validation) ==={RESET}")
    log_session("Running validation suite")

    validate_script = "scripts/validate.sh"
    if not os.path.exists(validate_script):
        print(f"{YELLOW}Warning: '{validate_script}' not found. Falling back to build tools...{RESET}")
        if os.path.exists("build/Makefile"):
            cmd = ["make", "-C", "build"]
        elif os.path.exists("CMakeLists.txt"):
            cmd = ["cmake", "--build", "build"]
        else:
            return False, "No validation script or build system found."
    else:
        cmd = ["bash", validate_script]

    print(f"Executing validation: {' '.join(cmd)}")
    
    with open(VALIDATION_LOG, "w") as log_file:
        res = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    log_content = ""
    if os.path.exists(VALIDATION_LOG):
        with open(VALIDATION_LOG, "r") as f:
            log_content = f.read()
            print(log_content)

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
    print(f"Executing: {' '.join(cmd)}")
    log_session(f"Starting critique phase, iteration {iteration}")

    res = subprocess.run(cmd)
    return res.returncode == 0

def check_success():
    """Verify if the critique step declared success."""
    if not os.path.exists(CRITIQUE_FILE):
        return False
    with open(CRITIQUE_FILE, "r") as f:
        content = f.read().strip().upper()
        # Look for the exact word 'SUCCESS' in the response
        if "SUCCESS" in content and len(content) < 30:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Swarm Orchestrator (Multiswarm)")
    parser.add_argument("--task", required=True, help="The development task or bug to fix")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations/loops (default: 3)")
    parser.add_argument("--skip-permissions", action="store_true", help="Auto-approve tool permissions (skip prompts)")
    parser.add_argument("--no-interactive", action="store_true", help="Run without asking for confirmation between phases")
    parser.add_argument("--continue-session", action="store_true", help="Continue previous CLI sessions if possible")
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
    for f in [PLAN_FILE, SUMMARY_FILE, CRITIQUE_FILE, VALIDATION_LOG]:
        if os.path.exists(f):
            os.remove(f)

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
        if not run_planning(args.task, iteration, critique, args.model_agy, args.skip_permissions, args.continue_session):
            print(f"{RED}Planning failed in iteration {iteration}. Aborting.{RESET}")
            break

        if not args.no_interactive:
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
