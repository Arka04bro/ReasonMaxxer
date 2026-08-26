#!/usr/bin/env python3
"""Every flag a caller passes must exist in the script it calls.

This is the cheapest possible check and it catches the most expensive class of
failure in this repo: a pilot step that dies on argparse after the GPU hours
before it have already been spent. `eval_checkpoints.py` shipped a
`--prompt_condition` that `generate_rollouts.py` never accepted, which meant
every checkpoint evaluation crashed the moment it was reached -- after
generation, scoring and training had all run.

Callers checked: the example recipes' shell scripts, and the one place a Python
script builds an argv for another one.

Run: python tests/test_script_interfaces.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FLAG = re.compile(r"(--[a-z0-9_]+)")
ADD_ARG = re.compile(r"add_argument\(\s*['\"](--[a-z0-9_]+)")


def accepted_flags(script: Path) -> set[str]:
    return set(ADD_ARG.findall(script.read_text(encoding="utf-8")))


def shell_invocations() -> list[tuple[str, Path, set[str]]]:
    """(caller, callee, flags) for every `$PYTHON scripts/x.py ...` in the recipes."""
    found = []
    for sh in sorted(ROOT.glob("examples/*/*.sh")):
        # Shell line continuations first, so one invocation is one line.
        text = sh.read_text(encoding="utf-8").replace("\\\n", " ")
        for line in text.splitlines():
            m = re.search(r"\$PYTHON\s+(scripts/[a-z_0-9]+\.py)(.*)", line)
            if m:
                found.append((sh.name, ROOT / m.group(1), set(FLAG.findall(m.group(2)))))
    return found


def python_invocations() -> list[tuple[str, Path, set[str]]]:
    """eval_checkpoints.py shells out to generate_rollouts.py."""
    src = (ROOT / "scripts/eval_checkpoints.py").read_text(encoding="utf-8")
    flags: set[str] = set()
    block = re.search(r"cmd = \[(.*?)\n        \]", src, re.S)
    if block:
        flags |= set(re.findall(r"['\"](--[a-z0-9_]+)['\"]", block.group(1)))
    for ext in re.findall(r"cmd\.extend\(\[(.*?)\]\)", src, re.S):
        flags |= set(re.findall(r"['\"](--[a-z0-9_]+)['\"]", ext))
    return [("eval_checkpoints.py", ROOT / "scripts/generate_rollouts.py", flags)]


def main() -> None:
    invocations = shell_invocations() + python_invocations()
    assert invocations, "no invocations found -- the parser stopped matching"

    failures = []
    for caller, callee, used in invocations:
        if not callee.exists():
            failures.append(f"{caller} calls {callee.name}, which does not exist")
            continue
        for flag in sorted(used - accepted_flags(callee)):
            failures.append(f"{caller} passes {flag} to {callee.name}, which does not accept it")

    if failures:
        for f in failures:
            print("FAIL:", f)
        sys.exit(1)

    print(f"{len(invocations)} invocations checked, every flag resolves")

    # The pilot runs on a T4, where vLLM's dtype="auto" would resolve to the
    # checkpoint's bfloat16 and fail. Each vLLM step must be told the dtype.
    pilot = ROOT / "examples/qwen25_0p5b_pilot"
    vllm_steps = ["02_generate_score.sh", "06_eval_passk.sh", "07_eval_base.sh"]
    for name in vllm_steps:
        text = (pilot / name).read_text(encoding="utf-8")
        if "--dtype" not in text:
            print(f"FAIL: {name} starts a vLLM engine without passing --dtype")
            sys.exit(1)
    print(f"{len(vllm_steps)} vLLM pilot steps pass an explicit dtype")

    from reasonmaxxer.eval_lib import build_llm  # noqa: F401  (import must not need vllm)

    print("\nPASS")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
