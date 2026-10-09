#!/usr/bin/env python
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-notebook test runner for the MONAI tutorials.

Mirrors the execution contract of tutorials/runner.sh:
  * Reduce long-running loop variables (max_epochs, val_interval, ...) to 1
    so a notebook exercises its full code path quickly.
  * Execute the (modified) notebook with papermill, from the notebook's own
    directory, with a wall-clock timeout.
  * Report a structured JSON result on stdout.

Usage:
  python run_nb.py <notebook.ipynb> [--timeout SECONDS] [--out OUTPUT.ipynb]

The original notebook on disk is NOT modified; a temporary reduced copy is
executed instead. Fixes to the notebook itself are made separately by editing
the original file.
"""
import argparse
import json
import os
import re
import signal
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REDUCE_VARS = [
    "max_epochs",
    "val_interval",
    "disc_train_interval",
    "disc_train_steps",
    "num_batches_for_histogram",
]


def reduce_epochs(nb_text: str) -> str:
    """Replace `<var> = <int>` with `<var> = 1` inside notebook JSON source."""
    out = nb_text
    for var in REDUCE_VARS:
        # match assignments like "max_epochs = 50" possibly inside JSON string lines
        out = re.sub(rf"({var}\s*=\s*)\d+", r"\g<1>1", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default=None)
    ap.add_argument("--kernel", default="python3")
    ap.add_argument("--no-reduce", action="store_true")
    args = ap.parse_args()

    nb_path = Path(args.notebook).resolve()
    if not nb_path.exists():
        print(json.dumps({"notebook": str(nb_path), "status": "error",
                          "reason": "file not found"}))
        return 2

    workdir = nb_path.parent
    src = nb_path.read_text()
    if not args.no_reduce:
        src = reduce_epochs(src)

    # write reduced copy next to original so relative paths resolve identically
    tmp_in = tempfile.NamedTemporaryFile(
        mode="w", suffix=".ipynb", dir=workdir, delete=False, prefix=".nbtest_in_")
    tmp_in.write(src)
    tmp_in.close()
    out_path = args.out or (str(tmp_in.name) + ".out.ipynb")

    cmd = [
        sys.executable, "-m", "papermill",
        tmp_in.name, out_path,
        "-k", args.kernel,
        "--log-output",
        "--no-progress-bar",
    ]

    env = dict(os.environ)
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    # Ensure that bare `python`/`pip` inside the notebook's %%bash and ! shell cells
    # resolve to the SAME interpreter running the kernel (the project venv). Otherwise
    # they fall back to whatever is first on PATH (here: a base conda MONAI 1.4.0), which
    # breaks `python -m monai.bundle` and other shell-invoked tools.
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    start = time.time()
    status = "passed"
    reason = ""
    tail = ""
    # Run in its own process group so a timeout can kill the whole tree
    # (papermill + the ipykernel it spawns), not just the immediate child.
    proc = subprocess.Popen(
        cmd, cwd=str(workdir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )
    try:
        combined, _ = proc.communicate(timeout=args.timeout)
        combined = combined or ""
        tail = combined[-6000:]
        if proc.returncode != 0:
            status = "failed"
            m = re.findall(r"(?:Exception|Error|Traceback|raise)\b.*", combined)
            reason = (m[-1] if m else f"exit code {proc.returncode}")[:500]
    except subprocess.TimeoutExpired:
        status = "timeout"
        reason = f"exceeded {args.timeout}s"
        # kill the entire process group, then reap
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            combined, _ = proc.communicate(timeout=30)
            tail = (combined or "")[-6000:]
        except Exception:
            pass
    finally:
        elapsed = round(time.time() - start, 1)
        try:
            os.unlink(tmp_in.name)
        except OSError:
            pass
        # keep the executed output notebook only on failure for debugging
        if status == "passed" and not args.out:
            try:
                os.unlink(out_path)
            except OSError:
                pass

    result = {
        "notebook": str(nb_path.relative_to(nb_path.parents[1]))
        if len(nb_path.parents) > 1 else nb_path.name,
        "status": status,
        "elapsed_s": elapsed,
        "reason": reason,
    }
    print("NBTEST_RESULT " + json.dumps(result))
    if status != "passed":
        print("----- log tail -----")
        print(tail)
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
