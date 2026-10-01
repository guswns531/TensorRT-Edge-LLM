#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Record the identity of a retained result campaign.

Call ``start`` from a campaign driver before the first measurement and ``finish``
after the last one. ``start`` writes ``<result-dir>/manifest.json`` with the
source, binary, and engine identity in the schema used by the phase-serving
runners, copies the driver to ``run.sh``, and saves the uncommitted source diff.

    scripts/write_result_manifest.py start --result-dir .local/results/x-20261001 \\
        --work-item W014 --purpose "..." --driver "$0" \\
        --binary .local/current/active/runtime/llm_phase_context_smoke \\
        --engine gemma=.local/current/gemma4/engine --repeats 3 -- "$@"
    scripts/write_result_manifest.py finish --result-dir .local/results/x-20261001 \\
        --summary summary.json --exit-code $?
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

STATES = ("scratch", "diagnostic", "validation", "citable")


def digest(path):
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def git(repo, *args, binary=False):
    output = subprocess.check_output(["git", "-C", str(repo), *args])
    return output if binary else output.decode().strip()


def source_identity(repo):
    diff = git(repo, "diff", "HEAD", binary=True)
    untracked = git(repo, "ls-files", "--others", "--exclude-standard")
    return {
        "source_commit": git(repo, "rev-parse", "HEAD"),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_status": git(repo, "status", "--porcelain").splitlines(),
        "untracked_source_sha256": {
            name: digest(repo / name)
            for name in untracked.splitlines() if (repo / name).is_file()
        },
    }, diff


def engine_identity(directory):
    directory = pathlib.Path(directory).resolve()
    files = ("llm.engine", "visual.engine", "config.json")
    return {
        "path": str(directory),
        **{
            name.replace(".", "_") + "_sha256": digest(directory / name)
            for name in files if (directory / name).is_file()
        }
    }


def key_value(text):
    key, separator, value = text.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("expected NAME=PATH: " + text)
    return key, value


def start(args):
    result = args.result_dir.resolve()
    manifest_path = result / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        sys.exit("manifest exists; pass --overwrite to replace it: " +
                 str(manifest_path))
    inputs = [args.binary, args.plugin, args.driver
              ] + [pathlib.Path(path) for _, path in args.engine]
    missing = [str(path) for path in inputs if path and not path.exists()]
    if missing:
        sys.exit("identity inputs do not exist: " + ", ".join(missing))
    result.mkdir(parents=True, exist_ok=True)
    repo = pathlib.Path(git(pathlib.Path.cwd(), "rev-parse",
                            "--show-toplevel"))
    identity, diff = source_identity(repo)
    if diff:
        patch = result / f"source-{identity['tracked_diff_sha256'][:12]}.patch"
        patch.write_bytes(diff)
        identity["source_patch"] = patch.name
    if args.binary:
        binary = args.binary.resolve()
        identity["binary"] = str(binary)
        identity["binary_sha256"] = digest(binary)
    if args.plugin:
        identity["plugin_sha256"] = digest(args.plugin)
    identity["container"] = args.container or os.environ.get(
        "EDGELLM_CONTAINER_IMAGE")
    identity["engines"] = {
        name: engine_identity(path)
        for name, path in args.engine
    }
    driver = None
    if args.driver:
        driver = "run.sh"
        if args.driver.resolve() != (result / driver).resolve():
            shutil.copy2(args.driver, result / driver)
        identity["driver_sha256"] = digest(result / driver)
    manifest = {
        "state": args.state,
        "work_item": args.work_item,
        "purpose": args.purpose,
        "notes": args.note,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": args.command or None,
        "driver": driver,
        "identity": identity,
        "workload": dict(args.workload),
        "repeat_count": args.repeats,
        "summary_paths": [],
        "complete": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


def finish(args):
    manifest_path = args.result_dir.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    missing = [
        path for path in args.summary
        if not (manifest_path.parent / path).exists()
    ]
    if missing:
        sys.exit("summary paths do not exist: " + ", ".join(missing))
    manifest["summary_paths"] = sorted(
        set(manifest.get("summary_paths", [])) | set(args.summary))
    manifest["finished_at"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat()
    manifest["exit_code"] = args.exit_code
    manifest["complete"] = args.exit_code == 0
    if args.state:
        manifest["state"] = args.state
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="action", required=True)

    begin = commands.add_parser("start", help="write the manifest")
    begin.add_argument("--result-dir", type=pathlib.Path, required=True)
    begin.add_argument("--state", choices=STATES, default="diagnostic")
    begin.add_argument("--work-item", help="W### that owns this campaign")
    begin.add_argument("--purpose", required=True)
    begin.add_argument("--note",
                       action="append",
                       default=[],
                       help="citing note path; repeatable")
    begin.add_argument("--driver",
                       type=pathlib.Path,
                       help="driver script copied to run.sh")
    begin.add_argument("--binary", type=pathlib.Path)
    begin.add_argument("--plugin", type=pathlib.Path)
    begin.add_argument("--container",
                       help="image digest; default $EDGELLM_CONTAINER_IMAGE")
    begin.add_argument("--engine",
                       type=key_value,
                       action="append",
                       default=[],
                       metavar="NAME=DIR",
                       help="engine directory; repeatable")
    begin.add_argument("--workload",
                       type=key_value,
                       action="append",
                       default=[],
                       metavar="NAME=TRACE",
                       help="workload trace; repeatable")
    begin.add_argument("--repeats", type=int)
    begin.add_argument("--overwrite", action="store_true")
    begin.add_argument("command", nargs="*", help="campaign command, after --")
    begin.set_defaults(handler=start)

    end = commands.add_parser("finish", help="record completion")
    end.add_argument("--result-dir", type=pathlib.Path, required=True)
    end.add_argument("--summary",
                     action="append",
                     default=[],
                     help="summary path relative to the result dir")
    end.add_argument("--exit-code", type=int, default=0)
    end.add_argument("--state", choices=STATES)
    end.set_defaults(handler=finish)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
