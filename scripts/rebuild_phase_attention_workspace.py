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
"""Plan fresh same-capability Gemma/Cosmos engines; execute only with --execute."""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess

IMAGE = "nvcr.io/nvidia/tensorrt@sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3"
LINEAGES = {
    "gemma": (
        "gemma-4-e2b-it-awq/onnx-int4-awq-packed-p128/llm",
        "gemma-4-e2b-it-awq/engine-packed-p8-d24-kv2048-p192",
    ),
    "cosmos": (
        "text/cosmos-reason2-2b/onnx-fp16-p128/llm",
        "text/cosmos-reason2-2b/engine-p8-d64-kv256-p128-vp1024-atomic",
    ),
}
FLAGS = {
    "max_batch_size": "--maxBatchSize",
    "max_prefill_batch_size": "--maxPrefillBatchSize",
    "max_decode_batch_size": "--maxDecodeBatchSize",
    "max_input_len": "--maxInputLen",
    "max_kv_cache_capacity": "--maxKVCacheCapacity",
    "max_kv_pool_pages": "--maxKVPoolPages",
    "max_prefill_chunk_tokens": "--maxPrefillChunkTokens",
    "max_vision_prefill_batch_size": "--maxVisionPrefillBatchSize",
    "max_vision_prefill_chunk_tokens": "--maxVisionPrefillChunkTokens",
    "max_lora_rank": "--maxLoraRank",
}


def sha256(path):
    """Hash retained inputs without loading a multi-GiB file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as data:
        for block in iter(lambda: data.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_model(root, build, artifact_root, result_root, model):
    """Preserve source engine capabilities and immutable ONNX sidecar inodes."""
    onnx_relative, old_relative = LINEAGES[model]
    base = root / ".local/artifacts/v0101-forward-port"
    onnx = base / onnx_relative
    old = base / old_relative
    output = artifact_root / model
    report = result_root / model
    if output.exists() or output.is_symlink() or report.exists(
    ) or report.is_symlink():
        raise FileExistsError("Refusing to overwrite engine or result: " +
                              model)
    config = json.loads((old / "config.json").read_text())["builder_config"]
    if config.get("spec_base") or config.get("spec_draft") or config.get(
            "tp_size", 1) != 1:
        raise ValueError(
            "This rebuild recipe supports vanilla single-GPU engines only")
    for path in (onnx / "model.onnx", onnx / "model.onnx.data",
                 onnx / "config.json", build / "examples/llm/llm_build",
                 build / "libNvInfer_edgellm_plugin.so.1.0"):
        if not path.is_file():
            raise FileNotFoundError(path)
    sidecars = sorted(path.name for path in onnx.glob("*.safetensors"))
    if not sidecars:
        raise FileNotFoundError("No external weight sidecars: " + str(onnx))
    link_root = pathlib.Path(os.path.commonpath([onnx, output]))
    if not link_root.is_relative_to(root / ".local/artifacts"):
        raise ValueError(
            "Output engines must remain in the shared .local/artifacts store")
    mounts = [
        "-v",
        str(onnx) + ":/opt/onnx:ro", "-v",
        str(output) + ":/opt/output:rw"
    ]
    isolation = [
        "--pull", "never", "--network", "none", "--read-only", "--cap-drop",
        "ALL", "--security-opt", "no-new-privileges"
    ]
    # One mount avoids cross-mount link failures. FOWNER permits linking Cosmos
    # sidecars owned by another UID; DAC_OVERRIDE permits writing the new directory.
    link = [
        "docker", "run", "--rm", *isolation, "--cap-add", "FOWNER",
        "--cap-add", "DAC_OVERRIDE", "--user", "0:0", "-v",
        str(link_root) + ":/opt/artifacts:rw", "--entrypoint", "/bin/ln", IMAGE
    ]
    link += [
        str(
            pathlib.Path("/opt/artifacts") /
            (onnx / name).relative_to(link_root)) for name in sidecars
    ]
    link += [
        str(pathlib.Path("/opt/artifacts") / output.relative_to(link_root)) +
        "/"
    ]
    command = [
        "docker", "run", "--rm", "--gpus", "all", *isolation, "--tmpfs",
        "/tmp:rw,size=536870912", "--user", f"{os.getuid()}:{os.getgid()}",
        *mounts, "-v",
        str(build) + ":/opt/edgellm:ro", "-e", "TRT_PACKAGE_DIR=/opt/tensorrt",
        "-e",
        "EDGELLM_PLUGIN_PATH=/opt/edgellm/libNvInfer_edgellm_plugin.so.1.0",
        "-e",
        "LD_LIBRARY_PATH=/opt/edgellm:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu",
        "-e", "CUDA_CACHE_PATH=/tmp/cuda-cache", IMAGE,
        "/opt/edgellm/examples/llm/llm_build", "--onnxDir", "/opt/onnx",
        "--engineDir", "/opt/output"
    ]
    for name, flag in FLAGS.items():
        if name in config:
            command.extend([flag, str(config[name])])
    if config.get("allow_kv_pool_undercommit"):
        command.append("--allowKVPoolUndercommit")
    return {
        "model": model,
        "source_onnx": str(onnx),
        "reference_engine": str(old),
        "output_engine": str(output),
        "result_dir": str(report),
        "builder_config": config,
        "sidecars": sidecars,
        "sidecar_link_command": link,
        "build_command": command,
    }


def execute(plan, root, build):
    """Build one new artifact and retain failed attempts without promotion."""
    output = pathlib.Path(plan["output_engine"])
    report = pathlib.Path(plan["result_dir"])
    output.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=False)
    onnx = pathlib.Path(plan["source_onnx"])
    manifest = {
        "state":
        "diagnostic",
        "status":
        "building",
        "plan":
        plan,
        "source_commit":
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "source_status":
        subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True),
        "builder_sha256":
        sha256(build / "examples/llm/llm_build"),
        "plugin_sha256":
        sha256(build / "libNvInfer_edgellm_plugin.so.1.0"),
        "onnx_sha256":
        sha256(onnx / "model.onnx"),
        "onnx_data_sha256":
        sha256(onnx / "model.onnx.data"),
        "onnx_config_sha256":
        sha256(onnx / "config.json"),
        "sidecar_sha256": {
            name: sha256(onnx / name)
            for name in plan["sidecars"]
        },
        "reference_engine_config_sha256":
        sha256(pathlib.Path(plan["reference_engine"]) / "config.json"),
        "source_checkout_proves_artifact_build_commit":
        False,
        "repeat_count":
        1,
        "inference_validated":
        False,
        "note_references": [
            "notes/331-runtime-contract-memory-and-full24-revalidation-20260926.md"
        ],
    }
    try:
        subprocess.run(plan["sidecar_link_command"], check=True)
        if not all(
                os.path.samefile(onnx / name, output / name)
                for name in plan["sidecars"]):
            raise RuntimeError(
                "Sidecar deduplication did not preserve source inodes")
        with (report / "build.log").open("w") as log:
            subprocess.run(plan["build_command"],
                           stdout=log,
                           stderr=subprocess.STDOUT,
                           check=True)
        manifest["engine_sha256"] = sha256(output / "llm.engine")
        generated = json.loads(
            (output / "config.json").read_text())["builder_config"]
        if generated != plan["builder_config"]:
            raise RuntimeError(
                "Rebuilt engine capability differs from source builder_config")
        manifest["status"] = "built_not_inference_validated"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        (report /
         "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    """Print the no-overwrite plan by default; never update current pointers."""
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*LINEAGES, "all"), default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--build-root",
                        type=pathlib.Path,
                        default=root / ".local/builds/v0101-validation")
    parser.add_argument(
        "--artifact-root",
        type=pathlib.Path,
        default=root /
        ".local/artifacts/v0101-forward-port/workspace-corrected-20260926")
    parser.add_argument(
        "--result-root",
        type=pathlib.Path,
        default=root /
        ".local/results/runtime-contract-revalidation-20260926/workspace-rebuild"
    )
    args = parser.parse_args()
    models = list(LINEAGES) if args.model == "all" else [args.model]
    plans = [
        plan_model(root,
                   args.build_root.resolve(), args.artifact_root.resolve(),
                   args.result_root.resolve(), model) for model in models
    ]
    print(json.dumps({"execute": args.execute, "plans": plans}, indent=2))
    if args.execute:
        for plan in plans:
            execute(plan, root, args.build_root.resolve())


if __name__ == "__main__":
    main()
