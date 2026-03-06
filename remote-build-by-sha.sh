#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sslab@117.16.143.14}"
REMOTE_PORT="${REMOTE_PORT:-7022}"
REMOTE_BASE_REPO="${REMOTE_BASE_REPO:-/home/sslab/nfs/thor/TensorRT-Edge-LLM}"
REMOTE_WORKTREE_ROOT="${REMOTE_WORKTREE_ROOT:-/home/sslab/Document}"
REMOTE_WORKTREE_PREFIX="${REMOTE_WORKTREE_PREFIX:-TensorRT-Edge-LLM-wt}"
REMOTE_WORKTREE_NAME="${REMOTE_WORKTREE_NAME:-}"
REMOTE_BUILD_DIR_NAME="${REMOTE_BUILD_DIR_NAME:-build}"
REMOTE_LINK_SUBMODULES_FROM_BASE="${REMOTE_LINK_SUBMODULES_FROM_BASE:-true}"
REMOTE_PUSH_REMOTE="${REMOTE_PUSH_REMOTE:-origin}"
REMOTE_PUSH_REF="${REMOTE_PUSH_REF:-HEAD}"
REMOTE_BUILD_JOBS="${REMOTE_BUILD_JOBS:-\$(nproc)}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-wip: remote build checkpoint}"

usage() {
    cat <<'EOF'
Usage: ./remote-build-by-sha.sh [--no-commit] [--no-push] [--configure-only]

Behavior:
  1. Optionally creates a local checkpoint commit from the current workspace.
  2. Pushes the current HEAD to the configured git remote.
  3. Reuses a fixed remote worktree for this branch/workspace and checks it out at the current commit SHA.
  4. Configures and builds in that remote worktree.

Environment overrides:
  REMOTE_HOST              default: sslab@117.16.143.14
  REMOTE_PORT              default: 7022
  REMOTE_BASE_REPO         default: /home/sslab/nfs/thor/TensorRT-Edge-LLM
  REMOTE_WORKTREE_ROOT     default: /home/sslab/Document
  REMOTE_WORKTREE_PREFIX   default: TensorRT-Edge-LLM-wt
  REMOTE_WORKTREE_NAME     default: <prefix>-<sanitized-current-branch>
  REMOTE_BUILD_DIR_NAME    default: build
  REMOTE_LINK_SUBMODULES_FROM_BASE
                           default: true
  REMOTE_PUSH_REMOTE       default: origin
  REMOTE_PUSH_REF          default: HEAD
  REMOTE_BUILD_JOBS        default: $(nproc) on remote
  COMMIT_MESSAGE           default: wip: remote build checkpoint

Examples:
  ./remote-build-by-sha.sh
  COMMIT_MESSAGE="wip: disagg build check" ./remote-build-by-sha.sh
  REMOTE_BUILD_DIR_NAME=build-jetson-thor ./remote-build-by-sha.sh
EOF
}

need_commit=true
need_push=true
configure_only=false

while (($# > 0)); do
    case "$1" in
        --no-commit)
            need_commit=false
            ;;
        --no-push)
            need_push=false
            ;;
        --configure-only)
            configure_only=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "This script must be run inside a git repository." >&2
    exit 1
fi

if $need_commit; then
    git add -A
    if ! git diff --cached --quiet; then
        git commit -m "${COMMIT_MESSAGE}"
    else
        echo "No staged changes detected; skipping local commit."
    fi
fi

if $need_push; then
    git push "${REMOTE_PUSH_REMOTE}" "${REMOTE_PUSH_REF}"
fi

FULL_SHA="$(git rev-parse HEAD)"
BRANCH_NAME="$(git rev-parse --abbrev-ref HEAD)"
SANITIZED_BRANCH_NAME="$(printf '%s' "${BRANCH_NAME}" | sed 's#[^A-Za-z0-9._-]#-#g')"
WORKTREE_NAME="${REMOTE_WORKTREE_NAME:-${REMOTE_WORKTREE_PREFIX}-${SANITIZED_BRANCH_NAME}}"
REMOTE_WORKTREE_PATH="${REMOTE_WORKTREE_ROOT}/${WORKTREE_NAME}"

if $configure_only; then
    remote_build_cmd='cmake .. \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor'
else
    remote_build_cmd='cmake .. \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor && \
make -j'"${REMOTE_BUILD_JOBS}"
fi

echo "Local HEAD: ${FULL_SHA}"
echo "Remote branch/workspace: ${BRANCH_NAME}"
echo "Remote worktree: ${REMOTE_WORKTREE_PATH}"

ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" \
    FULL_SHA="${FULL_SHA}" \
    REMOTE_BASE_REPO="${REMOTE_BASE_REPO}" \
    REMOTE_WORKTREE_PATH="${REMOTE_WORKTREE_PATH}" \
    REMOTE_BUILD_DIR_NAME="${REMOTE_BUILD_DIR_NAME}" \
    REMOTE_LINK_SUBMODULES_FROM_BASE="${REMOTE_LINK_SUBMODULES_FROM_BASE}" \
    bash -lc "'
set -euo pipefail

cd \"${REMOTE_BASE_REPO}\"

bad_ref_paths=\$(find .git \
    \\( -path \"*@eaDir*\" -o -path \"*SynoEAStream*\" \\) \
    -print 2>/dev/null || true)
if [ -n \"\${bad_ref_paths}\" ]; then
    echo \"Removing invalid ref artifacts before fetch:\"
    printf \"%s\n\" \"\${bad_ref_paths}\"
    while IFS= read -r bad_ref_path; do
        [ -n \"\${bad_ref_path}\" ] || continue
        rm -rf \"\${bad_ref_path}\"
    done <<EOF
\${bad_ref_paths}
EOF
fi

git fetch ${REMOTE_PUSH_REMOTE}
ensure_safe_directory() {
    local target_path=\"\$1\"
    if ! git config --global --get-all safe.directory | grep -Fx -- \"\${target_path}\" >/dev/null 2>&1; then
        git config --global --add safe.directory \"\${target_path}\"
    fi
}

ensure_safe_directory \"${REMOTE_BASE_REPO}\"
ensure_safe_directory \"${REMOTE_WORKTREE_PATH}\"

if [ ! -d \"${REMOTE_WORKTREE_PATH}\" ]; then
    git worktree add --detach \"${REMOTE_WORKTREE_PATH}\" \"${FULL_SHA}\"
else
    git -C \"${REMOTE_WORKTREE_PATH}\" checkout --detach \"${FULL_SHA}\"
fi

cd \"${REMOTE_WORKTREE_PATH}\"

if [ \"${REMOTE_LINK_SUBMODULES_FROM_BASE}\" = \"true\" ] && [ -f .gitmodules ]; then
    while read -r _ submodule_path; do
        base_submodule_path=\"${REMOTE_BASE_REPO}/\${submodule_path}\"
        worktree_submodule_path=\"${REMOTE_WORKTREE_PATH}/\${submodule_path}\"

        if [ ! -d \"\${base_submodule_path}\" ] && [ ! -L \"\${base_submodule_path}\" ]; then
            continue
        fi

        if [ -L \"\${worktree_submodule_path}\" ]; then
            continue
        fi

        if [ -d \"\${worktree_submodule_path}\" ]; then
            if find \"\${worktree_submodule_path}\" -mindepth 1 -maxdepth 1 | read -r _; then
                continue
            fi
            rmdir \"\${worktree_submodule_path}\"
        elif [ -e \"\${worktree_submodule_path}\" ]; then
            continue
        fi

        mkdir -p \"\$(dirname \"\${worktree_submodule_path}\")\"
        ln -s \"\${base_submodule_path}\" \"\${worktree_submodule_path}\"
        echo \"Linked submodule path from base repo: \${submodule_path}\"
    done < <(git config -f .gitmodules --get-regexp path)
fi

mkdir -p \"${REMOTE_BUILD_DIR_NAME}\"
cd \"${REMOTE_BUILD_DIR_NAME}\"
${remote_build_cmd}
'"
