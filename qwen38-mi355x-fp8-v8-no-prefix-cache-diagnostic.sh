#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Qwen3.8 Flash Next FP8 - MI355X / gfx950
# FP8 + MTP3 DIAGNOSTIC: SLOT-MAP OOB GUARD + PREFIX CACHE DISABLED
#
# Derived from the exact qwen38-mi355x-all-v4-mtp3.sh configuration that
# produced ~211 tok/s with MTP3 on one MI355X.
#
# This script is intentionally conservative: it preserves the known-good
# serving parameters and only adds host bootstrap / convenience commands.
# ============================================================================

# ----------------------------- exported settings -----------------------------
export BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai-rocm:nightly@sha256:91e381f072d6a44e1e4c97c82dce06e50e5189905cb3999a11471c5a8fc6a563}"
export PATCHED_IMAGE="${PATCHED_IMAGE:-qwen38-mi355x-rocm:ple-fp8-slotmap-v8-noprefix}"
export CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-mi355x}"
export MODEL="${MODEL:-Qwen/Qwen3.8-Flash-Next-FP8}"

export PORT="${PORT:-8000}"
export GPU_ID="${GPU_ID:-0}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export MTP_TOKENS="${MTP_TOKENS:-3}"

# The known-good script used --max-model-len auto. For this model that resolves
# to the native model limit; keep it unchanged to preserve the baseline.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"

# Known-good ROCm feature switches.
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-rocm}"
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE:-0}"

# Optional credentials. VLLM_API_KEY is required to start.
export VLLM_API_KEY="${VLLM_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"

export HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
export VLLM_CACHE_VOLUME="${VLLM_CACHE_VOLUME:-qwen38-vllm-cache}"
export WORKDIR="${WORKDIR:-${HOME}/.qwen38-mi355x-fp8-build}"

# Docker build settings.
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

usage() {
  cat <<EOF
Usage:
  $0 install       Check/install Docker Engine + Docker Buildx
  $0 build         Pull vLLM ROCm nightly and build patched FP8 image
  $0 check         Validate GPU, vLLM, Transformers and AMD PLE FP8 patch
  $0 start         Start the known-good FP8 + MTP3 server
  $0 stop          Stop/remove server container
  $0 restart       Stop then start
  $0 status        Show container status
  $0 logs          Follow server logs
  $0 test          Query /v1/models
  $0 rebuild       Remove patched image, rebuild and validate
  $0 exports       Print the environment used by this baseline

Important exported variables:
  BASE_IMAGE                  ${BASE_IMAGE}
  PATCHED_IMAGE               ${PATCHED_IMAGE}
  MODEL                       ${MODEL}
  PORT                        ${PORT}
  GPU_ID                      ${GPU_ID}
  GPU_MEMORY_UTILIZATION      ${GPU_MEMORY_UTILIZATION}
  MAX_NUM_SEQS                ${MAX_NUM_SEQS}
  MAX_MODEL_LEN               ${MAX_MODEL_LEN}
  MTP_TOKENS                  ${MTP_TOKENS}
  VLLM_ROCM_USE_AITER         ${VLLM_ROCM_USE_AITER}
  VLLM_ROCM_USE_AITER_MOE     ${VLLM_ROCM_USE_AITER_MOE}

Before start:
  export VLLM_API_KEY="\$(openssl rand -hex 32)"
  export HF_TOKEN="hf_..."      # optional

Known functional baseline:
  Model       : Qwen/Qwen3.8-Flash-Next-FP8
  GPU         : 1 x MI355X / gfx950
  TP          : 1
  AITER       : ON
  AITER MoE   : OFF
  MTP         : 3
  Max seqs    : 4 (diagnostic default)
  Tool parser : qwen3_xml
  Reasoning   : qwen3
  Measured    : ~211 tok/s single stream in the successful run
EOF
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif have_cmd sudo; then
    sudo "$@"
  else
    echo "ERROR: root privileges are required and sudo is not installed." >&2
    exit 1
  fi
}

install_docker_ubuntu() {
  if [[ ! -r /etc/os-release ]]; then
    echo "ERROR: /etc/os-release not found; automatic Docker installation supports Ubuntu/Debian hosts." >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  source /etc/os-release

  case "${ID:-}" in
    ubuntu|debian) ;;
    *)
      echo "ERROR: automatic Docker installation supports Ubuntu/Debian; detected ID=${ID:-unknown}." >&2
      exit 1
      ;;
  esac

  echo "Installing Docker Engine and Buildx from Docker's official apt repository..."

  as_root apt-get update
  as_root apt-get install -y ca-certificates curl gnupg

  as_root install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/${ID}/gpg" \
    | as_root gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
  as_root chmod a+r /etc/apt/keyrings/docker.gpg

  local arch codename
  arch="$(dpkg --print-architecture)"
  codename="${VERSION_CODENAME:-}"
  if [[ -z "${codename}" ]]; then
    codename="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-}")"
  fi
  [[ -n "${codename}" ]] || {
    echo "ERROR: cannot determine apt distribution codename." >&2
    exit 1
  }

  echo \
    "deb [arch=${arch} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${ID} ${codename} stable" \
    | as_root tee /etc/apt/sources.list.d/docker.list >/dev/null

  as_root apt-get update
  as_root apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

  as_root systemctl enable --now docker 2>/dev/null || true
}

ensure_docker() {
  local need_install=0

  if ! have_cmd docker; then
    echo "Docker CLI not found."
    need_install=1
  fi

  if [[ "${need_install}" -eq 1 ]]; then
    install_docker_ubuntu
  fi

  if ! docker version >/dev/null 2>&1; then
    # Docker may be installed but daemon stopped.
    as_root systemctl enable --now docker 2>/dev/null || true
  fi

  docker version >/dev/null 2>&1 || {
    echo "ERROR: Docker is installed but the daemon is not usable." >&2
    echo "Try: sudo systemctl status docker" >&2
    exit 1
  }

  if ! docker buildx version >/dev/null 2>&1; then
    echo "Docker Buildx not found; installing docker-buildx-plugin..."
    install_docker_ubuntu
  fi

  docker buildx version >/dev/null 2>&1 || {
    echo "ERROR: Docker Buildx is still unavailable after installation." >&2
    exit 1
  }

  echo "Docker:  $(docker --version)"
  echo "Buildx:  $(docker buildx version | head -1)"
}

check_host() {
  ensure_docker

  [[ -e /dev/kfd ]] || {
    echo "ERROR: /dev/kfd missing. ROCm GPU device is not exposed on the host." >&2
    exit 1
  }

  [[ -d /dev/dri ]] || {
    echo "ERROR: /dev/dri missing. ROCm DRM devices are not exposed on the host." >&2
    exit 1
  }

  mkdir -p "${HF_CACHE}" "${WORKDIR}"
}

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"
}

container_running() {
  docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"
}

image_exists() {
  docker image inspect "${PATCHED_IMAGE}" >/dev/null 2>&1
}

write_build_files() {
  cat > "${WORKDIR}/patch_amd_ple_fp8.py" <<'PYEOF'
#!/usr/bin/env python3
from pathlib import Path
import shutil
import sys

target = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "vllm/models/qwen4_exp/amd/ple_layer.py"
)
marker = "# QWEN38_AMD_PLE_FP8_SCALE_PATCH"

if not target.exists():
    raise SystemExit(f"ERROR: missing {target}")

text = target.read_text()

if marker in text:
    print("Patch already applied")
    sys.exit(0)

backup = target.with_suffix(".py.orig")
if not backup.exists():
    shutil.copy2(target, backup)

needle1 = '''        self.ngram_embedding = PLEVocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
        )
'''

repl1 = '''        self.ngram_embedding = PLEVocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
        )

        # QWEN38_AMD_PLE_FP8_SCALE_PATCH
        self.ngram_embedding.register_buffer(
            "weight_scale",
            torch.ones((), dtype=torch.float32),
            persistent=False,
        )
'''

if needle1 not in text:
    raise SystemExit(
        "ERROR: constructor pattern not found; nightly source changed"
    )

text = text.replace(needle1, repl1, 1)

needle2 = '''        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
'''

repl2 = '''        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue

            # QWEN38_AMD_PLE_FP8_SCALE_PATCH
            if name == "ngram_embedding.weight_scale":
                if loaded_weight.numel() != 1:
                    raise ValueError(
                        "Expected scalar PLE FP8 weight scale, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                self.ngram_embedding.weight_scale.copy_(
                    loaded_weight.reshape(()).to(
                        device=self.ngram_embedding.weight_scale.device,
                        dtype=self.ngram_embedding.weight_scale.dtype,
                    )
                )
                loaded.add(name)
                continue
'''

if needle2 not in text:
    raise SystemExit(
        "ERROR: load_weights pattern not found; nightly source changed"
    )

text = text.replace(needle2, repl2, 1)

needle3 = '''        torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding(
            ngram_ids,
            output,
            self.layer_name,
        )
        return output
'''

repl3 = '''        torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding(
            ngram_ids,
            output,
            self.layer_name,
        )

        # QWEN38_AMD_PLE_FP8_SCALE_PATCH
        output.mul_(self.ngram_embedding.weight_scale.to(dtype=output.dtype))
        return output
'''

if needle3 not in text:
    raise SystemExit(
        "ERROR: forward pattern not found; nightly source changed"
    )

text = text.replace(needle3, repl3, 1)

target.write_text(text)
print("Patched:", target)
PYEOF

  cat > "${WORKDIR}/patch_vllm_slotmap_oob.py" <<'PYEOF'
#!/usr/bin/env python3
from pathlib import Path
import re
import shutil
import sys

target = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "vllm/v1/worker/block_table.py"
)
marker = "# QWEN38_SLOTMAP_OOB_GUARD"
range_expr = "in_range = block_indices < block_table_stride"

if not target.exists():
    raise SystemExit(f"ERROR: missing {target}")

text = target.read_text()

def validate(src: str) -> None:
    checks = {
        "marker": marker in src,
        "range": re.search(
            r"in_range\s*=\s*block_indices\s*<\s*block_table_stride", src
        ) is not None,
        "masked_load": re.search(
            r"mask\s*=\s*mask\s*&\s*is_local\s*&\s*in_range\s*,", src
        ) is not None,
        "masked_slot": re.search(
            r"tl\.where\(\s*is_local\s*&\s*in_range\s*,\s*slot_ids\s*,", src
        ) is not None,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("ERROR: slot-map validation failed: " + ", ".join(failed))
    compile(src, str(target), "exec")

# Already patched by this script.
if marker in text:
    validate(text)
    print("vLLM slot-mapping OOB guard already applied: OK")
    raise SystemExit(0)

# If upstream already contains the equivalent fix, only add our marker.
if (
    re.search(r"in_range\s*=\s*block_indices\s*<\s*block_table_stride", text)
    and re.search(
        r"mask\s*=\s*mask\s*&\s*is_local\s*&\s*in_range\s*,", text
    )
    and re.search(
        r"tl\.where\(\s*is_local\s*&\s*in_range\s*,\s*slot_ids\s*,", text
    )
):
    m = re.search(
        r"^(?P<indent>[ \t]*)in_range\s*=\s*block_indices\s*<\s*block_table_stride",
        text,
        re.MULTILINE,
    )
    if m is None:
        raise SystemExit("ERROR: upstream guard found but marker insertion point missing")
    text = text[:m.start()] + m.group("indent") + marker + "\n" + text[m.start():]
    target.write_text(text)
    validate(text)
    print("Equivalent slot-map OOB guard already present upstream: OK")
    raise SystemExit(0)

backup = target.with_suffix(".py.slotmap-orig")
if not backup.exists():
    shutil.copy2(target, backup)

# Structural matcher: locate the tl.load that indexes the flattened block
# table with row_offset + block_indices. Do not depend on exact formatting
# or on whether the padding constexpr is named PAD_ID or PAD_SLOT_ID.
anchor = "block_table_ptr + row_offset + block_indices"
anchor_positions = [m.start() for m in re.finditer(re.escape(anchor), text)]
candidates = []

for pos in anchor_positions:
    search_start = max(0, pos - 600)
    load_token = "block_numbers = tl.load("
    load_pos = text.rfind(load_token, search_start, pos)
    if load_pos < 0:
        continue

    line_start = text.rfind("\n", 0, load_pos) + 1
    indent = text[line_start:load_pos]
    if indent.strip():
        continue

    close_token = ").to(tl.int64)"
    close_pos = text.find(close_token, pos, min(len(text), pos + 900))
    if close_pos < 0:
        continue
    load_end = close_pos + len(close_token)
    load_block = text[line_start:load_end]

    # Must be the vulnerable load, not an unrelated block-table access.
    if not re.search(r"mask\s*=\s*mask\s*&\s*is_local\s*,", load_block):
        continue
    if "in_range" in load_block:
        continue

    prior = text[max(0, line_start - 1800):line_start]
    if "block_indices" not in prior or "block_table_stride" not in prior:
        continue

    candidates.append((line_start, load_end, indent, load_block))

if len(candidates) != 1:
    print(
        f"ERROR: expected exactly one vulnerable slot-map load; "
        f"found {len(candidates)} candidate(s), {len(anchor_positions)} anchor(s).",
        file=sys.stderr,
    )
    for i, pos in enumerate(anchor_positions, 1):
        lo = max(0, pos - 450)
        hi = min(len(text), pos + 700)
        print(f"--- anchor {i} context ---", file=sys.stderr)
        print(text[lo:hi], file=sys.stderr)
    raise SystemExit(1)

line_start, load_end, indent, load_block = candidates[0]

patched_load, count = re.subn(
    r"(?m)^(?P<i>[ \t]*)mask\s*=\s*mask\s*&\s*is_local\s*,",
    r"\g<i>mask=mask & is_local & in_range,",
    load_block,
    count=1,
)
if count != 1:
    raise SystemExit("ERROR: could not patch vulnerable tl.load mask")

insert = (
    indent + marker + "\n"
    + indent + range_expr + "\n"
    + patched_load
)
text = text[:line_start] + insert + text[load_end:]

# Patch the matching slot-id validity expression that follows this load.
search_from = line_start + len(insert)
search_to = min(len(text), search_from + 1200)
tail = text[search_from:search_to]
tail2, count = re.subn(
    r"(slot_ids\s*=\s*tl\.where\(\s*)is_local(\s*,\s*slot_ids\s*,)",
    r"\1is_local & in_range\2",
    tail,
    count=1,
)
if count != 1:
    raise SystemExit("ERROR: could not patch following slot_ids tl.where condition")
text = text[:search_from] + tail2 + text[search_to:]

target.write_text(text)
validate(text)
print("vLLM slot-mapping out-of-bounds guard: OK")
PYEOF

  cat > "${WORKDIR}/Dockerfile" <<EOF
FROM ${BASE_IMAGE}
COPY patch_amd_ple_fp8.py /tmp/patch_amd_ple_fp8.py
COPY patch_vllm_slotmap_oob.py /tmp/patch_vllm_slotmap_oob.py
RUN python3 /tmp/patch_amd_ple_fp8.py \
 && python3 /tmp/patch_vllm_slotmap_oob.py \
 && python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ple_layer.py \
 && python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/block_table.py \
 && grep -q QWEN38_AMD_PLE_FP8_SCALE_PATCH /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ple_layer.py \
 && grep -q QWEN38_SLOTMAP_OOB_GUARD /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/block_table.py \
 && grep -Eq "mask[[:space:]]*=[[:space:]]*mask[[:space:]]*&[[:space:]]*is_local[[:space:]]*&[[:space:]]*in_range" /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/block_table.py
ENTRYPOINT ["vllm"]
EOF
}

build_image() {
  check_host
  write_build_files

  echo
  echo "Pulling ${BASE_IMAGE} ..."
  docker pull "${BASE_IMAGE}"

  echo
  echo "Building ${PATCHED_IMAGE} with Docker Buildx ..."
  docker buildx build \
    --load \
    --no-cache \
    -t "${PATCHED_IMAGE}" \
    "${WORKDIR}"

  echo "Built: ${PATCHED_IMAGE}"
}

check_image() {
  check_host
  image_exists || build_image

  echo
  echo "Validating ROCm image, PLE FP8 patch, and slot-map OOB guard..."

  docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -e VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE}" \
    -e HIP_VISIBLE_DEVICES="${GPU_ID}" \
    -e ROCR_VISIBLE_DEVICES="${GPU_ID}" \
    --entrypoint python3 \
    "${PATCHED_IMAGE}" \
    -c '
import importlib.metadata
from pathlib import Path
import re
import torch
import transformers
from transformers import CONFIG_MAPPING

p = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "vllm/models/qwen4_exp/amd/ple_layer.py"
)
s = p.read_text()

bt = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "vllm/v1/worker/block_table.py"
)
bts = bt.read_text()

print("vLLM:", importlib.metadata.version("vllm"))
print("Transformers:", transformers.__version__)
print("Torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Arch:", torch.cuda.get_device_properties(0).gcnArchName)
print("qwen4_exp:", "qwen4_exp" in CONFIG_MAPPING)
print("AMD PLE patch:", "QWEN38_AMD_PLE_FP8_SCALE_PATCH" in s)
print("Slot-map OOB guard:", "QWEN38_SLOTMAP_OOB_GUARD" in bts)

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1
assert "qwen4_exp" in CONFIG_MAPPING
assert "QWEN38_AMD_PLE_FP8_SCALE_PATCH" in s
assert "QWEN38_SLOTMAP_OOB_GUARD" in bts
assert re.search(r"mask\s*=\s*mask\s*&\s*is_local\s*&\s*in_range\s*,", bts)
assert re.search(r"tl\.where\(\s*is_local\s*&\s*in_range\s*,\s*slot_ids\s*,", bts)
'

  echo "Image validation: OK"
}

require_api_key() {
  if [[ -z "${VLLM_API_KEY}" ]]; then
    cat >&2 <<'EOF'
ERROR: VLLM_API_KEY is empty.

Set it before starting, for example:

  export VLLM_API_KEY="$(openssl rand -hex 32)"

Then rerun:
  ./qwen38-mi355x-fp8-final-functional-all-in-one.sh start
EOF
    exit 1
  fi
}

start_server() {
  check_host
  require_api_key

  if container_running; then
    echo "${CONTAINER_NAME} already running"
    return 0
  fi

  container_exists && docker rm -f "${CONTAINER_NAME}" >/dev/null

  image_exists || build_image
  check_image

  echo
  echo "============================================================"
  echo " Qwen3.8 Flash Next FP8 - known functional MI355X baseline"
  echo "============================================================"
  echo "Image                  : ${PATCHED_IMAGE}"
  echo "Model                  : ${MODEL}"
  echo "GPU                    : ${GPU_ID} / MI355X gfx950"
  echo "Tensor parallel        : 1"
  echo "Max model len          : ${MAX_MODEL_LEN}"
  echo "Max sequences          : ${MAX_NUM_SEQS}"
  echo "GPU memory utilization : ${GPU_MEMORY_UTILIZATION}"
  echo "AITER                  : ${VLLM_ROCM_USE_AITER}"
  echo "AITER MoE              : ${VLLM_ROCM_USE_AITER_MOE}"
  echo "MTP                    : ${MTP_TOKENS}"
  echo "Tool parser            : qwen3_xml"
  echo "Reasoning parser       : qwen3"
  echo "Port                   : ${PORT}"
  echo "============================================================"
  echo

  docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size=16g \
    --security-opt seccomp=unconfined \
    --network host \
    -e VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE}" \
    -e HIP_VISIBLE_DEVICES="${GPU_ID}" \
    -e ROCR_VISIBLE_DEVICES="${GPU_ID}" \
    -e VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER}" \
    -e VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE}" \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN}" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${VLLM_CACHE_VOLUME}:/root/.cache/vllm" \
    --entrypoint vllm \
    "${PATCHED_IMAGE}" \
    serve "${MODEL}" \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --api-key "${VLLM_API_KEY}" \
      --tensor-parallel-size 1 \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --no-enable-prefix-caching \
      --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}" \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_xml \
      --reasoning-parser qwen3

  echo
  echo "Started: ${CONTAINER_NAME}"
  echo "Follow logs with:"
  echo "  $0 logs"
}

stop_server() {
  ensure_docker

  if container_exists; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null
    echo "Stopped: ${CONTAINER_NAME}"
  else
    echo "${CONTAINER_NAME} is already stopped."
  fi
}

status_server() {
  ensure_docker

  if container_exists; then
    docker ps -a \
      --filter "name=^/${CONTAINER_NAME}$" \
      --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
  else
    echo "STOPPED"
  fi
}

logs_server() {
  ensure_docker
  container_exists || {
    echo "ERROR: ${CONTAINER_NAME} does not exist." >&2
    exit 1
  }
  docker logs -f "${CONTAINER_NAME}"
}

test_server() {
  require_api_key

  if have_cmd curl; then
    curl -fsS \
      "http://127.0.0.1:${PORT}/v1/models" \
      -H "Authorization: Bearer ${VLLM_API_KEY}"
    echo
  else
    echo "ERROR: curl is not installed." >&2
    exit 1
  fi
}

rebuild_all() {
  check_host

  container_exists && docker rm -f "${CONTAINER_NAME}" >/dev/null || true
  docker image rm -f "${PATCHED_IMAGE}" >/dev/null 2>&1 || true

  build_image
  check_image
}

print_exports() {
  cat <<EOF
export BASE_IMAGE='${BASE_IMAGE}'
export PATCHED_IMAGE='${PATCHED_IMAGE}'
export CONTAINER_NAME='${CONTAINER_NAME}'
export MODEL='${MODEL}'
export PORT='${PORT}'
export GPU_ID='${GPU_ID}'
export GPU_MEMORY_UTILIZATION='${GPU_MEMORY_UTILIZATION}'
export MAX_NUM_SEQS='${MAX_NUM_SEQS}'
export MAX_MODEL_LEN='${MAX_MODEL_LEN}'
export MTP_TOKENS='${MTP_TOKENS}'
export VLLM_TARGET_DEVICE='${VLLM_TARGET_DEVICE}'
export VLLM_ROCM_USE_AITER='${VLLM_ROCM_USE_AITER}'
export VLLM_ROCM_USE_AITER_MOE='${VLLM_ROCM_USE_AITER_MOE}'
export HF_CACHE='${HF_CACHE}'
export VLLM_CACHE_VOLUME='${VLLM_CACHE_VOLUME}'
export WORKDIR='${WORKDIR}'
export DOCKER_BUILDKIT='${DOCKER_BUILDKIT}'
export HF_TOKEN='${HF_TOKEN}'
# Set your API key separately:
# export VLLM_API_KEY="\$(openssl rand -hex 32)"
EOF
}

case "${1:-}" in
  install) ensure_docker ;;
  build) build_image ;;
  check) check_image ;;
  start) start_server ;;
  stop) stop_server ;;
  restart) stop_server; start_server ;;
  status) status_server ;;
  logs) logs_server ;;
  test) test_server ;;
  rebuild) rebuild_all ;;
  exports) print_exports ;;
  *) usage; exit 1 ;;
esac
