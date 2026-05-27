# SWE-bench Verified ARM Image Handoff

This repo contains the ARM64 build wrapper and instance lists for `princeton-nlp/SWE-bench_Verified`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
```

## Build

```bash
export REGISTRY=registry.example.com/namespace/swe-bench

python3 -m swebench.harness.prepare_images \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --instance_ids_file swebench_verified_instances.txt \
  --arch arm64 \
  --max_workers 8 \
  --state_file build_state.json \
  --registry "${REGISTRY}"
```

For Slurm:

```bash
REGISTRY=registry.example.com/namespace/swe-bench sbatch launch_arm_build.slurm
```

For Slurm with verification before push:

```bash
REGISTRY=registry.example.com/namespace/swe-bench sbatch launch_arm_build.slurm --verify true
```

## Verification Before Push

SWE-bench Verified supports gold-patch verification before registry push with `--verify true`. When `--verify true` and `--registry` are both set, the script builds the image, runs the gold patch eval, and only pushes if verification passes.

```bash
export REGISTRY=registry.example.com/namespace/swe-bench

python3 -m swebench.harness.prepare_images \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --instance_ids_file swebench_verified_instances.txt \
  --arch arm64 \
  --max_workers 8 \
  --state_file build_state.json \
  --registry "${REGISTRY}" \
  --verify true
```

Failed-instance lists are under `handoff/failed_instances/`.
