# Pending: TTFT prefetch sidecar + NVMe-oF canary

Status: pending  
Scope: `spark-vllm-docker`, `spark-trtllm-docker`, `spark-sqlang-docker`, cluster nodes (`local` + `remote`)  
Goal: reduce cold-start and rotation penalties (time-to-first-token) by pre-warming model shards safely, then validate scale-out with a constrained NVMe-oF canary.

## Context

Cold model file reads (open/mmap over many shard files) are a major source of TTFT spikes after restart, node failover, or scheduler rotation.

The cluster already has NVMe-oF canary scripts for initiator/network hardening. This pending work complements that by reducing cold I/O through controlled prefetch and optional local mirror.

## Proposed approach

Implement a low-priority prefetch pattern without changing backend serving code:

1. `init` prefetch at pod/service startup.
2. `sidecar` loop every N seconds to maintain warm residency.
3. `ionice -c3` + `nice -n19` to avoid competing with vLLM/Ray runtime.
4. Prefer `vmtouch -t` for operational simplicity; keep a POSIX `WILLNEED` script fallback.

Optional step for remote storage paths:

- Mirror model shards to local NVMe (`rsync` at low priority) and point backend model path to local mirror.

## Safety guardrails

- Do not prefetch above a fixed budget: max `50%` of `MemAvailable`, and/or an absolute per-node cap (`N` GB).
- Run prefetch in constrained cgroup/container limits (CPU/memory + I/O quota).
- Auto-stop or backoff prefetch when `node_memory_MemAvailable_bytes` drops more than `20%` from baseline.
- Auto-stop or backoff prefetch when NVMe read p95 latency exceeds `2x` baseline.

## Telemetry and validation

Track at least:

- `cache_hit_ratio` (resident pages / total pages from `vmtouch`).
- `open_mmap_latency_p95_ms` (cold/warm script probe).
- `nvme_read_p95_ms` (fio or node storage telemetry).
- `ttft_p95_ms` on representative inference traffic.

Validation mode:

- Run concurrent synthetic read checks while server is alive.
- Compare baseline vs warm runs (minimum 3 cold and 3 warm samples).

## Rollout plan (canary)

1. Enable on one backend in one node only.
2. Validate no throughput regression and no elevated error/reconnect counters.
3. Expand to 1-5% of serving footprint.
4. Expand to second node.
5. Promote to default only after stable TTFT and I/O metrics.

## Rollback plan

- Disable sidecar/init prefetch feature flag.
- Revert to current direct model path behavior.
- If local mirror is enabled, switch backend path back to previous source and stop rsync sync loop.

## Deliverables (pending implementation)

- Helm or Compose profile for `init+sidecar` prefetch.
- Optional `systemd` service template for non-Kubernetes nodes.
- Dashboard panel for TTFT + cache residency + NVMe latency.
- Runbook for canary, promotion, and rollback.

## Relationship with existing canary toolkit

This item should be executed together with:

- `scripts/nvmeof-initiator-canary.sh`
- `scripts/net-tune-canary.sh`
- `scripts/install-nvmeof-canary-units.sh`

The intent is to combine transport/connectivity hardening (NVMe-oF canary) with warm-read behavior (prefetch), using progressive rollout.
