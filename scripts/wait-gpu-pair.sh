#!/usr/bin/env bash
# Foreground waiter: run with nohup; the experiment itself uses --detach.
set -uo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd -- "$repo_dir" || exit 1
threshold_mib=$((30 * 1024))
interval_s=30

log() { printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }

while :; do
    if ! snapshot=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits); then
        log 'nvidia-smi failed; retrying'
        sleep "$interval_s"
        continue
    fi
    first_uuid= second_uuid= first_free=-1 second_free=-1
    declare -A seen=()
    malformed=0
    while IFS= read -r line; do
        if [[ $line =~ ^[[:space:]]*(GPU-[a-fA-F0-9-]+)[[:space:]]*,[[:space:]]*([0-9]+)[[:space:]]*$ ]]; then
            uuid=${BASH_REMATCH[1]}
            value=${BASH_REMATCH[2]}
            if [[ -n ${seen[$uuid]+x} ]] || ((${#value} > 9)); then
                malformed=1
                break
            fi
            seen[$uuid]=1
            free=$((10#$value))
            if ((free > first_free)); then
                second_uuid=$first_uuid; second_free=$first_free
                first_uuid=$uuid; first_free=$free
            elif ((free > second_free)); then
                second_uuid=$uuid; second_free=$free
            fi
        else
            # An unreadable device is never selected; retry if fewer than two remain.
            log 'Skipping an unreadable GPU row'
        fi
    done <<< "$snapshot"
    unset seen
    if ((malformed)) || [[ -z $second_uuid ]]; then
        log 'Fewer than two valid distinct GPU readings, or duplicate UUID; retrying'
        sleep "$interval_s"
        continue
    fi
    total=$((first_free + second_free))
    log "Best pair: $first_uuid=${first_free}MiB $second_uuid=${second_free}MiB total=${total}MiB; need >${threshold_mib}MiB"
    if ((total > threshold_mib)); then
        export CUDA_VISIBLE_DEVICES="$first_uuid,$second_uuid"
        log "Threshold met; launching once with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        exec uv run --locked --extra inference python \
            experiments/nano_vllm_qwen3_8b_long_context/run.py \
            --config-name request_scaling mode=run --detach
    fi
    sleep "$interval_s"
done
