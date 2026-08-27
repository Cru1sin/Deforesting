#!/usr/bin/env bash
set -euo pipefail

# One cycle is one transaction:
# Store ZIP -> direct OneDrive cloud upload -> strict cloud verification ->
# temporary ZIP cleanup -> guarded source-cycle deletion.

if (( $# != 0 )); then
    echo "Usage: bash scripts/data/upload_cycle_images.sh" >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ===== Configuration =====
IMAGES_ROOT="${IMAGES_ROOT:-$PROJECT_ROOT/dataset/images}"
TMP_DIR="${TMP_DIR:-$PROJECT_ROOT/dataset/.image_archive_tmp}"

# This must be an authenticated rclone OneDrive remote, not a local
# ~/Library/CloudStorage path.
CLOUD_REMOTE="${CLOUD_REMOTE:-onedrive_hkust:HKUST/Project/Defrost/dataset/images}"

ZIP_BIN="${ZIP_BIN:-/usr/bin/zip}"
RCLONE="${RCLONE:-rclone}"
DF_BIN="${DF_BIN:-/bin/df}"
SLEEP_BIN="${SLEEP_BIN:-/bin/sleep}"
STAT_BIN="${STAT_BIN:-/usr/bin/stat}"

# =========================

if [[ ! -d "$IMAGES_ROOT" ]]; then
    echo "Images directory not found: $IMAGES_ROOT" >&2
    exit 1
fi

if [[ ! -x "$ZIP_BIN" ]]; then
    echo "zip executable not found: $ZIP_BIN" >&2
    exit 1
fi

if ! command -v "$RCLONE" >/dev/null 2>&1; then
    echo "rclone executable not found: $RCLONE" >&2
    echo "Install it with: brew install rclone" >&2
    exit 1
fi

rclone_direct() {
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        -u http_proxy -u https_proxy -u all_proxy -u RCLONE_HTTP_PROXY \
        NO_PROXY='*' no_proxy='*' "$RCLONE" "$@" --http-proxy ''
}

if [[ ! -x "$STAT_BIN" ]]; then
    echo "stat executable not found: $STAT_BIN" >&2
    exit 1
fi

# rclone and the desktop client must not write the same cloud ZIP together.
ONEDRIVE_SERVICE_PATTERN='/Applications/OneDrive.app/Contents/OneDrive Sync Service.app/Contents/MacOS/OneDrive Sync Service'
ONEDRIVE_WAS_RUNNING=0

onedrive_is_running() {
    pgrep -x OneDrive >/dev/null 2>&1 \
        || pgrep -f "$ONEDRIVE_SERVICE_PATTERN" >/dev/null 2>&1
}

restart_onedrive() {
    if (( ONEDRIVE_WAS_RUNNING == 1 )); then
        echo "[ONEDRIVE] restarting desktop client"
        open -a OneDrive >/dev/null 2>&1 || true
    fi
}

if onedrive_is_running; then
    ONEDRIVE_WAS_RUNNING=1
    trap restart_onedrive EXIT
    echo "[ONEDRIVE] pausing desktop client to prevent concurrent cloud writes"
    osascript -e 'tell application "OneDrive" to quit' >/dev/null 2>&1 || true
    pkill -TERM -x OneDrive >/dev/null 2>&1 || true
    pkill -TERM -f "$ONEDRIVE_SERVICE_PATTERN" >/dev/null 2>&1 || true

    for _ in 1 2 3 4 5 6 7 8 9 10; do
        if ! onedrive_is_running; then
            break
        fi
        "$SLEEP_BIN" 1
    done

    if onedrive_is_running; then
        echo "Unable to pause OneDrive desktop client safely." >&2
        exit 1
    fi
fi

mkdir -p "$TMP_DIR"
IMAGES_ROOT_REAL="$(cd "$IMAGES_ROOT" && pwd -P)"
CLOUD_REMOTE="${CLOUD_REMOTE%/}"

# Authentication and target-path failures must happen before packaging or
# deleting anything.
if ! rclone_direct lsd "$CLOUD_REMOTE" --max-depth 1 >/dev/null; then
    echo "Unable to access OneDrive cloud path: $CLOUD_REMOTE" >&2
    echo "Run 'rclone config' and verify CLOUD_REMOTE before retrying." >&2
    exit 1
fi

format_kib() {
    awk -v kib="$1" 'BEGIN {
        if (kib >= 1048576) {
            printf "%.2f GiB", kib / 1048576
        } else {
            printf "%.1f MiB", kib / 1024
        }
    }'
}

available_kib() {
    "$DF_BIN" -Pk "$TMP_DIR" | awk 'NR == 2 {print $4; exit}'
}

wait_for_space() {
    local needed_kib="$1"
    local cycle_name="$2"
    local free_kib
    local allowance_kib
    local waiting=0

    while true; do
        free_kib="$(available_kib)"
        allowance_kib=$((free_kib * 80 / 100))

        if (( needed_kib <= allowance_kib )); then
            if (( waiting == 1 )); then
                echo "[WAIT] enough space, continue"
            fi
            return 0
        fi

        if (( waiting == 0 )); then
            echo "[WAIT] insufficient free space for $cycle_name"
            echo "[WAIT] needed: $(format_kib "$needed_kib")"
            waiting=1
        fi
        echo "[WAIT] free: $(format_kib "$free_kib"), usable at 80%: $(format_kib "$allowance_kib")"
        echo "[WAIT] completed cycles free source and ZIP space; checking again in 30s"
        "$SLEEP_BIN" 30
    done
}

# This is the deletion gate: exact cloud size plus readable first and last byte.
verify_cloud() {
    local remote_file="$1"
    local expected_bytes="$2"
    local remote_size

    if ! remote_size="$(
        rclone_direct lsf "$remote_file" --files-only --format s 2>/dev/null
    )"; then
        return 1
    fi

    [[ "$remote_size" =~ ^[0-9]+$ ]] \
        && (( remote_size > 0 )) \
        && [[ "$remote_size" == "$expected_bytes" ]] \
        && rclone_direct cat "$remote_file" --head 1 --discard >/dev/null 2>&1 \
        && rclone_direct cat "$remote_file" --tail 1 --discard >/dev/null 2>&1
}

upload_cloud() {
    rclone_direct copyto "$archive" "$remote_file" "$1" \
        --onedrive-chunk-size 100Mi \
        --timeout 2m \
        --progress \
        --stats 1s \
        --retries 10 \
        --retries-sleep 10s
}

remote_size_bytes() {
    rclone_direct lsf "$1" --files-only --format s 2>/dev/null || true
}

# Refuse recursive deletion unless the target is one immediate numeric cycle.
delete_source_cycle() {
    local cycle_dir="$1"
    local cycle_name="$2"
    local parent_real
    local cycle_real

    if [[ ! "$cycle_name" =~ ^frost_cycle_[0-9]+$ ]]; then
        echo "[DELETE] refused unsafe cycle name: $cycle_name" >&2
        return 1
    fi

    parent_real="$(cd "$(dirname "$cycle_dir")" && pwd -P)"
    cycle_real="$(cd "$cycle_dir" && pwd -P)"
    if [[ "$parent_real" != "$IMAGES_ROOT_REAL" ]] \
        || [[ "$(dirname "$cycle_real")" != "$IMAGES_ROOT_REAL" ]] \
        || [[ "$cycle_real" == "$IMAGES_ROOT_REAL" ]]; then
        echo "[DELETE] refused path outside images root: $cycle_dir" >&2
        return 1
    fi

    echo "[DELETE] source images: $cycle_name"
    find "$cycle_real" -depth -delete

    # Finder can recreate an otherwise empty directory with only .DS_Store.
    if [[ -d "$cycle_real" ]] \
        && [[ -z "$(find "$cycle_real" -mindepth 1 ! -name .DS_Store -print -quit)" ]]; then
        rm -f "$cycle_real/.DS_Store"
        rmdir "$cycle_real" 2>/dev/null || true
    fi

    if [[ -e "$cycle_real" ]]; then
        echo "[DELETE] source cycle still exists: $cycle_real" >&2
        return 1
    fi
    echo "[DELETE] source images removed"
}

cycles=()
while IFS= read -r cycle_dir; do
    cycles+=("$cycle_dir")
done < <(
    find "$IMAGES_ROOT" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -name 'frost_cycle_*' \
        | sort
)

total=${#cycles[@]}

if (( total == 0 )); then
    echo "No frost_cycle_* directories found."
    exit 0
fi

echo "========================================"
echo "Direct cycle upload to OneDrive cloud"
echo "Images : $IMAGES_ROOT"
echo "Cloud  : $CLOUD_REMOTE"
echo "Cycles : $total"
echo "Space  : use up to 80% of currently free space"
echo "Cleanup: automatic, only after strict cloud API verification"
echo "========================================"

for i in "${!cycles[@]}"; do
    n=$((i + 1))
    cycle_dir="${cycles[$i]}"
    cycle_name="$(basename "$cycle_dir")"
    zip_name="${cycle_name}.zip"
    archive="$TMP_DIR/$zip_name"
    pack_dir="$TMP_DIR/.${cycle_name}.pack"
    building_archive="$pack_dir/$zip_name"
    remote_file="$CLOUD_REMOTE/$zip_name"

    echo
    echo "[$n/$total] $cycle_name"

    if [[ ! -f "$archive" ]]; then
        estimated_kib="$(du -sk "$cycle_dir" | awk '{print $1}')"
        wait_for_space "$estimated_kib" "$cycle_name"
    fi

    # A finished temporary ZIP survives failed uploads and is safe to reuse.
    if [[ -f "$archive" ]]; then
        echo "[PACK] existing temporary ZIP found, reuse"
        rm -rf "$pack_dir"
    else
        echo "[PACK] $cycle_name"
        rm -rf "$pack_dir"
        mkdir -p "$pack_dir"
        (
            cd "$IMAGES_ROOT"
            "$ZIP_BIN" -q -0 -r "$building_archive" "$cycle_name"
        )
        mv "$building_archive" "$archive"
        rmdir "$pack_dir"
        echo "[PACK] done"
    fi

    archive_bytes="$("$STAT_BIN" -f '%z' "$archive")"
    if [[ ! "$archive_bytes" =~ ^[0-9]+$ ]] || (( archive_bytes == 0 )); then
        echo "[VERIFY] invalid temporary ZIP size; preserving source and temporary files" >&2
        exit 1
    fi

    existing_remote_bytes="$(remote_size_bytes "$remote_file")"
    if [[ "$existing_remote_bytes" =~ ^[0-9]+$ ]] \
        && (( existing_remote_bytes > 0 )) \
        && [[ "$existing_remote_bytes" != "$archive_bytes" ]]; then
        echo "[VERIFY] refusing to replace different-size cloud ZIP" >&2
        echo "[VERIFY] cloud: $existing_remote_bytes bytes; local: $archive_bytes bytes" >&2
        echo "[VERIFY] preserving source and temporary ZIP for manual review" >&2
        exit 1
    fi

    echo "[UPLOAD] verify or transfer $zip_name"
    upload_cloud --size-only

    if ! verify_cloud "$remote_file" "$archive_bytes"; then
        echo "[UPLOAD] cloud object incomplete or unreadable; force one replacement"
        upload_cloud --ignore-times
        if ! verify_cloud "$remote_file" "$archive_bytes"; then
            echo "[VERIFY] cloud confirmation failed; preserving source and temporary ZIP" >&2
            exit 1
        fi
    fi
    echo "[VERIFY] OneDrive cloud ZIP confirmed: $archive_bytes bytes"

    # Nothing local is deleted before verify_cloud succeeds.
    rm -f "$archive"
    echo "[CLEAN] temporary ZIP removed"
    delete_source_cycle "$cycle_dir" "$cycle_name"
done

echo
echo "========================================"
echo "DONE: $total cycles uploaded and cloud-verified"
echo "========================================"
