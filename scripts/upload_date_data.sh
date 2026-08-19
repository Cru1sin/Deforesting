#!/usr/bin/env bash
set -euo pipefail

# Archive ownership is determined by location:
# data/NNNN.zip                 = user-owned source, always keep
# data/.date_archive_tmp/NNNN.zip = script-generated temporary ZIP
# data/NNNN/                    = source directory, always keep

FORCE=0
if (( $# == 1 )) && [[ "$1" == --force ]]; then
    FORCE=1
elif (( $# != 0 )); then
    echo "Usage: bash scripts/upload_date_data.sh [--force]" >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
TMP_DIR="${TMP_DIR:-$DATA_ROOT/.date_archive_tmp}"
# onedrive_hkust is the rclone label for the connected Personal drive;
# HKUST/Project/Defrost/data is a folder inside that drive.
CLOUD_REMOTE="${CLOUD_REMOTE:-onedrive_hkust:HKUST/Project/Defrost/data}"

ZIP_BIN="${ZIP_BIN:-/usr/bin/zip}"
RCLONE="${RCLONE:-rclone}"
DF_BIN="${DF_BIN:-/bin/df}"
SLEEP_BIN="${SLEEP_BIN:-/bin/sleep}"
STAT_BIN="${STAT_BIN:-/usr/bin/stat}"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Data directory not found: $DATA_ROOT" >&2
    exit 1
fi

if [[ ! -x "$ZIP_BIN" ]] || [[ ! -x "$STAT_BIN" ]]; then
    echo "Required zip/stat executable not found." >&2
    exit 1
fi

if ! command -v "$RCLONE" >/dev/null 2>&1; then
    echo "rclone executable not found: $RCLONE" >&2
    exit 1
fi

if pgrep -x rclone >/dev/null 2>&1; then
    if (( FORCE == 0 )); then
        echo "Another rclone transfer is already running; retry or use --force." >&2
        exit 1
    fi
    echo "[FORCE] another rclone transfer is running; continue anyway"
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
        onedrive_is_running || break
        "$SLEEP_BIN" 1
    done

    if onedrive_is_running; then
        echo "Unable to pause OneDrive desktop client safely." >&2
        exit 1
    fi
fi

mkdir -p "$TMP_DIR"
CLOUD_REMOTE="${CLOUD_REMOTE%/}"

# mkdir is idempotent and also checks authentication before packaging.
if ! "$RCLONE" mkdir "$CLOUD_REMOTE"; then
    echo "Unable to access OneDrive cloud path: $CLOUD_REMOTE" >&2
    exit 1
fi

wait_for_space() {
    local needed_kib="$1"
    local date_name="$2"
    local free_kib
    local usable_kib

    while true; do
        free_kib="$("$DF_BIN" -Pk "$TMP_DIR" | awk 'NR == 2 {print $4; exit}')"
        usable_kib=$((free_kib * 80 / 100))
        (( needed_kib <= usable_kib )) && return 0

        echo "[WAIT] $date_name needs $needed_kib KiB; 80% allowance is $usable_kib KiB"
        echo "[WAIT] checking again in 30s"
        "$SLEEP_BIN" 30
    done
}

verify_cloud() {
    local remote_file="$1"
    local expected_bytes="$2"
    local remote_size

    remote_size="$(
        "$RCLONE" lsf "$remote_file" --files-only --format s 2>/dev/null
    )" || return 1

    [[ "$remote_size" =~ ^[0-9]+$ ]] \
        && (( remote_size > 0 )) \
        && [[ "$remote_size" == "$expected_bytes" ]] \
        && "$RCLONE" cat "$remote_file" --head 1 --discard >/dev/null 2>&1 \
        && "$RCLONE" cat "$remote_file" --tail 1 --discard >/dev/null 2>&1
}

upload_cloud() {
    "$RCLONE" copyto "$archive" "$remote_file" "$1" \
        --onedrive-chunk-size 100Mi \
        --timeout 2m \
        --progress \
        --stats 1s \
        --retries 10 \
        --retries-sleep 10s
}

dates=()
while IFS= read -r date_name; do
    dates+=("$date_name")
done < <(
    {
        find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d \
            -name '[0-9][0-9][0-9][0-9]' -exec basename {} \;
        find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type f \
            -name '[0-9][0-9][0-9][0-9].zip' -exec basename {} .zip \;
    } | sort -u
)

total=${#dates[@]}
if (( total == 0 )); then
    echo "No four-digit date directories or ZIPs found."
    exit 0
fi

echo "========================================"
echo "Date data upload to OneDrive cloud"
echo "Data   : $DATA_ROOT"
echo "Cloud  : $CLOUD_REMOTE"
echo "Dates  : $total"
echo "Sources: always preserved"
echo "========================================"

for i in "${!dates[@]}"; do
    date_name="${dates[$i]}"
    source_dir="$DATA_ROOT/$date_name"
    source_zip="$DATA_ROOT/$date_name.zip"
    temporary_zip="$TMP_DIR/$date_name.zip"
    pack_dir="$TMP_DIR/.$date_name.pack"
    building_zip="$pack_dir/$date_name.zip"
    remote_file="$CLOUD_REMOTE/$date_name.zip"

    echo
    echo "[$((i + 1))/$total] $date_name"

    if [[ -f "$source_zip" ]]; then
        archive="$source_zip"
        disposable=0
        echo "[PACK] existing source ZIP found, preserve"
    else
        archive="$temporary_zip"
        disposable=1

        if [[ -f "$archive" ]]; then
            echo "[PACK] existing generated ZIP found, reuse"
        else
            [[ -d "$source_dir" ]] || {
                echo "Source directory not found: $source_dir" >&2
                exit 1
            }
            needed_kib="$(du -sk "$source_dir" | awk '{print $1}')"
            wait_for_space "$needed_kib" "$date_name"

            echo "[PACK] $date_name"
            rm -rf "$pack_dir"
            mkdir -p "$pack_dir"
            (
                cd "$DATA_ROOT"
                "$ZIP_BIN" -q -0 -r "$building_zip" "$date_name"
            )
            mv "$building_zip" "$archive"
            rmdir "$pack_dir"
            echo "[PACK] done"
        fi
    fi

    archive_bytes="$("$STAT_BIN" -f '%z' "$archive")"
    if [[ ! "$archive_bytes" =~ ^[0-9]+$ ]] || (( archive_bytes == 0 )); then
        echo "Invalid ZIP size; preserving all local data." >&2
        exit 1
    fi

    echo "[UPLOAD] verify or transfer $date_name.zip"
    upload_cloud --size-only

    if ! verify_cloud "$remote_file" "$archive_bytes"; then
        echo "[UPLOAD] cloud object incomplete or unreadable; force one replacement"
        upload_cloud --ignore-times
        if ! verify_cloud "$remote_file" "$archive_bytes"; then
            echo "[VERIFY] cloud confirmation failed; preserving all local data" >&2
            exit 1
        fi
    fi
    echo "[VERIFY] OneDrive cloud ZIP confirmed: $archive_bytes bytes"

    if (( disposable == 1 )); then
        rm -f "$archive"
        echo "[CLEAN] generated temporary ZIP removed"
    else
        echo "[KEEP] source ZIP preserved"
    fi
    [[ ! -d "$source_dir" ]] || echo "[KEEP] source directory preserved"
done

echo
echo "========================================"
echo "DONE: $total dates uploaded and cloud-verified"
echo "========================================"
