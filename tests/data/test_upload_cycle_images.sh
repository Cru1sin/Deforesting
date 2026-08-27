#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$PROJECT_ROOT/scripts/data/upload_cycle_images.sh"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_file_exists() {
    [[ -f "$1" ]] || fail "expected file to exist: $1"
}

assert_file_missing() {
    [[ ! -e "$1" ]] || fail "expected path to be absent: $1"
}

assert_contains() {
    grep -Fq -- "$2" "$1" || fail "expected '$2' in $1"
}

new_fixture() {
    if [[ -n "${fixture:-}" && -d "$fixture" ]]; then
        rm -rf "$fixture"
    fi
    fixture="$(mktemp -d)"
    images="$fixture/dataset/images"
    tmp="$fixture/dataset/.image_archive_tmp"
    cloud="$fixture/OneDrive/images"
    bin="$fixture/bin"
    log="$fixture/calls.log"
    copy_marker="$fixture/copy-complete"
    onedrive_main_state="$fixture/onedrive-main-running"
    onedrive_service_state="$fixture/onedrive-service-running"
    mkdir -p "$images" "$tmp" "$cloud" "$bin"
    : > "$log"
    install_fake_space_tools
    install_fake_onedrive_tools
}

trap '[[ -z "${fixture:-}" ]] || rm -rf "$fixture"' EXIT

install_fake_zip() {
    cat > "$bin/zip" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "ZIP_CALL $*" >> "$CALL_LOG"
archive="${@: -2:1}"
cycle_name="${@: -1}"
[[ -d "$cycle_name" ]]
printf 'fake zip for %s\n' "$cycle_name" > "$archive"
if [[ "${ZIP_FAIL:-0}" == 1 ]]; then
    printf 'interrupted\n' >> "$archive"
    exit 1
fi
EOF
    chmod +x "$bin/zip"
}

install_fake_rclone() {
    cat > "$bin/rclone" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
for proxy_name in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy RCLONE_HTTP_PROXY; do
    [[ -z "${!proxy_name+x}" ]] || {
        echo "proxy variable still set: $proxy_name" >&2
        exit 80
    }
done
[[ "${NO_PROXY:-}" == '*' ]] || { echo "NO_PROXY is not *" >&2; exit 80; }
[[ "${no_proxy:-}" == '*' ]] || { echo "no_proxy is not *" >&2; exit 80; }
[[ "${@: -2:1}" == --http-proxy && "${@: -1}" == "" ]] || {
    echo "missing empty --http-proxy flag" >&2
    exit 80
}
echo "rclone $*" >> "$CALL_LOG"
case "$1" in
    lsd)
        if [[ "${PREFLIGHT_FAIL:-0}" == 1 ]]; then
            exit 1
        fi
        [[ -d "$2" ]]
        ;;
    lsf)
        target="$2"
        if [[ "${STAT_FAIL_AFTER_COPY:-0}" == 1 && -f "$COPY_MARKER" ]]; then
            exit 1
        fi
        [[ -f "$target" ]]
        size="$(/usr/bin/stat -f '%z' "$target")"
        printf '%s\n' "$size"
        ;;
    cat)
        target="$2"
        [[ -f "$target" ]]
        if [[ "${READ_FAIL_BEFORE_COPY:-0}" == 1 && ! -f "$COPY_MARKER" ]]; then
            exit 1
        fi
        ;;
    copyto)
        if [[ "${UPLOAD_FAIL:-0}" == 1 ]]; then
            exit 1
        fi
        if [[ -f "$3" && " $* " == *" --size-only "* ]] \
            && [[ "$(/usr/bin/stat -f '%z' "$2")" == "$(/usr/bin/stat -f '%z' "$3")" ]]; then
            exit 0
        fi
        cp "$2" "$3"
        : > "$COPY_MARKER"
        ;;
    *)
        exit 2
        ;;
esac
EOF
    chmod +x "$bin/rclone"
}

install_fake_space_tools() {
    cat > "$bin/find" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ " $* " == *" -delete " ]]; then
    cycle_dir="$1"
    /usr/bin/find "$@"
    if [[ "${FINDER_RECREATES_DS_STORE:-0}" == 1 ]]; then
        mkdir -p "$cycle_dir"
        : > "$cycle_dir/.DS_Store"
    fi
else
    exec /usr/bin/find "$@"
fi
EOF
    chmod +x "$bin/find"

    cat > "$bin/df" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
index=0
if [[ -f "$DF_COUNTER_FILE" ]]; then
    index="$(<"$DF_COUNTER_FILE")"
fi
index=$((index + 1))
printf '%s\n' "$index" > "$DF_COUNTER_FILE"

chosen=""
position=0
for value in $AVAILABLE_KIB_SEQUENCE; do
    position=$((position + 1))
    chosen="$value"
    if (( position == index )); then
        break
    fi
done

printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n'
printf 'mock 100000000 0 %s 0%% /\n' "$chosen"
EOF
    chmod +x "$bin/df"

    cat > "$bin/sleep" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "SLEEP_CALL $*" >> "$CALL_LOG"
EOF
    chmod +x "$bin/sleep"
}

install_fake_onedrive_tools() {
    cat > "$bin/pgrep" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == -x && -f "$ONEDRIVE_MAIN_STATE" ]]; then
    exit 0
fi
if [[ "$1" == -f && -f "$ONEDRIVE_SERVICE_STATE" ]]; then
    exit 0
fi
exit 1
EOF
    chmod +x "$bin/pgrep"

    cat > "$bin/osascript" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "ONEDRIVE_QUIT" >> "$CALL_LOG"
rm -f "$ONEDRIVE_MAIN_STATE"
EOF
    chmod +x "$bin/osascript"

    cat > "$bin/pkill" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "ONEDRIVE_SERVICE_STOP" >> "$CALL_LOG"
rm -f "$ONEDRIVE_SERVICE_STATE"
EOF
    chmod +x "$bin/pkill"

    cat > "$bin/open" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "ONEDRIVE_RESTART" >> "$CALL_LOG"
: > "$ONEDRIVE_MAIN_STATE"
EOF
    chmod +x "$bin/open"
}

run_script() {
    HTTP_PROXY="http://127.0.0.1:7890" \
    HTTPS_PROXY="http://127.0.0.1:7890" \
    ALL_PROXY="http://127.0.0.1:7890" \
    http_proxy="http://127.0.0.1:7890" \
    https_proxy="http://127.0.0.1:7890" \
    all_proxy="http://127.0.0.1:7890" \
    RCLONE_HTTP_PROXY="http://127.0.0.1:7890" \
    NO_PROXY="inherited.example" \
    no_proxy="inherited.example" \
    CALL_LOG="$log" \
    COPY_MARKER="$copy_marker" \
    ONEDRIVE_MAIN_STATE="$onedrive_main_state" \
    ONEDRIVE_SERVICE_STATE="$onedrive_service_state" \
    PATH="$bin:/usr/bin:/bin" \
    IMAGES_ROOT="$images" \
    TMP_DIR="$tmp" \
    CLOUD_REMOTE="$cloud" \
    ZIP_BIN="$bin/zip" \
    RCLONE="$bin/rclone" \
    DF_BIN="$bin/df" \
    SLEEP_BIN="$bin/sleep" \
    DF_COUNTER_FILE="$fixture/df-counter" \
    AVAILABLE_KIB_SEQUENCE="${AVAILABLE_KIB_SEQUENCE:-104857600}" \
    FINDER_RECREATES_DS_STORE="${FINDER_RECREATES_DS_STORE:-0}" \
    bash "$SCRIPT" "$@"
}

test_running_onedrive_is_paused_and_restarted() {
    new_fixture
    install_fake_zip
    install_fake_rclone
    mkdir -p "$images/frost_cycle_000001"
    printf 'original\n' > "$images/frost_cycle_000001/frame.jpg"
    : > "$onedrive_main_state"
    : > "$onedrive_service_state"

    run_script > "$fixture/output.log"

    assert_contains "$log" "ONEDRIVE_QUIT"
    assert_contains "$log" "ONEDRIVE_SERVICE_STOP"
    assert_contains "$log" "ONEDRIVE_RESTART"
    assert_file_exists "$onedrive_main_state"
    assert_file_missing "$images/frost_cycle_000001"
}

test_matching_cloud_zip_is_verified_then_source_is_deleted() {
    new_fixture
    install_fake_zip
    install_fake_rclone
    mkdir -p "$images/frost_cycle_000002"
    printf 'original\n' > "$images/frost_cycle_000002/frame.jpg"
    printf 'fake zip for frost_cycle_000002\n' > "$cloud/frost_cycle_000002.zip"

    run_script > "$fixture/output.log"

    assert_contains "$fixture/output.log" "[VERIFY] OneDrive cloud ZIP confirmed"
    assert_file_missing "$images/frost_cycle_000002"
    assert_file_missing "$tmp/frost_cycle_000002.zip"
    assert_file_exists "$cloud/frost_cycle_000002.zip"
    [[ ! -f "$copy_marker" ]] || fail "matching readable cloud ZIP must not upload again"
    grep -Fq -- "--onedrive-chunk-size 100Mi" "$log" \
        || fail "OneDrive upload must use the tuned native chunk size"
    grep -Fq -- "--timeout 2m" "$log" \
        || fail "OneDrive upload must abandon and retry stalled network I/O"
    probe_count="$(grep -Fc "rclone cat" "$log")"
    [[ "$probe_count" == 2 ]] \
        || fail "strict verification must probe cloud head and tail (got $probe_count probes)"
}

test_same_size_but_unreadable_cloud_zip_is_reuploaded() {
    new_fixture
    install_fake_zip
    install_fake_rclone
    mkdir -p "$images/frost_cycle_000024"
    printf 'original\n' > "$images/frost_cycle_000024/frame.jpg"
    printf 'fake zip for frost_cycle_000024\n' > "$cloud/frost_cycle_000024.zip"

    READ_FAIL_BEFORE_COPY=1 run_script > "$fixture/output.log"

    [[ "$(grep -Fc "rclone copyto" "$log")" == 2 ]] \
        || fail "unreadable same-size ZIP must trigger one forced replacement"
    [[ -f "$copy_marker" ]] || fail "forced replacement must upload cloud content"
    assert_file_missing "$images/frost_cycle_000024"
    assert_file_missing "$tmp/frost_cycle_000024.zip"
}

test_different_size_cloud_zip_is_refused() {
    new_fixture
    install_fake_zip
    install_fake_rclone
    mkdir -p "$images/frost_cycle_000003"
    printf 'original\n' > "$images/frost_cycle_000003/frame.jpg"
    printf 'partial\n' > "$cloud/frost_cycle_000003.zip"

    if run_script > "$fixture/output.log" 2>&1; then
        fail "different-size cloud ZIP must stop for manual review"
    fi

    ! grep -Fq "rclone copyto" "$log" || fail "different-size cloud ZIP must not be replaced"
    assert_file_exists "$images/frost_cycle_000003/frame.jpg"
    assert_file_exists "$tmp/frost_cycle_000003.zip"
    [[ "$(/usr/bin/stat -f '%z' "$cloud/frost_cycle_000003.zip")" == 8 ]]
}

test_upload_failure_preserves_source_and_temp_zip() {
    new_fixture
    install_fake_zip
    install_fake_rclone
    mkdir -p "$images/frost_cycle_000004"
    printf 'original\n' > "$images/frost_cycle_000004/frame.jpg"

    if UPLOAD_FAIL=1 run_script > "$fixture/output.log" 2>&1; then
        fail "failed upload must return non-zero"
    fi

    assert_file_exists "$images/frost_cycle_000004/frame.jpg"
    assert_file_exists "$tmp/frost_cycle_000004.zip"
}

test_post_upload_cloud_verification_failure_preserves_source_and_temp() {
    new_fixture
    install_fake_zip
    install_fake_rclone
    mkdir -p "$images/frost_cycle_000005"
    printf 'original\n' > "$images/frost_cycle_000005/frame.jpg"

    if STAT_FAIL_AFTER_COPY=1 run_script > "$fixture/output.log" 2>&1; then
        fail "failed cloud verification must return non-zero"
    fi

    grep -Fq "rclone copyto" "$log" || fail "fixture must reach upload"
    assert_file_exists "$images/frost_cycle_000005/frame.jpg"
    assert_file_exists "$tmp/frost_cycle_000005.zip"
    assert_file_exists "$cloud/frost_cycle_000005.zip"
}

test_existing_temp_zip_is_reused() {
    new_fixture
    install_fake_zip
    install_fake_rclone
    mkdir -p "$images/frost_cycle_000006"
    printf 'original\n' > "$images/frost_cycle_000006/frame.jpg"
    printf 'existing complete zip\n' > "$tmp/frost_cycle_000006.zip"

    AVAILABLE_KIB_SEQUENCE="4 100" run_script > "$fixture/output.log"

    assert_contains "$fixture/output.log" "existing temporary ZIP found, reuse"
    ! grep -Fq "ZIP_CALL" "$log" || fail "existing temporary ZIP must not be repacked"
    ! grep -Fq "SLEEP_CALL" "$log" \
        || fail "reusing a complete ZIP needs no additional archive space"
    assert_file_missing "$images/frost_cycle_000006"
    assert_file_missing "$tmp/frost_cycle_000006.zip"
}

test_failed_pack_is_rebuilt_on_next_run() {
    new_fixture
    install_fake_zip
    install_fake_rclone
    mkdir -p "$images/frost_cycle_000007"
    printf 'original\n' > "$images/frost_cycle_000007/frame.jpg"

    if ZIP_FAIL=1 run_script > "$fixture/first.log" 2>&1; then
        fail "failed pack must return non-zero"
    fi
    run_script > "$fixture/second.log"

    [[ "$(grep -Fc "ZIP_CALL" "$log")" == 2 ]] \
        || fail "interrupted pack must be rebuilt"
    assert_file_missing "$images/frost_cycle_000007"
}

test_cloud_preflight_failure_changes_nothing() {
    new_fixture
    install_fake_zip
    install_fake_rclone
    mkdir -p "$images/frost_cycle_000008"
    printf 'original\n' > "$images/frost_cycle_000008/frame.jpg"

    if PREFLIGHT_FAIL=1 run_script > "$fixture/output.log" 2>&1; then
        fail "cloud preflight failure must return non-zero"
    fi

    assert_file_exists "$images/frost_cycle_000008/frame.jpg"
    assert_file_missing "$tmp/frost_cycle_000008.zip"
    ! grep -Fq "ZIP_CALL" "$log" || fail "preflight must fail before packaging"
}

test_finder_ds_store_recreation_does_not_stop_cleanup() {
    new_fixture
    install_fake_zip
    install_fake_rclone
    mkdir -p "$images/frost_cycle_000009"
    printf 'original\n' > "$images/frost_cycle_000009/frame.jpg"

    FINDER_RECREATES_DS_STORE=1 run_script > "$fixture/output.log"

    assert_file_missing "$images/frost_cycle_000009"
}

test_low_space_waits_then_completes_two_cycles() {
    new_fixture
    install_fake_zip
    install_fake_rclone
    mkdir -p "$images/frost_cycle_000009" "$images/frost_cycle_000010"
    dd if=/dev/zero of="$images/frost_cycle_000009/frame.jpg" bs=1024 count=32 2>/dev/null
    dd if=/dev/zero of="$images/frost_cycle_000010/frame.jpg" bs=1024 count=32 2>/dev/null

    AVAILABLE_KIB_SEQUENCE="100 20 100" run_script > "$fixture/output.log"

    assert_contains "$fixture/output.log" "[WAIT] insufficient free space"
    [[ "$(grep -Fc "ZIP_CALL" "$log")" == 2 ]] || fail "both cycles must complete"
    assert_file_missing "$images/frost_cycle_000009"
    assert_file_missing "$images/frost_cycle_000010"
}

test_running_onedrive_is_paused_and_restarted
test_matching_cloud_zip_is_verified_then_source_is_deleted
test_same_size_but_unreadable_cloud_zip_is_reuploaded
test_different_size_cloud_zip_is_refused
test_upload_failure_preserves_source_and_temp_zip
test_post_upload_cloud_verification_failure_preserves_source_and_temp
test_existing_temp_zip_is_reused
test_failed_pack_is_rebuilt_on_next_run
test_cloud_preflight_failure_changes_nothing
test_finder_ds_store_recreation_does_not_stop_cleanup
test_low_space_waits_then_completes_two_cycles

echo "PASS: upload_cycle_images"
