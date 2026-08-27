#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$PROJECT_ROOT/scripts/data/upload_date_data.sh"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_file_exists() {
    [[ -f "$1" ]] || fail "expected file: $1"
}

assert_missing() {
    [[ ! -e "$1" ]] || fail "expected missing path: $1"
}

new_fixture() {
    [[ -z "${fixture:-}" || ! -d "$fixture" ]] || rm -rf "$fixture"
    fixture="$(mktemp -d)"
    data_root="$fixture/data"
    tmp_dir="$data_root/.date_archive_tmp"
    cloud="$fixture/cloud"
    bin="$fixture/bin"
    calls="$fixture/calls.log"
    copy_marker="$fixture/copied"
    mkdir -p "$data_root" "$cloud" "$bin"
    : > "$calls"
    install_fakes
}

trap '[[ -z "${fixture:-}" ]] || rm -rf "$fixture"' EXIT

install_fakes() {
    cat > "$bin/zip" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "ZIP_CALL $*" >> "$CALLS"
archive="${@: -2:1}"
date_name="${@: -1}"
[[ -d "$date_name" ]]
printf 'archive for %s\n' "$date_name" > "$archive"
EOF
    chmod +x "$bin/zip"

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
echo "RCLONE_CALL $*" >> "$CALLS"
case "$1" in
    mkdir)
        mkdir -p "$2"
        ;;
    lsf)
        [[ -f "$2" ]]
        /usr/bin/stat -f '%z' "$2"
        ;;
    cat)
        [[ -f "$2" ]]
        [[ "${READ_FAIL:-0}" != 1 ]]
        ;;
    copyto)
        if [[ "${UPLOAD_FAIL:-0}" == 1 ]]; then
            exit 9
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

    cat > "$bin/df" <<'EOF'
#!/usr/bin/env bash
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n'
printf 'mock 100000000 0 100000000 0%% /\n'
EOF
    chmod +x "$bin/df"

    cat > "$bin/pgrep" <<'EOF'
#!/usr/bin/env bash
if [[ "$1" == -x && "$2" == rclone && "${ACTIVE_RCLONE:-0}" == 1 ]]; then
    exit 0
fi
exit 1
EOF
    chmod +x "$bin/pgrep"
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
    CALLS="$calls" \
    COPY_MARKER="$copy_marker" \
    PATH="$bin:/usr/bin:/bin" \
    DATA_ROOT="$data_root" \
    TMP_DIR="$tmp_dir" \
    CLOUD_REMOTE="$cloud" \
    ZIP_BIN="$bin/zip" \
    RCLONE="$bin/rclone" \
    DF_BIN="$bin/df" \
    UPLOAD_FAIL="${UPLOAD_FAIL:-0}" \
    READ_FAIL="${READ_FAIL:-0}" \
    ACTIVE_RCLONE="${ACTIVE_RCLONE:-0}" \
    bash "$SCRIPT" "$@"
}

test_generated_zip_is_deleted_but_source_directory_remains() {
    new_fixture
    mkdir -p "$data_root/0724"
    printf 'raw data\n' > "$data_root/0724/file.edf"

    run_script > "$fixture/output.log"

    assert_file_exists "$data_root/0724/file.edf"
    assert_file_exists "$cloud/0724.zip"
    assert_missing "$tmp_dir/0724.zip"
    grep -Fq 'ZIP_CALL' "$calls" || fail "date directory must be packaged"
}

test_existing_source_zip_is_uploaded_and_preserved() {
    new_fixture
    printf 'prepacked source\n' > "$data_root/0723.zip"

    run_script > "$fixture/output.log"

    assert_file_exists "$data_root/0723.zip"
    assert_file_exists "$cloud/0723.zip"
    ! grep -Fq 'ZIP_CALL' "$calls" || fail "existing source ZIP must not be repacked"
}

test_existing_source_zip_wins_when_directory_also_exists() {
    new_fixture
    mkdir -p "$data_root/0727"
    printf 'raw data\n' > "$data_root/0727/file.edf"
    printf 'prepacked source\n' > "$data_root/0727.zip"

    run_script > "$fixture/output.log"

    assert_file_exists "$data_root/0727/file.edf"
    assert_file_exists "$data_root/0727.zip"
    assert_file_exists "$cloud/0727.zip"
    ! grep -Fq 'ZIP_CALL' "$calls" || fail "source ZIP must take precedence"
}

test_failed_upload_preserves_directory_and_generated_zip() {
    new_fixture
    mkdir -p "$data_root/0728"
    printf 'raw data\n' > "$data_root/0728/file.edf"

    set +e
    UPLOAD_FAIL=1 run_script > "$fixture/output.log" 2>&1
    status=$?
    set -e
    (( status != 0 )) || fail "upload failure must return non-zero"

    assert_file_exists "$data_root/0728/file.edf"
    assert_file_exists "$tmp_dir/0728.zip"
}

test_matching_cloud_zip_skips_transfer_and_preserves_source_zip() {
    new_fixture
    printf 'prepacked source\n' > "$data_root/0729.zip"
    cp "$data_root/0729.zip" "$cloud/0729.zip"

    run_script > "$fixture/output.log"

    assert_file_exists "$data_root/0729.zip"
    [[ ! -e "$copy_marker" ]] || fail "matching cloud ZIP must not transfer"
    [[ "$(grep -Fc 'RCLONE_CALL cat' "$calls")" == 2 ]] \
        || fail "cloud ZIP must be checked at both edges"
}

test_force_allows_starting_while_another_rclone_runs() {
    new_fixture
    printf 'prepacked source\n' > "$data_root/0730.zip"

    set +e
    ACTIVE_RCLONE=1 run_script > "$fixture/blocked.log" 2>&1
    status=$?
    set -e
    (( status != 0 )) || fail "active rclone must block the default run"

    ACTIVE_RCLONE=1 run_script --force > "$fixture/forced.log"

    assert_file_exists "$data_root/0730.zip"
    assert_file_exists "$cloud/0730.zip"
}

test_generated_zip_is_deleted_but_source_directory_remains
test_existing_source_zip_is_uploaded_and_preserved
test_existing_source_zip_wins_when_directory_also_exists
test_failed_upload_preserves_directory_and_generated_zip
test_matching_cloud_zip_skips_transfer_and_preserves_source_zip
test_force_allows_starting_while_another_rclone_runs

echo "PASS: upload_date_data"
