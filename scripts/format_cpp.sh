#!/usr/bin/env bash

set -e

function usage() {
    cat <<EOF
This script applies or checks the defined formatting to the codebase.
It requires 'clang-format' to be available in the PATH.

Usage:
    $(basename ${0}) [options]

Options:
    -h, --help         Print this message and exit.
    -c, --check        Check if codebase is formatted.
    -p, --patch [file] Creates patch file. This requires no unstaged changes seen by git.
                       And 'git' has to be available in the PATH.
EOF
}

MLPACK_ROOT_DIR="$(dirname $(dirname $(readlink -f "$0")))"

check=false
patch_file=""

while [ -n "$1" ]
do
    case "$1" in
        (-h|--help)
            usage
            exit 0
            ;;
        (-c|--check)
            check=true
            shift
            ;;
        (-p|--patch)
            patch_file="$2"
            shift 2
            ;;
        (*)
            echo "ERROR: unknown argument $1"
            usage
            exit 1
            ;;
    esac
done

function check_no_git_diff() {
    cd "${MLPACK_ROOT_DIR}"
    set +e
    git diff --exit-code --no-patch
    if [ $? -ne 0 ]; then
        echo "ERROR: detected unstaged changes, abort."
        exit 1
    fi
    set -e
    cd - > /dev/null
}

if [ -n "${patch_file}" ] ; then
    check_no_git_diff
fi

if [ "${check}" = true ] && [ -z "${patch_file}" ] ; then
    clang_format_args=("--dry-run" "--Werror")
else
    clang_format_args=("-i")
fi

# Run formatting
find "${MLPACK_ROOT_DIR}" \
    -name "*[ch]pp" \
    ! -path "*third_party*" \
    ! -name "catch.hpp" \
    -exec clang-format --style=file "${clang_format_args[@]}" {} +

if [ -n "${patch_file}" ] ; then
    cd "${MLPACK_ROOT_DIR}"

    if [ "${check}" = true ] ; then
        set +e
        git diff --exit-code --no-patch
        is_formatted=$?
        set -e
    fi

    git diff > "${patch_file}"
    git restore .
    cd - > /dev/null

    if [ "${check}" = true ] ; then
        exit "${is_formatted}"
    fi
fi
