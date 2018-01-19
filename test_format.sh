#!/bin/bash

NAME=`basename "$0"`

if [ ! -f ".clang-format" ]; then
    echo ".clang-format file not found!"
    exit 1
fi

CLANG_FORMAT="clang-format"

which "clang-format-4.0" > /dev/null && CLANG_FORMAT="clang-format-4.0"

FILES=`git ls-files | grep -E "\.(cpp|h|hpp|c)$" | grep -Ev "doc/" | grep -Ev "CMake/" | grep -Ev "src/mlpack/bindings/matlab/" | grep -Ev "src/mlpack/core/arma_extend/" | grep -Ev "src/mlpack/core/boost_backport/" | grep -Ev "src/mlpack/core/core.hpp" | grep -Ev "src/mlpack/methods/ann/visitor/" | grep -Ev "src/mlpack/prereqs.hpp" | grep -Ev "src/mlpack/core.hpp" | grep -Ev ".travis/config.hpp"`

for FILE in $FILES; do
    if [ "$NAME" != "pre-commit" ]; then
        # if this is not a pre-commit hook format code inplace
        $CLANG_FORMAT -i $FILE
    else
        staged_file=`git show :$FILE`
        formatted_file=`cat << EOF | $CLANG_FORMAT
$staged_file
EOF`
        if [ "$staged_file" != "$formatted_file" ]; then
            actual_file=`cat $FILE`
            if [ "$actual_file" != "$staged_file" ]; then
                echo "WARNING: $FILE is not formatted properly. Cannot fix formatting as there are unstaged changes"
            else
                echo "Fixing formatting of $FILE automatically"
                $CLANG_FORMAT -i $FILE
                git add $FILE
            fi
        fi
    fi
done