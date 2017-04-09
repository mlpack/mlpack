#!/bin/bash -e

numFailures=0
fail=0
if [ "$TRAVIS_EVENT_TYPE" = "pull_request" ]; then
  # Get list of modified source code files in the pull request
  COMMIT_FILES="$(git diff --name-only --diff-filter=ACMRTUXB $TRAVIS_COMMIT_RANGE | grep '^src/[^.]*[.]\(hpp\|cpp\)$' | true)"
else
   # Get list of all source code files
  COMMIT_FILES="$(find ./src/mlpack/ -type f ! -path "./src/mlpack/core/arma_extend/*" ! -path "./src/mlpack/core/boost_backport/*")"
fi

# Get list of files on which style check is not applied
EXCLUDED_FILES=$(find ./src/mlpack/core/arma_extend/* ./src/mlpack/core/boost_backport/* -name '*.hpp' -o -name '*.cpp')

for f in ${COMMIT_FILES}; do
  # Check difference between clang-format output and commit file
  checkDiff=$(diff -u "$f" <(clang-format "$f") || true)
  if ! [ -z "$checkDiff" ]; then

    # Check if file is excluded from clang-format style check or not
    checkExcluded=$(awk '$1 == "'$f'" { print 1 }' "$EXCLUDED_FILES")

    # If not, mark as failure
    if [ -z ${checkExcluded} ]; then  
      numFailures=$((numFailures + 1))
      printf "The file %s is not compliant with the coding style" "$f"
      if [ ${numFailures} -gt 50 ]; then
        printf "\nToo many style errors encountered previously, this diff is hidden.\n"
      else
        printf ":\n%s\n" "$checkDiff"
      fi
      fail=1
    fi
  fi
done

fi
if [ "$fail" = 1 ]; then
  echo "Style check failed."
  exit 1
fi
echo "Style check passed."
