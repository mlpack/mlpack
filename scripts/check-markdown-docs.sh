#!/usr/bin/env bash
#
# This script checks to ensure that generated Markdown documentation is the same
# as what's committed to the repository.  After configuring mlpack with
# -DBUILD_MARKDOWN_BINDINGS=ON, use this simple script to detect changes in the
# documentation that should be committed.
#
# The only argument of the script is to the build directory. We use a default
# value of 'build' in the current directory as is common with 'cmake'.
build_dir=${1:-"build"}

# If the directory does not exist, complain and exit.
if [[ ! -d ${build_dir} ]]; then
  echo "Usage: $0 build/";
  echo "  (replace build/ with your build directory where you already"
  echo "   ran 'make markdown')";
  exit 1;
fi

# Check that Markdown documentation has been built.
if [[ ! -d "$build_dir/doc/" ]];
then
  echo "$build_dir/doc/ does not exist!";
  echo "Did you run 'make markdown' in your build directory ($build_dir)?";
  exit 1;
fi

# Overall state, default to no issues.
good=0

# Now check every file in the main repository bindings.
for f in doc/user/bindings/*;
do
  f_base=`basename $f`;

  if [[ ! -f "$build_dir/doc/$f_base" ]];
  then
    echo "$build_dir/doc/$f_base does not exist!";
    echo "Did you run 'make markdown' in your build directory ($build_dir)?";
    echo "Or does the file need to be removed from the repository?";
    exit 1;
  fi

  diff -Naq $f "$build_dir/doc/$f_base";

  if [[ "$?" -ne 0 ]];
  then
    echo -n "   *** Files $f and $build_dir/doc/$f_base differ! Run 'diff' "
    echo    "manually to examine";
    good=1
  fi
done

# If issue found, exit with error code.
if [[ ${good} -eq 1 ]]; then
    exit 1
fi
