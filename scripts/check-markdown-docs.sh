#!/usr/bin/env bash
#
# This script checks to ensure that generated Markdown documentation is the same
# as what's committed to the repository.  After configuring mlpack with
# -DBUILD_MARKDOWN_BINDINGS=ON, use this simple script to detect changes in the
# documentation that should be committed.
#
# The only argument of the script is to the build directory.
if [ "$#" -ne 1 ];
then
  echo "Usage: $0 build/";
  echo "  (replace build/ with your build directory, where you already"
  echo "   ran 'make markdown')";
  exit 1;
fi

build_dir="$1";

# Check that Markdown documentation has been built.
if [[ ! -d "$build_dir/doc/" ]];
then
  echo "$build_dir/doc/ does not exist!";
  echo "Did you run 'make markdown' in your build directory ($build_dir)?";
  exit 1;
fi

# Now check every file in the main repository bindings.
for f in doc/user/bindings/*;
do
  echo "Checking $f...";
  f_base=`basename $f`;

  if [[ ! -f "$build_dir/doc/$f_base" ]];
  then
    echo "$build_dir/doc/$f_base does not exist!";
    echo "Did you run 'make markdown' in your build directory ($build_dir)?";
    echo "Or does the file need to be removed from the repository?";
    exit 1;
  fi

  diff -Nau $f "$build_dir/doc/$f_base";

  if [ "$?" -ne 0 ];
  then
    echo "";
    echo "";
    echo "Files $f and $build_dir/doc/$f_base differ!  (See above.)";
    echo "";
    echo "If the sidebar differs, be sure to check if updates are needed in ";
    echo "quickstart/*.sidebar.html!";
    exit 1;
  fi
done
