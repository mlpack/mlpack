#!/bin/bash
#
# Convert all of the executables in this directory that are not tests to man
# pages in the given directory.
#
# Usage:
#   allexec2man.sh /full/path/of/exec2man.sh output_directory/
#
# For the executable 'cheese', the file 'cheese.1.gz' will be created in the
# output directory.
exec2man="$1"
outdir="$2"

mkdir -p "$outdir"
for program in `find . -perm /u=x,g=x,o=x | \
                grep -v '[.]$' | \
                grep -v '_test$' | \
                sed 's|^./||'`; do
  echo "Generating man page for $program...";
  "$1" "$program" "$outdir/$program.1"
  gzip -f "$outdir/$program.1"
done
