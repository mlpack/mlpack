#!/bin/sh

set -e

if [ $# != 2 ]; then
    echo "Convert all of the executables in this directory that are not tests to man"
    echo "pages in the given directory."
    echo
    echo "Usage:"
    echo "  allexec2man.sh /full/path/of/exec2man.sh output_directory/"
    echo
    echo "For the executable 'cheese', the file 'cheese.1.gz' will be created in the"
    echo "output directory."
    exit 1
fi

exec2man="$1"
outdir="$2"

mkdir -p "$outdir"
for program in $(find . -type f -perm -u+x -iname 'mlpack_*' | \
                grep -v '[.]$' | \
                grep -v '_test$'); do
  echo "Generating man page for $program...";
  "$exec2man" "$program" "$outdir/$program.1"
done
