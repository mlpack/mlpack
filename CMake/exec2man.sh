#!/bin/sh
#
# Convert the output of an mlpack executable into a man page.  This assumes that
# the IO subsystem is used to output help, that the executable is properly
# documented, and that the program is run in the directory that the executable
# is in.  Usually, this is used by CMake on Linux/UNIX systems to generate the
# man pages.
#
# Usage:
#  exec2man.sh executable_name output_file_name
#
# No warranties...
#
# @author Ryan Curtin

set -e

if [ $# != 2 ]; then
    echo "Generates man page from the help text of an mlpack utility program."
    echo "Usage: $0 mlpack_executable generated-man-page.1"
    exit 1
fi

exec="$1"
name="$(basename "$exec")"
output="$2"

if [ "$name" = "$exec" ]; then
    # if no directory prefix with explict ./ to avoid path search
    exec="./$exec"
fi

if [ ! -x "$exec" ]; then
   echo "error: cannot find executable file $exec"
   exit 1
fi

# Get the version.
version=$("$exec" --version | sed 's/^.* \([^ ]*\)\.$/\1/')

# Generate the synopsis.
# First, required options.
reqoptions="$("$exec" --help | \
  awk '/Required input options:/,/Optional input options:/' | \
  grep '^  --' | \
  sed 's/^  --/--/' | \
  sed 's/^--[A-Za-z0-9_-]* (\(-[A-Za-z0-9]\))/\1/' | \
  sed 's/\(^-[A-Za-z0-9]\) [^\[].*/\1/' | \
  sed 's/\(^-[A-Za-z0-9] \[[A-Za-z0-9]*\]\) .*/\1/' | \
  sed 's/\(^--[A-Za-z0-9_-]*\) [^[].*/\1/' | \
  sed 's/\(^--[A-Za-z0-9_-]* \[[A-Za-z0-9]*\]\) [^[].*/\1/' | \
  tr '\n' ' ' | \
  sed 's/\[//g' | \
  sed 's/\]//g')"

# Then, regular options.
options="$("$exec" -h | \
  awk '/Optional input options:/,/For further information,/' | \
  grep '^  --' | \
  sed 's/^  --/--/' | \
  grep -v -- '--help' | \
  grep -v -- '--info' | \
  grep -v -- '--verbose' | \
  sed 's/^--[A-Za-z0-9_-]* (\(-[A-Za-z0-9]\))/\1/' | \
  sed 's/\(^-[A-Za-z0-9]\) [^\[].*/\1/' | \
  sed 's/\(^-[A-Za-z0-9] \[[A-Za-z0-9]*\]\) .*/\1/' | \
  sed 's/\(^--[A-Za-z0-9_-]*\) [^[].*/\1/' | \
  sed 's/\(^--[A-Za-z0-9_-]* \[[A-Za-z0-9]*\]\) [^[].*/\1/' | \
  tr '\n' ' ' | \
  sed 's/\[//g' | \
  sed 's/\]//g' | \
  sed 's/\(-[A-Za-z0-9]\)\( [^a-z]\)/\[\1\]\2/g' | \
  sed 's/\(--[A-Za-z0-9_-]*\)\( [^a-z]\)/\[\1\]\2/g' | \
  sed 's/\(-[A-Za-z0-9] [a-z]*\) /\[\1\] /g' | \
  sed 's/\(--[A-Za-z0-9_-]* [a-z]*\) /\[\1\] /g')"

synopsis="$name $reqoptions $options [-h -v]";

# Preview the whole thing first.
#"$exec" -h | \
#  awk -v syn="$synopsis" \
#      '{ if (NR == 1) print "NAME\n '$name' - "tolower($0)"\nSYNOPSIS\n "syn" \nDESCRIPTION\n" ; else print } ' | \
#  sed '/^[^ ]/ y/qwertyuiopasdfghjklzxcvbnm:/QWERTYUIOPASDFGHJKLZXCVBNM /' | \
#  txt2man -T -P mlpack -t $name -d 1

# Now do it.
# The awk script is a little ugly, but it is meant to format parameters
# correctly so that the entire description of the parameter is on one line (this
# helps avoid 'man' warnings).
# The sed line at the end removes accidental macros from the output, replacing
# single-quotes at the beginning of a line with the troff escape code \(aq.
"$exec" -h | \
  sed 's/^For further information/Additional Information\n\n For further information/' | \
  sed 's/^consult the documentation/ consult the documentation/' | \
  sed 's/^distribution of mlpack./ distribution of mlpack./' | \
  awk -v syn="$synopsis" \
      '{ if (NR == 1) print "NAME\n '"$name"' - "tolower($0)"\nSYNOPSIS\n "syn" \nDESCRIPTION\n" ; else print } ' | \
  sed '/^[^ ]/ y/qwertyuiopasdfghjklzxcvbnm:/QWERTYUIOPASDFGHJKLZXCVBNM /' | \
  sed 's/  / /g' | \
  awk '/NAME/,/.*OPTIONS/ { if (!/.*OPTIONS/) { print; } } /ADDITIONAL INFORMATION/,0 { print; } /.*OPTIONS/,/ADDITIONAL INFORMATION/ { if (!/REQUIRED INPUT OPTIONS/ && !/OPTIONAL INPUT OPTIONS/ && !/OPTIONAL OUTPUT OPTIONS/ && !/ADDITIONAL INFORMATION/) { if (/ --/) { printf "\n" } sub(/^[ ]*/, ""); sub(/ [ ]*/, " "); printf "%s ", $0; } else { if (!/ADDITIONAL INFORMATION/) { print "\n"$0; } } }' | \
  sed 's/  ADDITIONAL INFORMATION/\n\nADDITIONAL INFORMATION/' | \
  txt2man -t "$name" -s 1 -r "mlpack-$version" -v "User Commands" | \
  sed "s/^'/\\\\(aq/" > "$output"

