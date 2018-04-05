#!/bin/bash
# Convert the output of an mlpack executable into a man page.  This assumes that
# the CLI subsystem is used to output help, that the executable is properly
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
name="$1"
output="$2"

# Generate the synopsis.
# First, required options.
reqoptions=`./"$name" -h | \
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
  sed 's/\]//g'`

# Then, regular options.
options=`./"$name" -h | \
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
  sed 's/\(--[A-Za-z0-9_-]* [a-z]*\) /\[\1\] /g'`

synopsis="$name $reqoptions $options [-h -v]";

# Preview the whole thing first.
#./$name -h | \
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
./"$name" -h | \
  sed 's/^For further information/Additional Information\n\n For further information/' | \
  sed 's/^consult the documentation/ consult the documentation/' | \
  sed 's/^distribution of mlpack./ distribution of mlpack./' | \
  awk -v syn="$synopsis" \
      '{ if (NR == 1) print "NAME\n '"$name"' - "tolower($0)"\nSYNOPSIS\n "syn" \nDESCRIPTION\n" ; else print } ' | \
  sed '/^[^ ]/ y/qwertyuiopasdfghjklzxcvbnm:/QWERTYUIOPASDFGHJKLZXCVBNM /' | \
  sed 's/  / /g' | \
  awk '/NAME/,/.*OPTIONS/ { if (!/.*OPTIONS/) { print; } } /ADDITIONAL INFORMATION/,0 { print; } /.*OPTIONS/,/ADDITIONAL INFORMATION/ { if (!/REQUIRED INPUT OPTIONS/ && !/OPTIONAL INPUT OPTIONS/ && !/OPTIONAL OUTPUT OPTIONS/ && !/ADDITIONAL INFORMATION/) { if (/ --/) { printf "\n" } sub(/^[ ]*/, ""); sub(/ [ ]*/, " "); printf "%s ", $0; } else { if (!/ADDITIONAL INFORMATION/) { print "\n"$0; } } }' | \
  sed 's/  ADDITIONAL INFORMATION/\n\nADDITIONAL INFORMATION/' | \
  txt2man -t "$name" -s 1 -r "MLPACK Utilities" -v "User Commands" | \
  sed "s/^'/\\\\(aq/" > "$output"

