#!/usr/bin/env bash
#
# Extract C++ code blocks from either an individual Markdown file or a directory
# full of Markdown files.  This does roughly what you would expect it to, but
# there is a little bit of magic:
#
#  * All ```c++ code blocks are extracted into their own .cpp files.
#
#  * mlpack.hpp is included in each file, and each code block is placed in an
#      `int main() { }` block.
#
#  * If Eigen or xtensor is detected, correct includes are added.
#
#  * All data referenced in http URLs is downloaded.
#
#  * If a code block is just a class declaration, it will be added to the *next*
#      code block file.
#
# Once all code blocks are compiled, they are run just to make sure they run
# correctly.  If a single file was passed, all inputs and outputs are printed;
# if an entire directory is given, program output is only printed on error.

if [[ $# -ne 1 ]];
then
  echo "Usages: " >&2;
  echo " - $0 input_file.md" >&2;
  echo " - $0 markdown_directory/" >&2;
  exit 2;
fi

if [ -z "$CXX" ];
then
  echo "You must set \$CXX to the compiler you want to use!" >&2;
  exit 1;
fi

if [ -z "$CXXFLAGS" ];
then
  echo "Warning: \$CXXFLAGS is unset.  If ensmallen, Armadillo, STB, or cereal ";
  echo "  are not in standard locations, builds will fail!  Be sure to use ";
  echo "  absolute paths, not relative paths.";
fi

if [ -z "$LDFLAGS" ];
then
  echo "Warning: \$LDFLAGS is unset.  If libarmadillo.so is not in a standard ";
  echo "  location, builds will fail!";
fi

# First determine what files we are looking at.
if [ -d $1 ];
then
  files=`find $1 -iname '*.md'`;
  mode="directory";
else
  files=$1;
  mode="file";
fi

# Extract the C++ code blocks from a particular file, creating
# $output_prefix1.cpp, $output_prefix2.cpp, and so on and so forth.
#
# The code in those snippets will be placed into an int main() { } block, and
# mlpack.hpp will be included.
extract_code_blocks()
{
  input_file=$1;
  output_prefix=$2;

  # Extract into temporary files.
  sed -n '/^```c++/,/^```/ p' < $input_file > $input_file.tmp;
  output_file_id=0;
  output_file_display="00"; # Hopefully no file has more than 100 examples...

  # Track whether or not the last line was a fence, since we get them two at a
  # time.  We initially set this to 1, because the first fence does not have a
  # preceding fence close above it.
  last_line_fence=1;

  # Track whether or not the entire last file corresponded to a class
  # declaration.
  class_decl=0;

  while IFS= read -r line;
  do
    if [[ $last_line_fence == 1 ]];
    then
      # Skip this line---it will be a fence opening.
      last_line_fence=0;

      # Create main() function to wrap the code in.
      echo "#include <mlpack.hpp>" > $output_prefix$output_file_display.cpp;
      echo "" >> $output_prefix$output_file_display.cpp;

      # If we have a class declaration from the previous file, insert it.
      if [[ $class_decl == 1 ]];
      then
        class_decl=0;
        last_output_file_id=$(($output_file_id - 1));
        last_output_file_display=$(printf "%02d" $last_output_file_id);

        cat $output_prefix$last_output_file_display.cpp | awk '
            BEGIN { p=0 }
            /int main()/ { p=1 }
            /^{/ { if(p == 1) { p=2; o=1 } }
            /^}/ { p=0; }
            // { if (p == 2 && o == 0) { print substr($0, 3) } o=0 }' >> $output_prefix$output_file_display.cpp;
        echo "" >> $output_prefix$output_file_display.cpp;
        rm -f $output_prefix$last_output_file_display.cpp;
      fi

      echo "int main()" >> $output_prefix$output_file_display.cpp;
      echo "{" >> $output_prefix$output_file_display.cpp;
      continue;
    fi

    if [[ $line == '```'* ]];
    then
      last_line_fence=1;

      # Close main() function.
      echo "}" >> $output_prefix$output_file_display.cpp;

      # Check after the fact: was this file only a class declaration?  If so, we
      # want to put it instead into the next file.
      has_class1=`grep '^  class' $output_prefix$output_file_display.cpp | wc -l`;
      has_class2=`grep '^  };' $output_prefix$output_file_display.cpp | wc -l`;
      if [[ "$has_class1" != "0" && "$has_class2" != "0" ]];
      then
        class_decl=1;
      fi;

      # Detect if we need any to add any special headers.  We have to do this
      # when we finish with the file...
      if [[ `grep 'Eigen::' $output_prefix$output_file_display.cpp | wc -l` -gt 0 ]];
      then
        sed -i '1s/^/#include <Eigen\/Dense>\n/' $output_prefix$output_file_display.cpp;
      fi

      if [[ `grep 'xt::' $output_prefix$output_file_display.cpp | wc -l` -gt 0 ]];
      then
        sed -i '1s/^/#include <xtensor\/xrandom.hpp>\n/' $output_prefix$output_file_display.cpp;
        sed -i '1s/^/#include <xtensor\/xarray.hpp>\n/' $output_prefix$output_file_display.cpp;
      fi

      output_file_id=$(($output_file_id + 1));
      output_file_display=$(printf "%02d" $output_file_id);

      continue;
    fi

    # Include indentation (two spaces).
    echo "  $line" >> $output_prefix$output_file_display.cpp;
  done < $input_file.tmp;

  # The last file is always invalid---we opened it without knowing whether
  # anything would be in it.
  rm -f $output_prefix$output_file_display.cpp;

  # Check the "true" last file: if it's only class declarations, no need to
  # compile it.
  output_file_id=$(($output_file_id - 1));
  output_file_display=$(printf "%02d" $output_file_id);
  if [ -f $output_prefix$output_file_display.cpp ];
  then
    cat $output_prefix$output_file_display.cpp | awk '
        BEGIN { p=0 }
        /int main()/ { p=1 }
        /^{/ { if(p == 1) { p=2; o=1 } }
        /^}/ { p=0 }
        // { if (p == 2 && o == 0) { print substr($0, 3) } o=0 }' >> $output_prefix$output_file_display.cpp.tmp;
    has_class1=`grep '^class' $output_prefix$output_file_display.cpp.tmp | wc -l`;
    has_class2=`grep '^};' $output_prefix$output_file_display.cpp.tmp | wc -l`;
    if [[ "$has_class1" != "0" && "$has_class2" != "0" ]];
    then
      # The file's main() function is just a class declaration.  Nuke it.
      rm -f $output_prefix$output_file_display.cpp;
    fi
    rm -f $output_prefix$output_file_display.cpp.tmp;
  fi

  rm -f $input_file.tmp;
}

compile_code_blocks()
{
  input_dir=$1;

  # If there are no files to compile, leave early.
  if ! compgen -G $input_dir/*.cpp >/dev/null;
  then
    return;
  fi

  for f in $input_dir/*.cpp;
  do
    echo "  Compiling $f...";
    of=${f%.cpp};

    if ! $CXX -std=c++17 -Isrc/ $CXXFLAGS -o $of $f $LDFLAGS -larmadillo 2>$of.tmp;
    then
      echo "Compilation of the following program failed:";
      echo "";
      cat $f;
      echo "";
      echo "First ten lines of error output:";
      head $of.tmp;
      echo "";
      echo "For full error output run either:";
      echo " - less $of.tmp";
      echo " - $CXX -std=c++17 -Isrc/ $CXXFLAGS -o $of $f $LDFLAGS -larmadillo";
      echo "";
      echo "Did you set \$CXX, \$CXXFLAGS, and \$LDFLAGS correctly?"
      exit 1;
    fi
  done
}

download_http_artifacts()
{
  input_dir=$1;
  output_dir=$2;

  # Get a list of all HTTP resources.
  artifacts=`grep 'http[s]*://' $input_dir/*.cpp |\
      sed 's/^.*\(http[^ ]*\).*$/\1/' |\
      sort |\
      uniq |\
      grep 'csv\|arff\|bin\|png' |\
      sed 's/\.$//'`;
  cd $output_dir;
  for a in $artifacts;
  do
    out_a=`basename $a`;
    if [ ! -f $out_a ];
    then
      echo "  Downloading $a...";
      if ! curl -s -O $a;
      then
        echo "Error downloading $a!";
        exit 1;
      fi
    fi
  done
  cd - >/dev/null;

  # Special case: if we are looking at core.md, this has two special files we
  # need to create that is used in the example.
  f=`basename $input_dir`;
  if [[ "$f" == "core" || "$f" == "matrices" ]];
  then
    cd $output_dir;
    echo "  Creating data.csv...";
    cat > data.csv << EOF
3,3,3,3,0
3,4,4,3,0
3,4,4,3,0
3,3,4,3,0
3,6,4,3,0
2,4,4,3,0
2,4,4,1,0
3,3,3,2,0
3,4,4,2,0
3,4,4,2,0
3,3,4,2,0
3,6,4,2,0
2,4,4,2,0
EOF

    echo "  Creating mixed_string_data.csv...";
    cat > mixed_string_data.csv << EOF
3,"hello",3,"f",0
3,"goodbye",4,"f",0
3,"goodbye",4,"e",0
3,"hello",4,"d",0
3,"hello",4,"d",0
2,"hello",4,"d",0
2,"hello",4,"d",0
3,"goodbye",3,"f",0
3,"goodbye",4,"f",0
3,"hello",4,"f",0
3,"hello",4,"c",0
3,"hello",4,"f",0
2,"hello",4,"c",0
EOF
    cd - >/dev/null;
  fi
}

run_code_blocks()
{
  input_dir=$1;

  for f in $input_dir/*.cpp;
  do
    f_exec=${f%.cpp};
    if [[ "$mode" == "directory" ]];
    then
      echo "  Running $f_exec...";
      if ! ./$f_exec 2>&1 >/dev/null;
      then
        echo "  Error running $f_exec!";
        exit 1;
      fi
    else
      echo "  ---------------------------------------------------------------------  ";
      echo "  Contents of $f:";
      echo "";
      cat $f;
      echo "";
      echo "  ---------------------------------------------------------------------  ";
      echo "  Output of $f_exec:";
      echo "";

      if ! ./$f_exec;
      then
        echo "";
        echo "Error running $f_exec!  See output above.";
        exit 1;
      fi
      echo "";
      echo "  ---------------------------------------------------------------------  ";
    fi
  done
}

# Main loop: process the files we were asked to process.
mkdir -p doc/build/;
for f in $files;
do
  if [[ "$mode" == "directory" ]];
  then
    declare -a files_to_skip=(
        # These files have small incomplete snippets that can't compile into
        # standalone programs.
        "sample_ml_app.md"
        "hpt.md"
        "cv.md"
        "timer.md"
        "bindings.md"
        "elemtype.md"
        "iodoc.md"
        "kernels.md"
        "metrics.md"
        "trees.md"
        # The tutorials are old and are likely to be replaced, so let's not test
        # them.
        "amf.md"
        "ann.md"
        "approx_kfn.md"
        "cf.md"
        "datasetmapper.md"
        "det.md"
        "emst.md"
        "fastmks.md"
        "image.md"
        "kmeans.md"
        "linear_regression.md"
        "neighbor_search.md"
        "range_search.md"
        "reinforcement_learning.md"
        "asynchronous_learning.md"
        "ddpg.md"
        "q_learning.md"
        "sac.md"
        "td3.md"
        # Skip quickstarts, although we should eventually test them.
        "cpp.md"
    );

    skip=0;
    for skip_f in "${files_to_skip[@]}";
    do
        base_f=`basename $f`;
        if [ "$base_f" = "$skip_f" ];
        then
            skip=1;
            break;
        fi
    done

    if [[ $skip -eq 1 ]];
    then
      continue;
    fi
  fi

  echo "Building documentation for $f...";

  build_dir_tmp=${f#doc/};
  build_dir=${build_dir_tmp%.md};
  base_file=`basename $f .md`;
  mkdir -p doc/build/$build_dir/;

  extract_code_blocks $f doc/build/$build_dir/$base_file;
  # If there are no C++ files, don't do anything else..
  if ! compgen -G doc/build/$build_dir/*.cpp >/dev/null;
  then
    continue;
  fi

  compile_code_blocks doc/build/$build_dir;
  download_http_artifacts doc/build/$build_dir doc/build/;
  cd doc/build/;
  run_code_blocks $build_dir;
  cd ../../;
done
