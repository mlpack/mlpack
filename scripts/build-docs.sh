#!/usr/bin/env bash
#
# Convert all the Markdown files in doc/ to HTML.
# This requires `kramdown` to be available on the path.
# `tidy` and `checklink` (from Debian's w3c-linkchecker package) are used to
# test the output and must also be available and on the path.
# Run this from the root directory of the repository.
# The output directory can be specified as the first option.
# If the environment variable DISABLE_HTML_CHECKS is specified, then checks are
# skipped.

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [output_dir/]";
  exit 1;
elif [ "$#" -eq 1 ]; then
  output_dir=$1;
else
  output_dir=doc/html;
fi

# If the header and footer already exist, they will not be overwritten.
template_html_header="${output_dir}/template.html.header";
template_html_footer="${output_dir}/template.html.footer";
template_html_sidebar="${output_dir}/template.html.sidebar";

if ! command -v kramdown &>/dev/null
then
  echo "kramdown not installed!  Cannot build documentation.";
  exit 1;
fi

# If DISABLE_HTML_CHECKS is set, then we won't use tidy or checklink.
if [ -z ${DISABLE_HTML_CHECKS+x} ];
then
  if ! command -v tidy &>/dev/null
  then
    echo "tidy not installed!  Cannot build documentation.";
    exit 1;
  fi

  if ! command -v checklink &>/dev/null
  then
    echo "checklink not installed!  Cannot build documentation.";
    exit 1;
  fi
fi

if [ ! -d doc/ ];
then
  echo "Run this script from the root of the mlpack repository.";
  exit 1;
fi

# Define utility function to run kramdown and turn an .md file to an .html file.
run_kramdown()
{
  input_file=$1;
  # This converts, e.g., ./doc/user/index.md -> doc/html/user/index.html.
  tmp=${input_file#./doc/}; # Strip leading ./doc/.
  output_file="$output_dir/${tmp%.md}.html";

  # Determine what the link root is.  If we're in the root directory, it's
  # nothing, otherwise it's one of more '../'s.
  dir_name=$(dirname $tmp);
  link_root="";
  if [[ "$dir_name" != "." ]];
  then
    levels_below_root=`echo $dir_name | awk -F'/' '{ print NF }'`;
    link_root=$(printf '../%.0s' `seq 1 $levels_below_root`);
  fi

  # Make the enclosing directory if needed.
  out_dir=`dirname "$output_file"`;
  mkdir -p "$out_dir";

  # Kramdown doesn't detect languages correctly with the "```" fence; instead it
  # needs the "~~~" fence.
  sed 's/^```/~~~/' $input_file > $input_file.tmp;

  # Our documentation is full of relative links, like
  # [name](other_file.md#anchor).  We need these to turn into links to the
  # rendered HTML file, like [name](other_file.html#anchor).  We'll do this with
  # regular expressions...
  #
  # - Note that this assumes there are no spaces in any filenames.
  # - We also only catch the second part of the link '](' because the name of
  #   the link could be spread on multiple lines.
  #
  # We start by trying to catch the special cases README.md and HISTORY.md,
  # which our documentation puts in a slightly different place.  In addition,
  # because those files are being moved to the root of the documentation, we
  # must adjust links differently.
  if [[ $input_file != "README.md" ]] && [[ $input_file != "HISTORY.md" ]];
  then
    sed -i "s|\]([./]*README.md)|](${link_root}README.html)|g" $input_file.tmp;
    sed -i "s|\]([./]*README.md#[0-9]-\([^ ]*\))|](${link_root}README.html#\1)|g" $input_file.tmp;
    sed -i 's/\](\([^ ]*\).md)/](\1.html)/g' $input_file.tmp;
    sed -i 's/\](\([^ ]*\).md#\([^ ]*\))/](\1.html#\2)/g' $input_file.tmp;
  else
    sed -i 's/\](doc\/\([^ ]*\).md)/](\1.html)/g' $input_file.tmp;
    sed -i 's/\](doc\/\([^ ]*\).md#\([^ ]*\))/](\1.html#\2)/g' $input_file.tmp;

    # The README specifically has a link to GOVERNANCE.md, but we want to
    # preserve that.  We're not building that file into Markdown.
    sed -i 's|(./GOVERNANCE.md)|(https://github.com/mlpack/mlpack/blob/master/GOVERNANCE.md)|' $input_file.tmp;

    # Ugh!  Github naming of anchors is different than kramdown, and so we have
    # to adjust all the table-of-contents anchor links in the README (and in
    # that file only).
    sed -i 's/\](#[0-9][0-9]-\([^ ]*\))/](#\1)/g' $input_file.tmp;
    sed -i 's/\](#[0-9]-\([^ ]*\))/](#\1)/g' $input_file.tmp;
    sed -i 's/\](#[0-9][0-9]\([^ ]*\))/](#\1)/g' $input_file.tmp;
    sed -i 's/\](#[0-9]\([^ ]*\))/](#\1)/g' $input_file.tmp;

    # For HISTORY.md, we want to turn all references to Github issues into
    # actual links, and all references to Github usernames into links to their
    # profile.
    if [[ $input_file == "HISTORY.md" ]];
    then
      sed -i 's/#\([0-9][0-9]*\)/[#\1](https:\/\/github.com\/mlpack\/mlpack\/issues\/\1)/g' $input_file.tmp;
      sed -i 's/\([^`]\)@\([a-zA-Z0-9_-][a-zA-Z0-9_-]*\)/\1[@\2](https:\/\/github.com\/\2)/g' $input_file.tmp;
    fi
  fi

  # Replace any links to source files with a link to the current version of the
  # source file on Github.
  sed -i 's/\](\/src\/\([^ ]*\)\.hpp)/](https:\/\/github.com\/mlpack\/mlpack\/blob\/master\/src\/\1.hpp)/' $input_file.tmp;

  # If this is binding documentation or quickstart documentation, don't set the
  # default language to C++.
  set_lang=1;
  if [[ `dirname $input_file` == "./doc/user/bindings" ]];
  then
    set_lang=0;
  elif [[ `dirname $input_file` == "./doc/quickstart" ]];
  then
    if [[ `basename $input_file .md` != "cpp" ]];
    then
      set_lang=0;
    fi
  elif [[ $input_file == "HISTORY.md" ]];
  then
    set_lang=0;
  fi

  if [[ "$set_lang" == "0" ]];
  then
    kramdown \
        -x parser-gfm \
        --syntax-highlighter rouge \
        --auto_ids \
        $input_file.tmp > "$output_file.tmp" || exit 1;
  else
    kramdown \
        -x parser-gfm \
        --syntax-highlighter rouge \
        --syntax-highlighter-opts '{ default_lang: c++ }' \
        --auto_ids \
        $input_file.tmp > "$output_file.tmp" || exit 1;
  fi
  cat "$template_html_header" | sed "s|LINKROOT|$link_root|" > "$output_file";

  # Create the sidebar.  Extract anchors from the page, unless we are looking at
  # index.md, since the permanent part of the sidebar links all over index.md
  # anyway.  If we are looking at binding documentation, use a slightly
  # different sidebar.
  if { [[ $dir_name != "user/bindings" ]] && \
       [[ $dir_name != "quickstart" ]] } ||
     [[ $input_file == "./doc/quickstart/cpp.md" ]];
  then
    cat "$template_html_sidebar" | sed "s|LINKROOT|$link_root|" \
        >> "$output_file";
    create_page_sidebar_section "$output_file.tmp" "$output_file" "$dir_name";
  else
    echo "Using custom sidebar...";
    cat "$template_html_sidebar" | sed "s|LINKROOT|$link_root|" |\
        sed 's|<details> <!-- default closed for non-binding pages -->|<details open="true">|' |\
        sed 's|<details open="true"> <!-- default open for non-binding pages -->|<details>|' \
        >> "$output_file";
    # Some pages may have a custom sidebar HTML file.  (Specifically,
    # generated language bindings.)
    if [[ $dir_name == "user/bindings" ]];
    then
      cat "${input_file/%.md/.sidebar.html}" | sed "s|LINKROOT|$link_root|" \
          >> "$output_file";
    else
      sidebar_file=`basename $input_file .md`.sidebar.html;
      cat "./doc/user/bindings/$sidebar_file" | sed "s|LINKROOT|$link_root|" \
          >> "$output_file";
    fi
  fi

  # Add clickable anchors to h2 and h3 headers.
  echo "<div id=\"content\">" >> "$output_file";
  sed -E 's/<h([23]) id="([^"]*)">/<h\1 id="\2"><a href="#\2" class="pl">🔗<\/a> /' "$output_file.tmp" >> "$output_file";

  # Simple postprocessing to make tidy a little happier.
  # (Muting the warning won't change the error code!)
  sed -i 's/<table>/<table summary="">/' "$output_file";

  cat "$template_html_footer" >> "$output_file";
  rm -f $input_file.tmp "$output_file.tmp";
}

# Create the template header file.
create_template_header()
{
  output_file="$1";

  # Note that LINKROOT will be substituted into place by run_kramdown.
  cat > "$output_file" << EOF
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
  <meta content="text/html; charset=utf-8" http-equiv="Content-Type">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link type="text/css" rel="stylesheet" href="LINKROOTgfm-mod.css">
  <link type="text/css" rel="stylesheet" href="LINKROOTrouge-github-mod.css">
  <title>mlpack documentation</title>
</head>
<body>
EOF
}

# Create the template footer.
create_template_footer()
{
  output_file="$1";

  cat > "$output_file" << EOF
</div>
</body>
</html>
EOF
}

# Extract anchors to build a sidebar.
# This should take the input HTML (before anchor elements are added), and it
# appends a sidebar list to the output HTML.
create_page_sidebar_section()
{
  sb_input_file="$1";
  sb_output_file="$2";
  sb_dir_name="$3"; # The directory containing the documentation.
  sb_input_file_base=`basename "$sb_input_file" .html.tmp`;

  # Extract h2/h3 anchors into a list.  For individual method documentation, we
  # only extract h3 anchors because those use h2s as their headings.  And, for
  # core.md, we want to extract both h2 and h3 anchors.
  if [[ "$sb_dir_name" == "user/methods" ]];
  then
    # The page title on individual methods is encoded as an h2.
    page_title=`grep '<h2 id=' "$sb_input_file" |\
        head -1 |\
        sed 's/^<h2 id="[^"]*">\(.*\)<\/h2>/\1/'`;

    grep '<h3 id=' "$sb_input_file" | sed 's/<h3 id="\([^"]*\)">\(.*\)<\/h3>/<li><a href="#\1">\2<\/a><\/li>/' > "$sb_output_file.side.tmp";
  elif [[ "$sb_input_file_base" == "core" || "$sb_dir_name" == "user/core" || "$sb_dir_name" == "user/core/trees" ]];
  then
    # The page title on the core class documentation page is encoded as an h1.
    page_title=`grep '<h1 id=' "$sb_input_file" |\
        head -1 |\
        sed 's/^<h1 id="[^"]*">\(.*\)<\/h1>/\1/'`;

    # We want to collect h2s and h3s as individual documentation; each h2 should
    # have a summary/details block.  This is a little tedious to create... we'll
    # do this by creating a temporary tab-separated file with lines like
    #
    # h2  anchor_name   Anchor Title
    # h3  anchor_name   Anchor Title
    # ...
    #
    # and then we'll construct the actual sidebar using that list.
    grep '<h[23] id=' "$sb_input_file" |\
        sed 's/^<\(h[23]\) id="\([^"]*\)">\(.*\)<\/h[23]>/\1\t\2\t\3/' \
        > "$sb_output_file.side.list.tmp";
    in_block=0;
    while read line; do
      # First, extract the pieces of each line.
      line_type=`echo "$line"    | awk -F'\t' '{ print $1 }'`;
      anchor_name=`echo "$line"  | awk -F'\t' '{ print $2 }'`;
      anchor_title=`echo "$line" | awk -F'\t' '{ print $3 }'`;

      # For an h2, we have to print a summary block, if the h2 has any children.
      # (Below is a hacky way to detect that.)
      h3_lines=`grep -A 1 "$line" "$sb_output_file.side.list.tmp" |\
                tail -1 |\
                grep 'h3' |\
                wc -l`;

      # Note that this assumes that *all* h2s have h3 children.  If that's not
      # true, some extra processing will be needed.
      if [ "$line_type" = "h2" ]
      then
        # Close a block if necessary.
        if [ "$in_block" = "1" ];
        then
          # We have to close the previous block.
          echo "</ul></details></li>" >> "$sb_output_file.side.tmp";
          in_block=0;
        fi

        # Create the new details block, if the h2 has children.
        if [ "$h3_lines" -gt 0 ];
        then
          echo "<li><details><summary>" >> "$sb_output_file.side.tmp";
          echo "<a href=\"#$anchor_name\">" >> "$sb_output_file.side.tmp";
          echo "$anchor_title" >> "$sb_output_file.side.tmp";
          echo "</a>" >> "$sb_output_file.side.tmp";
          echo "</summary>" >> "$sb_output_file.side.tmp";
          echo "<ul>" >> "$sb_output_file.side.tmp";
          in_block=1;
        else
          echo "  <li><a href=\"#$anchor_name\">" >> "$sb_output_file.side.tmp";
          echo "  $anchor_title" >> "$sb_output_file.side.tmp";
          echo "  </a></li>" >> "$sb_output_file.side.tmp";
        fi
      else
        echo "  <li><a href=\"#$anchor_name\">" >> "$sb_output_file.side.tmp";
        echo "  $anchor_title" >> "$sb_output_file.side.tmp";
        echo "  </a></li>" >> "$sb_output_file.side.tmp";
      fi
    done < "$sb_output_file.side.list.tmp";

    # Close the last h2 block, if we need to.
    if [ "$in_block" = "1" ];
    then
      echo "</ul></details></li>" >> "$sb_output_file.side.tmp";
    fi

    rm -f "$sb_output_file.side.list.tmp";
  else
    # On other pages, the page title is encoded as an h1.
    page_title=`grep '<h1 id=' "$sb_input_file" |\
        head -1 |\
        sed 's/^<h1 id="[^"]*">\(.*\)<\/h1>/\1/'`;

    grep '<h2 id=' "$sb_input_file" |\
        sed 's/<h2 id="\([^"]*\)">\(.*\)<\/h2>/<li><a href="#\1">\2<\/a><\/li>/' \
        > "$sb_output_file.side.tmp";
  fi
  lines=`cat "$sb_output_file.side.tmp" | wc -l`;

  echo "<ul>" >> "$sb_output_file";

  # Make the top of the sidebar.
  if [ -n "$page_title" ];
  then
    echo "<li class=\"page_title\"><b>$page_title</b> <a href=\"#\">[top]</a>" >> "$sb_output_file";
  else
    echo "<li><a href=\"#\">[top of page]</a>" >> "$sb_output_file";
  fi

  if [[ "$lines" -gt 0 ]];
  then
    echo "<ul>" >> "$sb_output_file";
    cat "$sb_output_file.side.tmp" >> "$sb_output_file";
    echo "</ul>" >> "$sb_output_file";
  fi
  echo "</li>" >> "$sb_output_file";
  echo "</ul>" >> "$sb_output_file";
  echo "</div>" >> "$sb_output_file";

  rm -f "$sb_output_file.side.tmp";
}

# Save any existing template.
if [ -f "$template_html_header" ];
then
  mv "$template_html_header" template.html.header.tmp;
fi

if [ -f "$template_html_footer" ];
then
  mv "$template_html_footer" template.html.footer.tmp;
fi

rm -rf "$output_dir";
mkdir -p "$output_dir";
cp -v doc/css/* "$output_dir";
mkdir -p "$output_dir/img/";
cp -v doc/img/* "$output_dir/img/";
mkdir -p "$output_dir/tutorials/res/";
cp -v doc/tutorials/res/* "$output_dir/tutorials/res/";

# Create the template files we will use, if they don't already exist.
if [ -f template.html.header.tmp ];
then
  mv template.html.header.tmp "$template_html_header";
else
  create_template_header "$template_html_header";
  del_header=1;
fi

if [ -f template.html.footer.tmp ];
then
  mv template.html.footer.tmp "$template_html_footer";
else
  create_template_footer "$template_html_footer";
  del_footer=1;
fi

cp doc/sidebar.html "$template_html_sidebar";

# Process all the .md files.
for f in README.md HISTORY.md `find ./doc/ -iname '*.md'`;
do
  # Skip the JOSS paper...
  if [[ $f == *"joss_paper"* ]]; then
    continue;
  fi

  echo "Processing $f...";
  run_kramdown $f;

  # This converts, e.g., ./doc/user/index.md -> doc/html/user/index.html.
  tmp=${f#./doc/}; # Strip leading ./doc/.
  of="$output_dir/${tmp%.md}.html";

  # Check HTML, if we did not disable that.
  if [ -z ${DISABLE_HTML_CHECKS+x} ];
  then
    tidy -qe "$of" || exit 1;
  fi
done

# Now take a second pass to check all the links, if we need to.
if [ -z ${DISABLE_HTML_CHECKS+x} ];
then
  find "$output_dir" -iname '*.html' -print0 | while read -d $'\0' f
  do
    echo "Checking links in $f...";

    # To run checklink we have to strip out some perl stderr warnings...
    checklink -qs \
        --follow-file-links \
        --suppress-broken 405 \
        --suppress-broken 503 \
        --suppress-broken 301 \
        --suppress-broken 400 \
        -X "https://eigen.tuxfamily.org/index.php\?title=Main_Page" \
        -X "https://mlpack.slack.com/" "$f" 2>&1 |
        grep -v 'Use of uninitialized value' > checklink_out;
    if [ -s checklink_out ];
    then
      cat checklink_out;
      exit 1;
    fi
    rm -f checklink_out;
  done
fi

# Remove temporary files.
if [ "a$del_header" == "a1" ];
then
  rm -f "$template_html_header";
fi

if [ "a$del_footer" == "a1" ];
then
  rm -f "$template_html_footer";
fi
