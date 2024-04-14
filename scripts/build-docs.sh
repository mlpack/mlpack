#!/usr/bin/env bash
#
# Convert all the Markdown files in doc/ to HTML.
# This requires `kramdown` to be available on the path.
# `tidy` and `checklink` (from Debian's w3c-linkchecker package) are used to
# test the output and must also be available and on the path.
# Run this from the root directory of the repository.
# The output directory can be specified as the first option.

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [output_dir/]";
  exit 1;
elif [ "$#" -eq 1 ]; then
  output_dir=$1;
else
  output_dir=doc/html;
fi

template_html_header="${output_dir}/template.html.header";
template_html_footer="${output_dir}/template.html.footer";
template_html_sidebar="${output_dir}/template.html.sidebar";

if ! command -v kramdown &>/dev/null
then
  echo "kramdown not installed!  Cannot build documentation.";
  exit 1;
fi

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
  # We start by trying to catch a special case of README.md, which our
  # documentation puts in a slightly different place.  In addition, because
  # README.md is being moved to the root of the documentation, we must adjust
  # links in that file differently.
  if [[ $input_file != "README.md" ]];
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
  fi

  # Replace any links to source files with a link to the current version of the
  # source file on Github.
  sed -i 's/\](\/src\/\([^ ]*\)\.hpp)/](https:\/\/github.com\/mlpack\/mlpack\/blob\/master\/src\/\1.hpp)/' $input_file.tmp;

  kramdown \
      -x parser-gfm \
      --syntax-highlighter rouge \
      --syntax-highlighter-opts '{ default_lang: c++ }' \
      --auto_ids \
      $input_file.tmp > "$output_file.tmp" || exit 1;
  cat "$template_html_header" | sed "s|LINKROOT|$link_root|" > "$output_file";

  # Create the sidebar.  Extract anchors from the page, unless we are looking at
  # index.md, since the permanent part of the sidebar links all over index.md
  # anyway.
  cat "$template_html_sidebar" | sed "s|LINKROOT|$link_root|" >> "$output_file";
  if [[ $input_file != "doc/index.md" ]];
  then
    create_page_sidebar_section "$output_file.tmp" "$output_file" "$dir_name";
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
  input_file="$1";
  output_file="$2";
  dir_name="$3"; # The directory containing the documentation.
  input_file_base=`basename "$input_file" .html.tmp`;

  # Extract h2/h3 anchors into a list.  For individual method documentation, we
  # only extract h3 anchors because those use h2s as their headings.  And, for
  # core.md, we want to extract both h2 and h3 anchors.
  if [[ "$dir_name" == "user/methods" ]];
  then
    # The page title on individual methods is encoded as an h2.
    page_title=`grep '<h2 id=' "$input_file" |\
        head -1 |\
        sed 's/^<h2 id="[^"]*">\(.*\)<\/h2>/\1/'`;

    grep '<h3 id=' "$input_file" | sed 's/<h3 id="\([^"]*\)">\(.*\)<\/h3>/<li><a href="#\1">\2<\/a><\/li>/' > "$output_file.side.tmp";
  elif [[ "$input_file_base" == "core" ]];
  then
    # The page title on the core class documentation page is encoded as an h1.
    page_title=`grep '<h1 id=' "$input_file" |\
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
    grep '<h[23] id=' "$input_file" |\
        sed 's/^<\(h[23]\) id="\([^"]*\)">\(.*\)<\/h[23]>/\1\t\2\t\3/' > "$output_file.side.list.tmp";
    in_block=0;
    while read line; do
      # First, extract the pieces of each line.
      line_type=`echo "$line"    | awk -F'\t' '{ print $1 }'`;
      anchor_name=`echo "$line"  | awk -F'\t' '{ print $2 }'`;
      anchor_title=`echo "$line" | awk -F'\t' '{ print $3 }'`;

      # For an h2, we have to print a summary block.
      # Note that this assumes that *all* h2s have h3 children.  If that's not
      # true, some extra processing will be needed.
      if [ "$line_type" = "h2" ];
      then
        if [ "$in_block" = "1" ];
        then
          # We have to close the previous block.
          echo "</ul></details></li>" >> "$output_file.side.tmp";
        fi

        # Create the new details block.
        echo "<li><details><summary>" >> "$output_file.side.tmp";
        echo "<a href=\"#$anchor_name\">" >> "$output_file.side.tmp";
        echo "$anchor_title" >> "$output_file.side.tmp";
        echo "</a>" >> "$output_file.side.tmp";
        echo "</summary>" >> "$output_file.side.tmp";
        echo "<ul>" >> "$output_file.side.tmp";
        in_block=1;
      else
        echo "  <li><a href=\"#$anchor_name\">" >> "$output_file.side.tmp";
        echo "  $anchor_title" >> "$output_file.side.tmp";
        echo "  </a></li>" >> "$output_file.side.tmp";
      fi
    done < "$output_file.side.list.tmp";

    # Close the last h2 block, if we need to.
    if [ "$in_block" = "1" ];
    then
      echo "</ul></details></li>" >> "$output_file.side.tmp";
    fi

    rm -f "$output_file.side.list.tmp";
  else
    # On other pages, the page title is encoded as an h1.
    page_title=`grep '<h1 id=' "$input_file" |\
        head -1 |\
        sed 's/^<h1 id="[^"]*">\(.*\)<\/h1>/\1/'`;

    grep '<h2 id=' "$input_file" | sed 's/<h2 id="\([^"]*\)">\(.*\)<\/h2>/<li><a href="#\1">\2<\/a><\/li>/' > "$output_file.side.tmp";
  fi
  lines=`cat "$output_file.side.tmp" | wc -l`;

  echo "<ul>" >> "$output_file";

  # Make the top of the sidebar.
  if [ -n "$page_title" ];
  then
    echo "<li class=\"page_title\"><b>$page_title</b> <a href=\"#\">[top]</a>" >> "$output_file";
  else
    echo "<li><a href=\"#\">[top of page]</a>" >> "$output_file";
  fi

  if [[ "$lines" -gt 0 ]];
  then
    echo "<ul>" >> "$output_file";
    cat "$output_file.side.tmp" >> "$output_file";
    echo "</ul>" >> "$output_file";
  fi
  echo "</li>" >> "$output_file";
  echo "</ul>" >> "$output_file";
  echo "</div>" >> "$output_file";

  rm -f "$output_file.side.tmp";
}

rm -rf "$output_dir";
mkdir -p "$output_dir";
cp doc/css/* "$output_dir";
mkdir -p "$output_dir/user/img/";
cp doc/img/* "$output_dir/user/img/";
mkdir -p "$output_dir/tutorials/res/";
cp doc/tutorials/res/* "$output_dir/tutorials/res/";

# Create the template files we will use.
create_template_header "$template_html_header";
create_template_footer "$template_html_footer";
cp doc/sidebar.html "$template_html_sidebar";

# Process all the .md files.
for f in README.md `find ./doc/ -iname '*.md'`;
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

  tidy -qe "$of" || exit 1;
done

# Now take a second pass to check all the links.
find "$output_dir" -iname '*.html' -print0 | while read -d $'\0' f
do
  echo "Checking links in $f...";

  # To run checklink we have to strip out some perl stderr warnings...
  checklink -qs --follow-file-links --suppress-broken 405 --suppress-broken 301 "$f" 2>&1 |
      grep -v 'Use of uninitialized value' > checklink_out;
  if [ -s checklink_out ];
  then
    cat checklink_out;
    exit 1;
  fi
  rm -f checklink_out;
done

# Remove temporary files.
rm -f "$template_html_header";
rm -f "$template_html_footer";
