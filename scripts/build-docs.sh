#!/usr/bin/env bash
#
# Convert all the Markdown files in doc/ to HTML.
#
# This requires `kramdown` to be available on the path.  `tidy` and
# `linkchecker` (the Python package) and `checklink` (from w3c-linkchecker on
# Debian) are used to test the output and must also be available and on the
# path.  `sqlite3` must also be available.
#
# Run this from the root directory of the repository.
#
# The output directory can be specified as the first option.
#
# If the environment variable DISABLE_HTML_CHECKS is specified, then checks are
# skipped.
#
# If the environment variable LINK_CACHE_FILE is specified, then that file is
# used as a cache of already-valid links that will not be checked.

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
    echo "tidy not installed!  Cannot check documentation.";
    exit 1;
  fi

  if ! command -v checklink &>/dev/null
  then
    echo "checklink not installed!  Cannot check documentation.";
    exit 1;
  fi

  if ! command -v linkchecker &>/dev/null
  then
    echo "linkchecker not installed!  Cannot check documentation.";
    exit 1;
  fi

  if ! command -v sqlite3 &> /dev/null
  then
    echo "sqlite3 not installed!  Cannot check documentation.";
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

# Now take a second pass to check all local links.
if [ -z ${DISABLE_HTML_CHECKS+x} ];
then
  find "$output_dir" -iname '*.html' -print0 | while read -d $'\0' f
  do
    echo "Checking local links and anchors in $f...";

    # To run checklink we have to strip out some perl stderr warnings...
    checklink -qs \
        --follow-file-links \
        --suppress-broken 405 \
        --suppress-broken 503 \
        --suppress-broken 301 \
        --suppress-broken 400 \
        -X "^http.*$" "$f" 2>&1 |
        grep -v 'Use of uninitialized value' > checklink_out;
    if [ -s checklink_out ];
    then
      # Store up all failures to print them at once.
      cat checklink_out >> overall_checklink_out;
    fi
    rm -f checklink_out;
  done

  # Check to see if there were any failures, all at once.
  if [ -f overall_checklink_out ];
  then
    cat overall_checklink_out;
    rm -f overall_checklink_out;
    exit 1;
  fi

  # Check to see if there were any failures, all at once.
fi

# Utility script to create linkchecker result SQL table, with a bit of extra
# information.
cat > create.sql << EOF
create table linksdb (
    urlname        varchar(256) not null,
    parentname     varchar(256),
    baseref        varchar(256),
    valid          int,
    result         varchar(256),
    warning        varchar(512),
    info           varchar(512),
    url            varchar(256),
    line           int,
    col            int,
    name           varchar(256),
    checktime      int,
    dltime         int,
    size           int,
    cached         int,
    level          int not null,
    modified       int,
    resulttime     timestamp,
    validdays      int
);
EOF

# Finally, take a third pass to check external links.
if [ -z ${DISABLE_HTML_CHECKS+x} ];
then
  # Create a basic config file for linkchecker.  We will append domains to ignore
  # to this as we go.
  echo "[checking]" > "$output_dir/linkcheckerrc.in";
  echo "maxrequestspersecond=2" >> "$output_dir/linkcheckerrc.in";
  echo "" >> "$output_dir/linkcheckerrc.in";
  echo "[filtering]" >> "$output_dir/linkcheckerrc.in";
  echo "ignore=" >> "$output_dir/linkcheckerrc.in";
  echo "  ^(?!http).*$" >> "$output_dir/linkcheckerrc.in";
  # Github issues/pull requests redirect to each other and we link to so many of
  # them it's not worth checking them.
  echo "  ^https://github.com/mlpack/mlpack/issues/[0-9]*$" >> "$output_dir/linkcheckerrc.in";
  echo "  ^https://github.com/mlpack/mlpack/issues[?]q.*$" >> "$output_dir/linkcheckerrc.in";
  echo "  ^https://github.com/mlpack/mlpack/pulls[?]q.*$" >> "$output_dir/linkcheckerrc.in";

  # Initialize our cache or take the current version of it.
  if [ ! -z ${LINK_CACHE_FILE+x} ];
  then
    if [ -f ${LINK_CACHE_FILE} ];
    then
      cp "$LINK_CACHE_FILE" "$output_dir/all_links.db";
    else
      rm -f "$output_dir/all_links.db";
      cat create.sql | sqlite3 "$output_dir/all_links.db";
    fi
  else
    rm -f "$output_dir/all_links.db";
    cat create.sql | sqlite3 "$output_dir/all_links.db";
  fi

  find "$output_dir" -iname '*.html' -print0 | while read -d $'\0' f
  do
    echo "Checking external links in $f...";

    # Generate our config file for this file by appending all valid files that
    # we have already seen.  Note that we have to append $ to all the ignore
    # patterns so that we don't accidentally match anchors that haven't been
    # checked yet.
    cp "$output_dir/linkcheckerrc.in" "$output_dir/linkcheckerrc";
    echo "SELECT DISTINCT urlname FROM linksdb
          WHERE valid = 1 AND
                urlname LIKE 'http%' AND
                julianday(datetime()) - julianday(resulttime) < validdays AND
                (result LIKE '200%' OR
                 result = 'filtered' OR
                 result = 'syntax OK');" | sqlite3 "$output_dir/all_links.db" |\
        sed 's/^/  /' |\
        sed 's/?/\\?/g' |\
        sed 's/$/$/' >> "$output_dir/linkcheckerrc";

    # Run linkchecker, and make things a little bit prettier if there are
    # failures.
    rm -f links.sql;
    linkchecker --check-extern \
        --recursion-level=1 \
        --threads=4 \
        --verbose \
        --no-status \
        --output=failures \
        --file-output=sql/ascii/links.sql \
        --config="$output_dir/linkcheckerrc" \
        $f |\
        awk -F"', '" '{ print $2; }' |\
        sed 's/'"'"')"$//' |\
        sed 's/^/Failed: /' |\
        sed 's/$/; will try again at the end of the run./';

    # Print the number of links we checked and the number we filtered.
    total_links=`cat links.sql | grep -v '^--' | grep 'http' | wc -l`;
    filtered_links=`grep 'filtered' links.sql | grep -v '^--' | grep 'http' |\
        wc -l`;
    echo "  $filtered_links of $total_links external links were cached.";

    # Insert results into the database.  We have to insert the timestamp and the
    # number of days the result is valid for.  For that, we use a random number
    # of days, because we don't want *all* of our results to expire on the same
    # CI run and have it take forever.
    cat links.sql |\
        sed 's/modified) values (/modified,resulttime,validdays) values (/' |\
        sed "s/);$/, current_timestamp, random() % 10 + 25);/" |\
        sqlite3 "$output_dir/all_links.db";

    # Print any warnings too, because we will try them again later.
    cat create.sql | sqlite3 tmp.db;
    cat links.sql |\
        sed 's/modified) values (/modified,resulttime,validdays) values (/' |\
        sed "s/);$/, current_timestamp, random() % 10 + 25);/" |\
        sqlite3 tmp.db;
    echo "SELECT DISTINCT urlname, warning FROM linksdb
          WHERE valid = 1 AND
                warning IS NOT NULL AND
                (result NOT LIKE '200%' AND
                 warning NOT LIKE '%307 Temporary Redirect%' AND
                 result <> 'filtered' AND
                 result <> 'syntax OK');" |\
        sqlite3 tmp.db |\
        awk -F'|' '{ print "Warning: "$1": "$2"; will try again at the end of the run."; }';
    rm -f tmp.db;
  done

  # Second chance on errors and warnings: filter out any spurious failures.
  echo "SELECT DISTINCT urlname FROM linksdb
        WHERE valid = 0 OR
              (warning IS NOT NULL AND
               warning NOT LIKE '%307 Temporary Redirect%') OR
              (result NOT LIKE '200%' AND
               result <> 'filtered' AND
               result <> 'syntax OK');" | sqlite3 "$output_dir/all_links.db" >\
      links_to_check.txt;
  num_links=`cat links_to_check.txt | wc -l`;
  if [ $num_links -gt 0 ];
  then
    echo "Second check for the following URLs that failed the first time:";
    cat links_to_check.txt | sed 's/^/  /';

    # Slow down the process to try and fix any links that got rate limited.
    cat "$output_dir/linkcheckerrc.in" |\
        sed 's/maxrequestspersecond=.*$/maxrequestspersecond=1/' >\
        "$output_dir/linkcheckerrc";

    linkchecker --check-extern \
        --recursion-level=0 \
        --threads=1 \
        --file-output=sql/ascii/links_failed.sql \
        --no-status \
        --verbose \
        --config="$output_dir/linkcheckerrc" \
        `cat links_to_check.txt | tr '\n' ' '`;

    cat create.sql | sqlite3 tmp.db;
    cat links_failed.sql |\
        sed 's/modified) values (/modified,resulttime,validdays) values (/' |\
        sed "s/);$/, current_timestamp, random() % 10 + 25);/" |\
        sqlite3 tmp.db;
    echo "SELECT DISTINCT urlname, result FROM linksdb
          WHERE valid = 0" | sqlite3 tmp.db |\
        awk -F'|' '{ print "  "$1": "$2; }' > links_failed.txt;
    echo "SELECT DISTINCT urlname, warning FROM linksdb
          WHERE valid = 1 AND warning IS NOT NULL" | sqlite3 tmp.db |\
        awk -F'|' '{ print "  "$1": "$2; }' > links_warned.txt;

    # Also add the second pass results to the global cache.
    cat links_failed.sql |\
        sed 's/modified) values (/modified,resulttime,validdays) values (/' |\
        sed "s/);$/, current_timestamp, random() % 10 + 25);/" |\
        sqlite3 "$output_dir/all_links.db";

    total_links_failed=`cat links_failed.txt links_warned.txt | wc -l`;
    if [ $total_links_failed -gt 0 ];
    then
      echo "The following links have failed:";

      cat links_failed.txt links_warned.txt;
      rm -f links_failed.sql tmp.db links_failed.txt links_warned.txt;
      exitcode=1;
    else
      exitcode=0;
    fi

    rm -f tmp.db links_failed.sql;
  else
    exitcode=0;
  fi

  rm -f links_to_check.txt;

  # Add to the global cache.
  if [ ! -z ${LINK_CACHE_FILE+x} ];
  then
    mv "$output_dir/all_links.db" "${LINK_CACHE_FILE}";
    echo "DELETE FROM linksdb
          WHERE valid = 0 OR
                warning IS NOT NULL OR
                julianday(datetime()) - julianday(resulttime) >= validdays;" |\
        sqlite3 "${LINK_CACHE_FILE}";

    # Keep only the most recent entry for a given urlname, to keep the size of
    # the cache as small as possible.
    echo "CREATE TABLE tmp_linksdb AS SELECT * FROM linksdb
          GROUP BY urlname HAVING MAX(resulttime) ORDER BY urlname;" |\
        sqlite3 "${LINK_CACHE_FILE}";
    echo "DROP TABLE linksdb;" | sqlite3 "${LINK_CACHE_FILE}";
    echo "ALTER TABLE tmp_linksdb RENAME TO linksdb;" | sqlite3 "${LINK_CACHE_FILE}";
  fi

  # Pick all the links that are within a week of timing out and run them again,
  # to see if we can "refresh" them.  This is intended to handle situations
  # where flaky URLs may not always work, but they will be tried a handful of
  # times over the week before their last run expires.  The hope is that one of
  # those runs in the last week before they expire will succeed, preventing a
  # documentation job from failing due to a bad link.
  echo "SELECT DISTINCT urlname FROM linksdb
        WHERE valid = 1 AND
            urlname LIKE 'http%' AND
            validdays -
                (julianday(datetime()) - julianday(resulttime)) <= 7 AND
            (result LIKE '200%' OR
             result = 'filtered' OR
             result = 'syntax OK');" |\
      sqlite3 "$output_dir/all_links.db" > links_to_check.txt;
  num_links=`cat links_to_check.txt | wc -l`;
  if [ $num_links -gt 0 ];
  then
    echo "Checking $num_links links before their cache entry expires...";
    linkchecker --check-extern \
        --recursion-level=0 \
        --threads=1 \
        --file-output=sql/ascii/links_output.sql \
        --output=failures \
        --no-status \
        --verbose \
        --config="$output_dir/linkcheckerrc" \
        `cat links_to_check.txt | tr '\n' ' '` |\
        awk -F"', '" '{ print $2; }' |\
        sed 's/'"'"')"$//' |\
        sed 's/^/Warning: /' |\
        sed 's/$/ failed, but cache entry not yet expired./';

    cat links_output.sql |\
        sed 's/modified) values (/modified,resulttime,validdays) values (/' |\
        sed "s/);$/, current_timestamp, random() % 10 + 25);/" |\
        sqlite3 "$output_dir/all_links.db";
    # Filter out any bad links.
    echo "DELETE FROM all_links WHERE valid = 0;" |\
        sqlite3 "$output_dir/all_links.db";
  fi

  # Clean up unnecessary files.
  rm -f "$output_dir/link_errors.csv" "$output_dir/all_links.csv" \
      "$output_dir/linkcheckerrc.in" "$output_dir/linkcheckerrc";
  rm -f links.csv links_failed.csv;
else
  exitcode=0;
fi

# Remove temporary files.
rm -f create.sql;
if [ "a$del_header" == "a1" ];
then
  rm -f "$template_html_header";
fi

if [ "a$del_footer" == "a1" ];
then
  rm -f "$template_html_footer";
fi

exit $exitcode;
