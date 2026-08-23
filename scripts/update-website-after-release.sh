#!/usr/bin/env bash
#
# This script is used to update the website after an mlpack release is made.
# Push access to the mlpack.org website repository is needed.  Generally, this
# script will be run by mlpack-bot, so it never needs to be run by hand.
#
# Usage: update-website-after-release.sh <major> <minor> <patch>

MAJOR=$1;
MINOR=$2;
PATCH=$3;

# Make sure that the mlpack repository exists.
dest_remote_name=`git remote -v |\
                  grep "mlpack/mlpack (fetch)" |\
                  head -1 |\
                  awk -F' ' '{ print $1 }'`;

if [ "a$dest_remote_name" == "a" ]; then
  echo "No git remote found for mlpack/mlpack!";
  echo "Make sure that you've got the mlpack repository as a remote, and" \
      "that the master branch from that remote is checked out.";
  echo "You can do this with a fresh repository via \`git clone" \
      "https://github.com/mlpack/mlpack\`.";
  exit 1;
fi

# Update the checked out repository, so that we can get the tags.
git fetch $dest_remote_name;

# Check out a copy of the ensmallen.org repository.
git clone git@github.com:mlpack/mlpack.org /tmp/mlpack.org/;

# Create the release file.
git archive --prefix=mlpack-$MAJOR.$MINOR.$PATCH/ $MAJOR.$MINOR.$PATCH |\
    gzip > /tmp/mlpack.org/files/mlpack-$MAJOR.$MINOR.$PATCH.tar.gz;

# Now update the website.
wd=`pwd`;
cd /tmp/mlpack.org/;

# Get the release date.
cd files/;
tar -xvzpf mlpack-$MAJOR.$MINOR.$PATCH.tar.gz;
cd mlpack-$MAJOR.$MINOR.$PATCH;
full_release_date=`grep -m 1 '^_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_$' HISTORY.md | sed 's/_//g'`;
rel_year=`echo $full_release_date | awk -F'-' '{ print $1 }'`;
rel_mon=`echo $full_release_date | awk -F'-' '{ print $2 }'`;
rel_day=`echo $full_release_date | awk -F'-' '{ print $3 }' | sed 's/^0//'`;
rel_mon_txt=`date -d "$rel_mon/01" +%B`;
cd ../;
rm -r mlpack-$MAJOR.$MINOR.$PATCH/;

# Get the size.
tar_size=`ls -lh mlpack-$MAJOR.$MINOR.$PATCH.tar.gz | awk -F' ' '{ print $5 }'`;
cd ../;

# Update URLs to downloadable files (tarball and MSI).
sed -i 's/mlpack-[0-9]\.[0-9]\.[0-9]/mlpack-'$MAJOR'.'$MINOR'.'$PATCH'/g' index.html;
sed -i 's/mlpack-[0-9]\.[0-9]\.[0-9]/mlpack-'$MAJOR'.'$MINOR'.'$PATCH'/g' download.html;

# Update listed version number.
sed -i 's/Latest version: <b>[0-9]\.[0-9]\.[0-9]<\/b>/Latest version: <b>'$MAJOR'.'$MINOR'.'$PATCH'<\/b>/g' index.html;

# Update release date.
sed -i 's/Released [A-Za-z]* [0-9]\{1,2\}, [0-9][0-9][0-9][0-9]/Released '$rel_mon_txt' '$rel_day', '$rel_year'/' download.html;
sed -i 's/released [A-Za-z]* [0-9]\{1,2\}, [0-9][0-9][0-9][0-9]/released '$rel_mon_txt' '$rel_day', '$rel_year'/' index.html;

# Update release size.
sed -i 's/<span id="file-size">[^<]*<\/span>/<span id="file-size">'$tar_size'B<\/span>/' index.html;

git add index.html download.html;
git commit -m "Update links to latest stable version.";

git add files/mlpack-$MAJOR.$MINOR.$PATCH.tar.gz;
git commit -m "Release version $MAJOR.$MINOR.$PATCH.";

# Finally, push, and we're done.
git push origin;
cd $wd;

rm -rf /tmp/mlpack.org;
