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

# These may be specific to the old website.
sed --in-place 's/[0-9]\.[0-9]\.[0-9]/'$MAJOR'.'$MINOR'.'$PATCH'/g' index.md;
sed --in-place 's/[0-9]\.[0-9]\.[0-9]/'$MAJOR'.'$MINOR'.'$PATCH'/g' docs.md;
sed --in-place 's/[0-9]\.[0-9]\.[0-9]/'$MAJOR'.'$MINOR'.'$PATCH'/g' getstarted.md;
sed --in-place 's/[0-9]\.[0-9]\.[0-9]/'$MAJOR'.'$MINOR'.'$PATCH'/g' community.md;
git add index.md docs.md getstarted.md community.md;

# These may be specific to the new website.
sed --in-place 's/mlpack-[0-9]\.[0-9]\.[0-9]/mlpack-'$MAJOR'.'$MINOR'.'$PATCH'/g' html/index.html;
sed --in-place 's/Version [0-9]\.[0-9]\.[0-9]/Version '$MAJOR'.'$MINOR'.'$PATCH'/g' html/index.html;
sed --in-place 's/[0-9]\.[0-9]\.[0-9]/'$MAJOR'.'$MINOR'.'$PATCH'/g' html/getstarted.html;
sed --in-place 's/[0-9]\.[0-9]\.[0-9]/'$MAJOR'.'$MINOR'.'$PATCH'/g' html/config/install.md;
git add html/index.html html/getstarted.html html/config/install.md;

git commit -m "Update links to latest stable version.";

git add files/mlpack-$MAJOR.$MINOR.$PATCH.tar.gz;
git commit -m "Release version $MAJOR.$MINOR.$PATCH.";

# Finally, push, and we're done.
git push origin;
cd $wd;

rm -rf /tmp/mlpack.org;
