#!/usr/bin/env bash
#
# Release a new version of mlpack.
#
# Usage: release-mlpack.sh X Y Z
#
# where X is the major version, Y is the minor version, and Z is the patch
# version.  Run this from the root of the repository.
#
# Make sure HISTORY.md is updated first!
set +e

if [ "$#" -ne "4" ];
then
  echo "Usage: mlpack-release.sh <github username> <major> <minor> <patch>";
  exit 1;
fi

# First, check for any unlicensed files.
output=$(
    for i in $(find src/ -iname '*.[hc]pp');
    do
        echo -n $i": ";
        cat $i | grep 'mlpack is free software;' | wc -l;
    done |\
        grep -v ': 1' |\
        grep -v 'arma_extend' |\
        grep -v 'core/cereal' |\
        grep -v 'std_backport' |\
        grep -v 'arma_config.hpp' |\
        grep -v 'gitversion.hpp' |\
        grep -v 'CLI11.hpp' |\
        grep -v 'bindings/R/mlpack/src/boost/serialization' |\
        grep -v 'tests/catch.hpp');
lines=`echo $output | grep -v '^[ ]*$' | wc -l`;

if [ "0$lines" -gt "0" ];
then
  echo "Unlicensed files found!  Aborting release.";
  echo "$output";
  exit 1;
fi

# Now, check that there are no local changes.
lines=`git diff | wc -l | sed -e 's/^\s*//g'`;
if [ "$lines" != "0" ]; then
  echo "git diff returned a nonzero result!";
  echo "";
  git diff;
  exit 1;
fi

# Next, make sure the origin is right.
dest_remote_name=`git remote -v |\
                  grep "mlpack/mlpack (fetch)" |\
                  head -1 |\
                  awk -F' ' '{ print $1 }'`;

if [ "a$dest_remote_name" == "a" ]; then
  echo "No git remote found for https://github.com/mlpack/mlpack!";
  echo "Make sure that you've got the ensmallen repository as a remote, and" \
      "that the master branch from that remote is checked out.";
  echo "You can do this with a fresh repository via \`git clone" \
      "https://github.com/mlpack/mlpack\`.";
  exit 1;
fi

# Also check that we're on the master branch, from the correct origin.
current_branch=`git branch --no-color | grep '^\* ' | awk -F' ' '{ print $2 }'`;
current_origin=`git rev-parse --abbrev-ref --symbolic-full-name @{u} |\
                awk -F'/' '{ print $1 }'`;
if [ "a$current_branch" != "amaster" ]; then
  echo "Current branch is $current_branch.";
  echo "This script has to be run from the master branch.";
  exit 1;
elif [ "a$current_origin" != "a$dest_remote_name" ]; then
  echo "Current branch does not track from remote mlpack repository!";
  echo "Instead, it tracks from $current_origin/master.";
  echo "Make sure to check out a branch that tracks $dest_remote_name/master.";
  exit 1;
fi

# Make sure `hub` is installed.
hub_output="`which hub`" || true;
if [ "a$hub_output" == "a" ]; then
  echo "The Hub command-line tool must be installed for this script to run" \
      "successfully.";
  echo "See https://hub.github.com for more details and installation" \
      "instructions.";
  echo "";
  echo "(apt-get install hub on Debian and Ubuntu)";
  echo "(brew install hub via Homebrew)";
  exit 1;
fi

# Check git remotes: we need to make sure we have a fork to push to.
github_user=$1;
remote_name=`git remote -v |\
            grep "$github_user/mlpack (push)" |\
            head -1 |\
            awk -F' ' '{ print $1 }'`;
if [ "a$remote_name" == "a" ]; then
  echo "No git remote found for $github_user/mlpack!";
  echo "Adding remote '$github_user'.";
  git remote add $github_user https://github.com/$github_user/mlpack;
  remote_name="$github_user";
fi
git fetch $github_user;

# Make sure everything is up to date.
git pull;

# Make updates to files that will be needed for the release.
MAJOR="$2";
MINOR="$3";
PATCH="$4";

# Update version.
sed --in-place 's/MLPACK_VERSION_MAJOR [0-9]*$/MLPACK_VERSION_MAJOR '$MAJOR'/' \
    src/mlpack/core/util/version.hpp;
sed --in-place 's/MLPACK_VERSION_MINOR [0-9]*$/MLPACK_VERSION_MINOR '$MINOR'/' \
    src/mlpack/core/util/version.hpp;
sed --in-place 's/MLPACK_VERSION_PATCH [0-9]*$/MLPACK_VERSION_PATCH '$PATCH'/' \
    src/mlpack/core/util/version.hpp;

sed --in-place 's/mlpack-[0-9]\.[0-9]\.[0-9]/mlpack-'$MAJOR'.'$MINOR'.'$PATCH'/g' \
    doc/user/sample_ml_app.md;
sed --in-place 's/mlpack-[0-9]\.[0-9]\.[0-9]/mlpack-'$MAJOR'.'$MINOR'.'$PATCH'/g' \
    doc/examples/sample-ml-app/README.txt;
sed --in-place 's/mlpack-[0-9]\.[0-9]\.[0-9]/mlpack-'$MAJOR'.'$MINOR'.'$PATCH'/g' \
    doc/examples/sample-ml-app/sample-ml-app/sample-ml-app.vcxproj;

sed --in-place 's/mlpack-[0-9]\.[0-9]\.[0-9]/mlpack-'$MAJOR'.'$MINOR'.'$PATCH'/g' \
    README.md;
sed --in-place 's/([0-9]\.[0-9]\.[0-9])/('$MAJOR'.'$MINOR'.'$PATCH')/g' \
    README.md;
sed --in-place 's/mlpack [0-9]\.[0-9]\.[0-9]/mlpack '$MAJOR'.'$MINOR'.'$PATCH'/g' \
    README.md;

sed --in-place 's/### mlpack ?[.]?[.]?/### mlpack '$MAJOR'.'$MINOR'.'$PATCH'/g' HISTORY.md;
year=`date +%Y`;
month=`date +%m`;
day=`date +%d`;
sed --in-place 's/###### ????-??-??/###### '$year'-'$month'-'$day'/g' \
    HISTORY.md;

# Get the latest release of ensmallen.
git clone https://github.com/mlpack/ensmallen /tmp/ensmallen;
cd /tmp/ensmallen;
ens_ver=`git describe --tags $(git rev-list --tags --max-count=1)`;
echo "Latest version of ensmallen: $ens_ver"
cd -;
sed --in-place "s/ensmallen-latest.tar.gz/ensmallen-$ens_ver.tar.gz/" CMakeLists.txt;
rm -rf /tmp/ensmallen;

# Make these changes on a release branch.
git checkout -b release-$MAJOR.$MINOR.$PATCH;

git add src/mlpack/core/util/version.hpp \
    doc/user/sample_ml_app.md \
    doc/examples/sample-ml-app/sample-ml-app/sample-ml-app.vcxproj \
    CMakeLists.txt \
    README.md \
    HISTORY.md;

git commit -m "Update and release version $MAJOR.$MINOR.$PATCH.";

changelog_str=`cat HISTORY.md |\
    awk '/^### /{f=0} /^### mlpack '"$MAJOR"'.'"$MINOR"'.'"$PATCH"'/{f=1} f{print}' |\
    grep -v '^#' |\
    tr '\n' '!' |\
    sed -e 's/!  [ ]*/ /g' |\
    tr '!' '\n'`;
echo "Changelog string:"
echo "$changelog_str"

# Update version again and add a new block for HISTORY.md.
sed --in-place 's/MLPACK_VERSION_PATCH [0-9]*$/MLPACK_VERSION_PATCH '$(($PATCH + 1))'/' \
    src/mlpack/core/util/version.hpp;
sed --in-place 's/ensmallen-'$ens_ver'.tar.gz/ensmallen-latest.tar.gz/' CMakeLists.txt;

echo "### mlpack ?.?.?" > HISTORY.md.new;
echo "###### ????-??-??" >> HISTORY.md.new;
echo "" >> HISTORY.md.new;
cat HISTORY.md >> HISTORY.md.new;
mv HISTORY.md.new HISTORY.md;

git add HISTORY.md;
git add src/mlpack/core/util/version.hpp CMakeLists.txt;

git commit -m "Add new block for next release to HISTORY.md.";

# Push to new branch.
git push --set-upstream $github_user release-$MAJOR.$MINOR.$PATCH;

# Next, we have to actually open the PR for the release.
hub pull-request \
    -b mlpack:master \
    -h $github_user:release-$MAJOR.$MINOR.$PATCH \
    -m "Release version $MAJOR.$MINOR.$PATCH" \
    -m "This automatically-generated pull request adds the commits necessary to
make the $MAJOR.$MINOR.$PATCH release." \
    -m "Once the PR is merged, mlpack-bot will tag the release as HEAD~1 (so
that it doesn't include the new HISTORY block) and publish it." \
    -m "Or, well, hopefully that will happen someday." \
    -m "When you merge this PR, be sure to merge it using a *rebase*." \
    -m "### Changelog" \
    -m "$changelog_str" \
    -l "t: release"

echo "";
echo "Switching back to 'master' branch.";
echo "If you want to access the release branch again, use \`git checkout " \
    "release-$MAJOR.$MINOR.$PATCH\`.";
echo 0;
