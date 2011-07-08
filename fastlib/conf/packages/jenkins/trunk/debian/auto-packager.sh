#!/bin/bash
# Test the upstream jenkins mirror for war and plugin updates.
#
# The subversion password required for the script to commit its
#  updates to the repository should be contained in a file called
#  "secret.txt" found in the directory passed in as the first argument
#
# Copyright 2011  Sterling Peet <sterling.peet@gatech.edu>

DEBEMAIL="sterling.peet@gatech.edu"
DEBFULLNAME="Sterling Peet"
export DEBEMAIL DEBFULLNAME

PKG_NAME="jenkins"
SVN_URL="http://svn.cc.gatech.edu/fastlab/fastlib/conf/packages/jenkins/trunk"
SVN_USR="speet3"
SVN_MSG="[auto-packager] updating packaging to reflect new upstream release"
SECRET_DIR=$1
SECRET=$(cat ${SECRET_DIR}/secret.txt)
PWD=$(pwd)


TDIR="$(mktemp -d)"
trap '[ ! -d "$TDIR" ] || rm -r "$TDIR"' EXIT

cd $TDIR
svn co $SVN_URL $PKG_NAME --username $SVN_USR --password $SECRET --non-interactive
cd $PKG_NAME
svn cat -r 8500 debian/changelog > debian/changelog
svn-upgrade -u --noninteractive

if [ $? -eq 0 ]
then
  echo "[auto-packager] uscan reports updates upstream, updating packaging info"

  CTDIR="$(mktemp -d)"
  trap '[ ! -d "$TDIR" ] || rm -r "$TDIR"' EXIT
  
  cp debian/changelog $CTDIR/changelog
  sed "s/UNRELEASED/$(lsb_release -cs)/" < $CTDIR/changelog > debian/changelog
  
  echo "[auto-packager] New changelog contents:"
  cat debian/changelog
  rm -rf $CTDIR
  
  echo "[auto-packager] [auto-packager-warning] attempting to update the repository with changes"
  echo svn ci -m "$SVN_MSG" --username $SVN_USR --password $SECRET
else
  echo "[auto-packager] svn-update reports local package is up-to-date"
fi

cd $PWD
rm -rf $TDIR

