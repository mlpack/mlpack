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

SECRET_DIR=$1
SECRET=$(cat ${SECRET_DIR}/secret.txt)

SVN_MSG="[auto-packager] updating packaging to reflect new upstream release"

svn-upgrade -u

if [ $? -eq 0 ]
then
  echo "[auto-packager] uscan reports updates upstream, updating packaging info"

  TDIR="$(mktemp -d)"
  trap '[ ! -d "$TDIR" ] || rm -r "$TDIR"' EXIT
  
  cp debian/changelog $TDIR/changelog
  sed "s/UNRELEASED/$(lsb_release -cs)/" < $TDIR/changelog > debian/changelog
   
  rm -rf $TDIR
  
  echo "[auto-packager] attempting to update the repository with changes"
  echo svn ci -m $SVN_MSG --username speet3 --password $SECRET
else
  echo "[auto-packager] svn-update reports local package is up-to-date"
fi
