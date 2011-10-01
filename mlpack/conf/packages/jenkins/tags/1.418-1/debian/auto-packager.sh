#!/bin/bash
# Test the upstream jenkins mirror for war and plugin updates.
#
# Copyright 2011  Sterling Peet <sterling.peet@gatech.edu>

DEBEMAIL="sterling.peet@gatech.edu"
DEBFULLNAME="Sterling Peet"
export DEBEMAIL DEBFULLNAME

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
  svn ci -m "[auto-packager] updating packaging to reflect new upstream release"
else
  echo "[auto-packager] svn-update reports local package is up-to-date"
fi
