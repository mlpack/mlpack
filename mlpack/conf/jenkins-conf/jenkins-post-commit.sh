#!/bin/bash

# Bash post-commit hook for subversion repository
# triggers build on jenkis where reposititory is
# checked out

# 2011 Sterling Peet <sterling.peet@gatech.edu>

REPOS="$1"
REV="$2"
JENKINS="http://hotwheels.cc.gt.atl.ga.us:8080"

UUID=`svnlook uuid $REPOS`
/usr/bin/wget \
  --header `wget -q --output-document - \
  "${JENKINS}/crumbIssuer/api/xml?xpath=concat(//crumbRequestField,\":\",//crumb)"` \
  --post-data "`svnlook changed --revision $REV $REPOS`" \
  --output-document "-" \
  --timeout=2 \
  ${JENKINS}/subversion/${UUID}/notifyCommit?rev=$REV

