#!/bin/bash
# Grab the jenkins-XXX.war file from uscan, tar and compress it into the appropriately
#   formatted jenkins_XXX.orig.tar.gz archive
#
# Copyright 2011  Sterling Peet <sterling.peet@gatech.edu>

set -e

if [ ! -f "$3" ] && [ ! -f "$1" ]; then
    echo "This script must be run via uscan or by manually specifying the war file" >&2
    exit 1
fi

WAR=

[ -f "$3" ] && WAR="$3"
[ -z "$WAR" -a -f "$1" ] && WAR="$1"

WAR_FILENAME=$(basename $WAR)
WAR_VERSION=$2
WAR_DIR=$(echo $WAR | sed -rne 's/\/jenkins-.*\.war//p')

TDIR="$(mktemp -d)"
trap '[ ! -d "$TDIR" ] || rm -r "$TDIR"' EXIT

mkdir -p $TDIR/jenkins-$WAR_VERSION
cp $WAR $TDIR/jenkins-$WAR_VERSION/jenkins.war

echo "Tarring and compressing $WAR"
tar -C $TDIR -cz --owner root --group root --mode a+rX \
	-f $WAR_DIR/jenkins_$WAR_VERSION.orig.tar.gz \
	jenkins-$WAR_VERSION

echo "Archive $WAR_DIR/jenkins_$WAR_VERSION.orig.tar.gz created"

rm -rf $TDIR


