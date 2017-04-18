#!/bin/bash
# Build OS X .pkg installer
rm -rf res
mkdir res
cp -r ../build/bin res/
cp -r ../build/lib res/
cp -r ../build/include res/

cp ../README.md docs/
cp ../LICENSE.txt docs/

rm -rf pkg
mkdir pkg
pkgbuild --identifier org.mlpack.pkg \
    --root ./res \
    --version 2.0.1 \
    --ownership recommended \
    pkg/mlpacktools.pkg

productbuild --distribution distribution.plist \
    --resources docs \
    --package-path pkg \
    --version 2.0.1 \
    mlpack.pkg