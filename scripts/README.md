This directory contains some utility scripts for mlpack, meant to be used during
the development and release process.  All scripts should be run from the root
directory.

#### `build-docs.sh`

This will convert all the Markdown documentation in `doc/` into HTML in
`doc/html/`.  This script is used as part of the documentation deployment
process but also can be run manually.

The `kramdown` parser with the `parser-gfm` and `rouge` extensions installed is
necessary, as are the `tidy` and `checklink` HTML checking packages.

```
scripts/build-docs.sh
```

This will build all the documentation and check for any broken links.  You can
then go to `doc/html/index.html` to browse it.

#### `release-mlpack.sh`

Run this to open a pull request to release a new version of mlpack.
You need the Hub command-line tool installed (see https://hub.github.com/).

```
scripts/release-mlpack.sh rcurtin 4 3 0
```

This will use Hub as the user `rcurtin` to open a PR to release mlpack 4.3.0.

Be sure that `HISTORY.md` is up to date before running this!

#### `update-website-after-release.sh`

This is run automatically on mlpack.org after a new version of mlpack is
released.
