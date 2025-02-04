This directory contains some utility scripts for mlpack, meant to be used during
the development and release process.  All scripts should be run from the root
directory.

#### `build-docs.sh`

This will convert all the Markdown documentation in `doc/` into HTML in
`doc/html/`.  This script is used as part of the documentation deployment
process but also can be run manually.

The `kramdown` parser with the `parser-gfm` and `rouge` extensions installed is
necessary, as are the `tidy`, `checklink`, and `linkchecker` HTML checking
packages.

```sh
scripts/build-docs.sh
```

This will build all the documentation and check for any broken links.  You can
then go to `doc/html/index.html` to browse it.

#### `test-docs.sh`

This can be used to compile and run C++ code blocks from Markdown documentation,
either for all Markdown files in a directory or for a specific Markdown file.

```sh
export CXX=g++
# Adapt the flags below to your setup.
export CXXFLAGS="-O3 -I/path/to/ensmallen/include/ -I/path/to/armadillo/include/ -I/path/to/cereal/include/"
export LDFLAGS="-fopenmp"
scripts/test-docs.sh doc/
```

That command first sets the compiler to use and the compilation flags, and then
all C++ snippets from all Markdown files in `doc/` will be compiled with those
flags.  Each snippet is enclosed in an `int main() { }` function, and
`mlpack.hpp` is included.

All snippets are extracted into `doc/build/`.

It is also possible to run for just one file:

```
export CXX=g++
# Adapt the flags below to your setup.
export CXXFLAGS="-O3 -I/path/to/ensmallen/include/ -I/path/to/armadillo/include/ -I/path/to/cereal/include/"
export LDFLAGS="-fopenmp"
scripts/test-docs.sh doc/user/core.md
```

When running for just one file, the full code for each snippet will be printed,
along with the output when the program is run.  When running for multiple files,
this will be omitted (it would be too much output!).

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
