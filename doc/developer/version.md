# mlpack versions in code

mlpack provides a couple of convenience macros and functions to get the version
of mlpack.  More information (and straightforward code) can be found in
`src/mlpack/core/util/version.hpp`.

The following three macros provide major, minor, and patch versions of mlpack
(i.e. for `mlpack-x.y.z`, `x` is the major version, `y` is the minor version,
and `z` is the patch version):

```c++
MLPACK_VERSION_MAJOR
MLPACK_VERSION_MINOR
MLPACK_VERSION_PATCH
```

In addition, the function `mlpack::util::GetVersion()` returns the mlpack
version as a string (for instance, `"mlpack 1.0.8"`).

## mlpack command-line program versions

Each mlpack command-line program supports the `--version` (or `-V`) option,
which will print the version of mlpack used.  If the version is not an official
release but instead from git, the version will be `mlpack git` (and will have a
git revision SHA appended to `git`).
