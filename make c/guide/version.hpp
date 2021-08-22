/*! @page verinfo mlpack version information

@section vercode mlpack versions in code

mlpack provides a couple of convenience macros and functions to get the version
of mlpack.  More information (and straightforward code) can be found in
src/mlpack/core/util/version.hpp.

The following three macros provide major, minor, and patch versions of mlpack
(i.e. for mlpack-x.y.z, 'x' is the major version, 'y' is the minor version, and
'z' is the patch version):

@code
MLPACK_VERSION_MAJOR
MLPACK_VERSION_MINOR
MLPACK_VERSION_PATCH
@endcode

In addition, the function \c mlpack::util::GetVersion() returns the mlpack
version as a string (for instance, "mlpack 1.0.8").

@section verex mlpack executable versions

Each mlpack executable supports the \c --version (or \c -V ) option, which will
print the version of mlpack used.  If the version is not an official release but
instead from svn trunk, the version will be "mlpack trunk" (and may have a
revision number appended to "trunk").

*/
