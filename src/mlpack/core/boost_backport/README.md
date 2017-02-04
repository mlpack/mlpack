The files in this directory are taken from Boost 1.56.0 and Boost 1.61.0 in
order to backport:

 * Serialization support for unordered_map (added in Boost 1.56).
 * Trigamma and polygamma function calculation for the gamma_distribution.hpp
   file (added in Boost 1.58.0, files taken from Boost 1.61.0)

These files are licensed under the Boost Software License, available in
LICENSE.txt in this directory.

If you want a copy of mlpack without a dependence on the Boost Software License,
then you will need to

 * remove this entire directory
 * remove the line "boost_backport" from src/mlpack/core/CMakeLists.txt
 * change the line "find_package(Boost x.yy" in the root CMakeLists.txt so that
   x is 1 and yy is at least 58.  (That is, make mlpack require Boost 1.58 or
   newer).
