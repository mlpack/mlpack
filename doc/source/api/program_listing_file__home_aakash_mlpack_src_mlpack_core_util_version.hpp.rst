
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_version.hpp:

Program Listing for File version.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_version.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/version.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTIL_VERSION_HPP
   #define MLPACK_CORE_UTIL_VERSION_HPP
   
   #include <string>
   
   // The version of mlpack.  If this is a git repository, this will be a version
   // with higher number than the most recent release.
   #define MLPACK_VERSION_MAJOR 3
   #define MLPACK_VERSION_MINOR 4
   #define MLPACK_VERSION_PATCH 3
   
   // The name of the version (for use by --version).
   namespace mlpack {
   namespace util {
   
   std::string GetVersion();
   
   } // namespace util
   } // namespace mlpack
   
   #endif
