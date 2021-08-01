
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_to_lower.hpp:

Program Listing for File to_lower.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_to_lower.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/to_lower.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTIL_LOWER_STRING_HPP
   #define MLPACK_CORE_UTIL_LOWER_STRING_HPP
   
   namespace mlpack {
   namespace util {
   
   inline std::string ToLower(const std::string& input)
   {
     std::string output;
     std::transform(input.begin(), input.end(), std::back_inserter(output),
         [](unsigned char c){ return std::tolower(c); });
     return output;
   }
   
   } // namespace util
   } // namespace mlpack
   
   #endif
