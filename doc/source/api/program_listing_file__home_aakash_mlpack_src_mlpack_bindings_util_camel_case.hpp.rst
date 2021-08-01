
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_util_camel_case.hpp:

Program Listing for File camel_case.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_util_camel_case.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/util/camel_case.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_UTIL_CAMEL_CASE_HPP
   #define MLPACK_BINDINGS_UTIL_CAMEL_CASE_HPP
   
   namespace mlpack {
   namespace util {
   
   inline std::string CamelCase(std::string s, bool lower)
   {
     if (!lower)
       s[0] = std::toupper(s[0]);
     else
       s[0] = std::tolower(s[0]);
     size_t n = s.length();
     size_t resInd = 0;
     for (size_t i = 0; i < n; i++)
     {
       // Check for spaces in the sentence.
       if (s[i] == '_')
       {
         // Conversion into upper case.
         s[i + 1] = toupper(s[i + 1]);
         continue;
       }
       // If not space, copy character.
       else
         s[resInd++] = s[i];
     }
     // Return string to main.
     return s.substr(0, resInd);
   }
   
   } // namespace util
   } // namespace mlpack
   
   #endif
