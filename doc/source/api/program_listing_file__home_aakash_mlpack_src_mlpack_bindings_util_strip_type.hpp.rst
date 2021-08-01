
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_util_strip_type.hpp:

Program Listing for File strip_type.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_util_strip_type.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/util/strip_type.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_UTIL_STRIP_TYPE_HPP
   #define MLPACK_BINDINGS_UTIL_STRIP_TYPE_HPP
   
   namespace mlpack {
   namespace util {
   
   inline std::string StripType(std::string cppType)
   {
     // Basically what we need to do is strip any '<' (template bits) from the
     // type.  We'll try first by removing any instances of <>.
     const size_t loc = cppType.find("<>");
     if (loc != std::string::npos)
       cppType.replace(loc, 2, "");
   
     // Let's just replace any invalid characters with valid '_' characters.
     std::replace(cppType.begin(), cppType.end(), '<', '_');
     std::replace(cppType.begin(), cppType.end(), '>', '_');
     std::replace(cppType.begin(), cppType.end(), ' ', '_');
     std::replace(cppType.begin(), cppType.end(), ',', '_');
   
     return cppType;
   }
   
   } // namespace util
   } // namespace mlpack
   
   #endif
