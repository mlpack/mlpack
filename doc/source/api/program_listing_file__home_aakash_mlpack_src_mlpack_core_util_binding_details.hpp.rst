
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_binding_details.hpp:

Program Listing for File binding_details.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_binding_details.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/binding_details.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTIL_BINDING_DETAILS_HPP
   #define MLPACK_CORE_UTIL_BINDING_DETAILS_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "program_doc.hpp"
   
   namespace mlpack {
   namespace util {
   
   struct BindingDetails
   {
     std::string programName;
     std::string shortDescription;
     std::function<std::string()> longDescription;
     std::vector<std::function<std::string()>> example;
     std::vector<std::pair<std::string, std::string>> seeAlso;
   };
   
   } // namespace util
   } // namespace mlpack
   
   #endif
