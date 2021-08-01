
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cereal_is_loading.hpp:

Program Listing for File is_loading.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cereal_is_loading.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cereal/is_loading.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CEREAL_IS_LOADING_HPP
   #define MLPACK_CORE_CEREAL_IS_LOADING_HPP
   
   #include <cereal/archives/binary.hpp>
   #include <cereal/archives/portable_binary.hpp>
   #include <cereal/archives/xml.hpp>
   #include <cereal/archives/json.hpp>
   
   namespace cereal {
   
   template<typename Archive>
   struct is_cereal_archive
   {
     // Archive::is_loading is not implemented yet, so we can use std::is_same<>
     // to check if it is a loading archive.
     constexpr static bool value = std::is_same<Archive,
         cereal::BinaryInputArchive>::value ||
   // #if (BINDING_TYPE != BINDING_TYPE_R)
         std::is_same<Archive, cereal::JSONInputArchive>::value ||
   // #endif
         std::is_same<Archive, cereal::XMLInputArchive>::value;
   };
   
   template<typename Archive>
   bool is_loading(
       const typename std::enable_if<
           is_cereal_archive<Archive>::value, Archive>::type* = 0)
   {
     return true;
   }
   
   template<typename Archive>
   bool is_loading(
       const typename std::enable_if<
         !is_cereal_archive<Archive>::value, Archive>::type* = 0)
   {
     return false;
   }
   
   } // namespace cereal
   
   #endif // CEREAL_IS_LOADING_HPP
