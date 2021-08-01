
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cereal_is_saving.hpp:

Program Listing for File is_saving.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cereal_is_saving.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cereal/is_saving.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CEREAL_IS_SAVING_HPP
   #define MLPACK_CORE_CEREAL_IS_SAVING_HPP
   
   #include <cereal/archives/binary.hpp>
   #include <cereal/archives/portable_binary.hpp>
   #include <cereal/archives/xml.hpp>
   #include <cereal/archives/json.hpp>
   
   namespace cereal {
   
   template<typename Archive>
   struct is_cereal_archive_saving
   {
     // Archive::is_saving is not implemented yet, so we can use std::is_same<>
     // to check if it is a loading archive.
     constexpr static bool value = std::is_same<Archive,
         cereal::BinaryOutputArchive>::value ||
   // #if (BINDING_TYPE != BINDING_TYPE_R)
         std::is_same<Archive, cereal::JSONOutputArchive>::value ||
   // #endif
         std::is_same<Archive, cereal::XMLOutputArchive>::value;
   };
   
   template<typename Archive>
   bool is_saving(
       const typename std::enable_if<
           is_cereal_archive_saving<Archive>::value, Archive>::type* = 0)
   {
     return true;
   }
   
   template<typename Archive>
   bool is_saving(
       const typename std::enable_if<
         !is_cereal_archive_saving<Archive>::value, Archive>::type* = 0)
   {
     return false;
   }
   
   } // namespace cereal
   
   #endif // CEREAL_IS_SAVING_HPP
