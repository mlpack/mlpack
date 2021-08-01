
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_string_encoding_policies_policy_traits.hpp:

Program Listing for File policy_traits.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_string_encoding_policies_policy_traits.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/string_encoding_policies/policy_traits.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_STRING_ENCODING_POLICIES_POLICY_TRAITS_HPP
   #define MLPACK_CORE_DATA_STRING_ENCODING_POLICIES_POLICY_TRAITS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   
   template<class PolicyType>
   struct StringEncodingPolicyTraits
   {
     static const bool onePassEncoding = false;
   };
   
   } // namespace data
   } // namespace mlpack
   
   #endif
