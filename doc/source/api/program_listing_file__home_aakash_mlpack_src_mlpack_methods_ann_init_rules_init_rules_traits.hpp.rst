
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_init_rules_traits.hpp:

Program Listing for File init_rules_traits.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_init_rules_traits.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/init_rules/init_rules_traits.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_INIT_RULES_INIT_RULES_TRAITS_HPP
   #define MLPACK_METHODS_ANN_INIT_RULES_INIT_RULES_TRAITS_HPP
   
   namespace mlpack {
   namespace ann {
   
   template<typename InitRuleType>
   class InitTraits
   {
    public:
     static const bool UseLayer = true;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
