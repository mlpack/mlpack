
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_statistic.hpp:

Program Listing for File statistic.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_statistic.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/statistic.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_STATISTIC_HPP
   #define MLPACK_CORE_TREE_STATISTIC_HPP
   
   namespace mlpack {
   namespace tree {
   
   class EmptyStatistic
   {
    public:
     EmptyStatistic() { }
     ~EmptyStatistic() { }
   
     template<typename TreeType>
     EmptyStatistic(TreeType& /* node */) { }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */)
     { }
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif // MLPACK_CORE_TREE_STATISTIC_HPP
