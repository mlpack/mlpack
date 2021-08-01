
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_cover_tree_typedef.hpp:

Program Listing for File typedef.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_cover_tree_typedef.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/cover_tree/typedef.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_COVER_TREE_TYPEDEF_HPP
   #define MLPACK_CORE_TREE_COVER_TREE_TYPEDEF_HPP
   
   #include "cover_tree.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using StandardCoverTree = CoverTree<MetricType,
                                       StatisticType,
                                       MatType,
                                       FirstPointIsRoot>;
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
