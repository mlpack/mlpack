
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_typedef.hpp:

Program Listing for File typedef.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_typedef.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hoeffding_trees/typedef.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HOEFFDING_TREES_TYPEDEF_HPP
   #define MLPACK_METHODS_HOEFFDING_TREES_TYPEDEF_HPP
   
   #include "streaming_decision_tree.hpp"
   #include "hoeffding_tree.hpp"
   
   namespace mlpack {
   namespace tree {
   
   typedef StreamingDecisionTree<HoeffdingTree<>> HoeffdingTreeType;
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
