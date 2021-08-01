
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_minimal_splits_number_sweep.hpp:

Program Listing for File minimal_splits_number_sweep.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_minimal_splits_number_sweep.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/minimal_splits_number_sweep.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_SPLITS_NUMBER_SWEEP_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_SPLITS_NUMBER_SWEEP_HPP
   
   namespace mlpack {
   namespace tree {
   
   template<typename SplitPolicy>
   class MinimalSplitsNumberSweep
   {
    public:
     template<typename>
     struct SweepCost
     {
       typedef size_t type;
     };
   
     template<typename TreeType>
     static size_t SweepNonLeafNode(
         const size_t axis,
         const TreeType* node,
         typename TreeType::ElemType& axisCut);
   
     template<typename TreeType>
     static size_t SweepLeafNode(
         const size_t axis,
         const TreeType* node,
         typename TreeType::ElemType& axisCut);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation
   #include "minimal_splits_number_sweep_impl.hpp"
   
   #endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_SPLITS_NUMBER_SWEEP_HPP
   
   
