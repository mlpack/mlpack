
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_minimal_coverage_sweep.hpp:

Program Listing for File minimal_coverage_sweep.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_minimal_coverage_sweep.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/minimal_coverage_sweep.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_COVERAGE_SWEEP_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_COVERAGE_SWEEP_HPP
   
   namespace mlpack {
   namespace tree {
   
   template<typename SplitPolicy>
   class MinimalCoverageSweep
   {
    public:
     template<typename TreeType>
     struct SweepCost
     {
       typedef typename TreeType::ElemType type;
     };
   
     template<typename TreeType>
     static typename TreeType::ElemType SweepNonLeafNode(
         const size_t axis,
         const TreeType* node,
         typename TreeType::ElemType& axisCut);
   
     template<typename TreeType>
     static typename TreeType::ElemType SweepLeafNode(
         const size_t axis,
         const TreeType* node,
         typename TreeType::ElemType& axisCut);
   
     template<typename TreeType, typename ElemType>
     static bool CheckNonLeafSweep(const TreeType* node,
                                   const size_t cutAxis,
                                   const ElemType cut);
   
     template<typename TreeType, typename ElemType>
     static bool CheckLeafSweep(const TreeType* node,
                                const size_t cutAxis,
                                const ElemType cut);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation
   #include "minimal_coverage_sweep_impl.hpp"
   
   #endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_COVERAGE_SWEEP_HPP
   
