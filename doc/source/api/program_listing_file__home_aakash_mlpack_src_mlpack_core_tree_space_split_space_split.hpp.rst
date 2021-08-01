
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_space_split_space_split.hpp:

Program Listing for File space_split.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_space_split_space_split.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/space_split/space_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_SPILL_TREE_SPACE_SPLIT_HPP
   #define MLPACK_CORE_TREE_SPILL_TREE_SPACE_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "hyperplane.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType, typename MatType>
   class SpaceSplit
   {
    public:
     static bool GetProjVector(
         const bound::HRectBound<MetricType>& bound,
         const MatType& data,
         const arma::Col<size_t>& points,
         AxisParallelProjVector& projVector,
         double& midValue);
   
     template<typename BoundType>
     static bool GetProjVector(
         const BoundType& bound,
         const MatType& data,
         const arma::Col<size_t>& points,
         ProjVector& projVector,
         double& midValue);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "space_split_impl.hpp"
   
   #endif
