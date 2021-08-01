
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_space_split_midpoint_space_split.hpp:

Program Listing for File midpoint_space_split.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_space_split_midpoint_space_split.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/space_split/midpoint_space_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_SPILL_TREE_MIDPOINT_SPACE_SPLIT_HPP
   #define MLPACK_CORE_TREE_SPILL_TREE_MIDPOINT_SPACE_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "hyperplane.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType, typename MatType>
   class MidpointSpaceSplit
   {
    public:
     template<typename HyperplaneType>
     static bool SplitSpace(
         const typename HyperplaneType::BoundType& bound,
         const MatType& data,
         const arma::Col<size_t>& points,
         HyperplaneType& hyp);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "midpoint_space_split_impl.hpp"
   
   #endif
