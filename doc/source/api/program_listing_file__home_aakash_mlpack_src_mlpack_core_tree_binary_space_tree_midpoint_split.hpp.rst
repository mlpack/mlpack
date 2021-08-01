
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_midpoint_split.hpp:

Program Listing for File midpoint_split.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_midpoint_split.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/binary_space_tree/midpoint_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_MIDPOINT_SPLIT_HPP
   #define MLPACK_CORE_TREE_BINARY_SPACE_TREE_MIDPOINT_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/tree/perform_split.hpp>
   
   namespace mlpack {
   namespace tree  {
   
   template<typename BoundType, typename MatType = arma::mat>
   class MidpointSplit
   {
    public:
     struct SplitInfo
     {
       size_t splitDimension;
       double splitVal;
     };
     static bool SplitNode(const BoundType& bound,
                           MatType& data,
                           const size_t begin,
                           const size_t count,
                           SplitInfo& splitInfo);
   
     static size_t PerformSplit(MatType& data,
                                const size_t begin,
                                const size_t count,
                                const SplitInfo& splitInfo)
     {
       return split::PerformSplit<MatType, MidpointSplit>(data, begin, count,
           splitInfo);
     }
   
     static size_t PerformSplit(MatType& data,
                                const size_t begin,
                                const size_t count,
                                const SplitInfo& splitInfo,
                                std::vector<size_t>& oldFromNew)
     {
       return split::PerformSplit<MatType, MidpointSplit>(data, begin, count,
           splitInfo, oldFromNew);
     }
   
     template<typename VecType>
     static bool AssignToLeftNode(const VecType& point,
                                  const SplitInfo& splitInfo)
     {
       return point[splitInfo.splitDimension] < splitInfo.splitVal;
     }
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "midpoint_split_impl.hpp"
   
   #endif
