
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_rp_tree_mean_split.hpp:

Program Listing for File rp_tree_mean_split.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_rp_tree_mean_split.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/binary_space_tree/rp_tree_mean_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MEAN_SPLIT_HPP
   #define MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MEAN_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "rp_tree_max_split.hpp"
   #include <mlpack/core/tree/perform_split.hpp>
   #include <mlpack/core/math/lin_alg.hpp>
   
   namespace mlpack {
   namespace tree  {
   
   template<typename BoundType, typename MatType = arma::mat>
   class RPTreeMeanSplit
   {
    public:
     typedef typename MatType::elem_type ElemType;
     struct SplitInfo
     {
       arma::Col<ElemType> direction;
       arma::Col<ElemType> mean;
       ElemType splitVal;
       bool meanSplit;
     };
   
     static bool SplitNode(const BoundType& /* bound */,
                           MatType& data,
                           const size_t begin,
                           const size_t count,
                           SplitInfo& splitInfo);
   
     static size_t PerformSplit(MatType& data,
                                const size_t begin,
                                const size_t count,
                                const SplitInfo& splitInfo)
     {
       return split::PerformSplit<MatType, RPTreeMeanSplit>(data, begin, count,
           splitInfo);
     }
   
     static size_t PerformSplit(MatType& data,
                                const size_t begin,
                                const size_t count,
                                const SplitInfo& splitInfo,
                                std::vector<size_t>& oldFromNew)
     {
       return split::PerformSplit<MatType, RPTreeMeanSplit>(data, begin, count,
           splitInfo, oldFromNew);
     }
   
     template<typename VecType>
     static bool AssignToLeftNode(const VecType& point, const SplitInfo& splitInfo)
     {
       if (splitInfo.meanSplit)
         return arma::dot(point - splitInfo.mean, point - splitInfo.mean) <=
             splitInfo.splitVal;
   
       return (arma::dot(point, splitInfo.direction) <= splitInfo.splitVal);
     }
   
    private:
     static ElemType GetAveragePointDistance(MatType& data,
                                             const arma::uvec& samples);
   
     static bool GetDotMedian(const MatType& data,
                              const arma::uvec& samples,
                              const arma::Col<ElemType>& direction,
                              ElemType& splitVal);
   
     static bool GetMeanMedian(const MatType& data,
                               const arma::uvec& samples,
                               arma::Col<ElemType>& mean,
                               ElemType& splitVal);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "rp_tree_mean_split_impl.hpp"
   
   #endif // MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MEAN_SPLIT_HPP
