
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_vantage_point_split.hpp:

Program Listing for File vantage_point_split.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_vantage_point_split.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/binary_space_tree/vantage_point_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_HPP
   #define MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/tree/perform_split.hpp>
   #include <mlpack/core/math/random.hpp>
   
   namespace mlpack {
   namespace tree  {
   
   template<typename BoundType,
            typename MatType = arma::mat,
            size_t MaxNumSamples = 100>
   class VantagePointSplit
   {
    public:
     typedef typename MatType::elem_type ElemType;
     typedef typename BoundType::MetricType MetricType;
     struct SplitInfo
     {
       arma::Col<ElemType> vantagePoint;
       ElemType mu;
       const MetricType* metric;
   
       SplitInfo() :
           mu(0),
           metric(NULL)
       { }
   
       template<typename VecType>
       SplitInfo(const MetricType& metric, const VecType& vantagePoint,
           ElemType mu) :
           vantagePoint(vantagePoint),
           mu(mu),
           metric(&metric)
       { }
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
       return split::PerformSplit<MatType, VantagePointSplit>(data, begin, count,
           splitInfo);
     }
   
     static size_t PerformSplit(MatType& data,
                                const size_t begin,
                                const size_t count,
                                const SplitInfo& splitInfo,
                                std::vector<size_t>& oldFromNew)
     {
       return split::PerformSplit<MatType, VantagePointSplit>(data, begin, count,
           splitInfo, oldFromNew);
     }
   
     template<typename VecType>
     static bool AssignToLeftNode(const VecType& point,
                                  const SplitInfo& splitInfo)
     {
       return (splitInfo.metric->Evaluate(splitInfo.vantagePoint, point) <
           splitInfo.mu);
     }
   
    private:
     static void SelectVantagePoint(const MetricType& metric,
                                    const MatType& data,
                                    const size_t begin,
                                    const size_t count,
                                    size_t& vantagePoint,
                                    ElemType& mu);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "vantage_point_split_impl.hpp"
   
   #endif  //  MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_HPP
