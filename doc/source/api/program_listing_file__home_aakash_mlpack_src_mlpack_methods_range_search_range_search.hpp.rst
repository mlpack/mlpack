
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_range_search_range_search.hpp:

Program Listing for File range_search.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_range_search_range_search.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/range_search/range_search.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_HPP
   #define MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   #include <mlpack/core/tree/binary_space_tree.hpp>
   #include "range_search_stat.hpp"
   
   namespace mlpack {
   namespace range  {
   
   template<template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType>
   class LeafSizeRSWrapper;
   
   template<typename MetricType = metric::EuclideanDistance,
            typename MatType = arma::mat,
            template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType = tree::KDTree>
   class RangeSearch
   {
    public:
     typedef TreeType<MetricType, RangeSearchStat, MatType> Tree;
   
     RangeSearch(MatType referenceSet,
                 const bool naive = false,
                 const bool singleMode = false,
                 const MetricType metric = MetricType());
   
     RangeSearch(Tree* referenceTree,
                 const bool singleMode = false,
                 const MetricType metric = MetricType());
   
     RangeSearch(const bool naive = false,
                 const bool singleMode = false,
                 const MetricType metric = MetricType());
   
     RangeSearch(const RangeSearch& other);
   
     RangeSearch(RangeSearch&& other);
   
     RangeSearch& operator=(const RangeSearch& other);
   
     RangeSearch& operator=(RangeSearch&& other);
   
     ~RangeSearch();
   
     void Train(MatType referenceSet);
   
     void Train(Tree* referenceTree);
   
     void Search(const MatType& querySet,
                 const math::Range& range,
                 std::vector<std::vector<size_t>>& neighbors,
                 std::vector<std::vector<double>>& distances);
   
     void Search(Tree* queryTree,
                 const math::Range& range,
                 std::vector<std::vector<size_t>>& neighbors,
                 std::vector<std::vector<double>>& distances);
   
     void Search(const math::Range& range,
                 std::vector<std::vector<size_t>>& neighbors,
                 std::vector<std::vector<double>>& distances);
   
     bool SingleMode() const { return singleMode; }
     bool& SingleMode() { return singleMode; }
   
     bool Naive() const { return naive; }
     bool& Naive() { return naive; }
   
     size_t BaseCases() const { return baseCases; }
     size_t Scores() const { return scores; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
     const MatType& ReferenceSet() const { return *referenceSet; }
   
     Tree* ReferenceTree() { return referenceTree; }
   
    private:
     std::vector<size_t> oldFromNewReferences;
     Tree* referenceTree;
     const MatType* referenceSet;
   
     bool treeOwner;
   
     bool naive;
     bool singleMode;
   
     MetricType metric;
   
     size_t baseCases;
     size_t scores;
   
     friend class LeafSizeRSWrapper<TreeType>;
   };
   
   } // namespace range
   } // namespace mlpack
   
   // Include implementation.
   #include "range_search_impl.hpp"
   
   #endif
