
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_rann_ra_search.hpp:

Program Listing for File ra_search.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_rann_ra_search.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/rann/ra_search.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     title={{Rank-Approximate Nearest Neighbor Search: Retaining Meaning and
         Speed in High Dimensions}},
     author={{Ram, P. and Lee, D. and Ouyang, H. and Gray, A. G.}},
     booktitle={{Advances of Neural Information Processing Systems}},
     year={2009}
   }
   
   #ifndef MLPACK_METHODS_RANN_RA_SEARCH_HPP
   #define MLPACK_METHODS_RANN_RA_SEARCH_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include <mlpack/core/tree/binary_space_tree.hpp>
   
   #include <mlpack/core/metrics/lmetric.hpp>
   #include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>
   
   #include "ra_query_stat.hpp"
   #include "ra_util.hpp"
   
   namespace mlpack {
   namespace neighbor {
   
   // Forward declaration.
   template<template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType>
   class LeafSizeRAWrapper;
   
   template<typename SortPolicy = NearestNeighborSort,
            typename MetricType = metric::EuclideanDistance,
            typename MatType = arma::mat,
            template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType = tree::KDTree>
   class RASearch
   {
    public:
     typedef TreeType<MetricType, RAQueryStat<SortPolicy>, MatType> Tree;
   
     RASearch(MatType referenceSet,
              const bool naive = false,
              const bool singleMode = false,
              const double tau = 5,
              const double alpha = 0.95,
              const bool sampleAtLeaves = false,
              const bool firstLeafExact = false,
              const size_t singleSampleLimit = 20,
              const MetricType metric = MetricType());
   
     RASearch(Tree* referenceTree,
              const bool singleMode = false,
              const double tau = 5,
              const double alpha = 0.95,
              const bool sampleAtLeaves = false,
              const bool firstLeafExact = false,
              const size_t singleSampleLimit = 20,
              const MetricType metric = MetricType());
   
     RASearch(const bool naive = false,
              const bool singleMode = false,
              const double tau = 5,
              const double alpha = 0.95,
              const bool sampleAtLeaves = false,
              const bool firstLeafExact = false,
              const size_t singleSampleLimit = 20,
              const MetricType metric = MetricType());
   
     ~RASearch();
   
     void Train(MatType referenceSet);
   
     void Train(Tree* referenceTree);
   
     void Search(const MatType& querySet,
                 const size_t k,
                 arma::Mat<size_t>& neighbors,
                 arma::mat& distances);
   
     void Search(Tree* queryTree,
                 const size_t k,
                 arma::Mat<size_t>& neighbors,
                 arma::mat& distances);
   
     void Search(const size_t k,
                 arma::Mat<size_t>& neighbors,
                 arma::mat& distances);
   
     void ResetQueryTree(Tree* queryTree) const;
   
     const MatType& ReferenceSet() const { return *referenceSet; }
   
     bool Naive() const { return naive; }
     bool& Naive() { return naive; }
   
     bool SingleMode() const { return singleMode; }
     bool& SingleMode() { return singleMode; }
   
     double Tau() const { return tau; }
     double& Tau() { return tau; }
   
     double Alpha() const { return alpha; }
     double& Alpha() { return alpha; }
   
     bool SampleAtLeaves() const { return sampleAtLeaves; }
     bool& SampleAtLeaves() { return sampleAtLeaves; }
   
     bool FirstLeafExact() const { return firstLeafExact; }
     bool& FirstLeafExact() { return firstLeafExact; }
   
     size_t SingleSampleLimit() const { return singleSampleLimit; }
     size_t& SingleSampleLimit() { return singleSampleLimit; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     std::vector<size_t> oldFromNewReferences;
     Tree* referenceTree;
     const MatType* referenceSet;
   
     bool treeOwner;
     bool setOwner;
   
     bool naive;
     bool singleMode;
   
     double tau;
     double alpha;
     bool sampleAtLeaves;
     bool firstLeafExact;
     size_t singleSampleLimit;
   
     MetricType metric;
   
     friend class LeafSizeRAWrapper<TreeType>;
   }; // class RASearch
   
   } // namespace neighbor
   } // namespace mlpack
   
   // Include implementation.
   #include "ra_search_impl.hpp"
   
   // Include convenient typedefs.
   #include "ra_typedef.hpp"
   
   #endif
