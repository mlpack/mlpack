
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_typedef.hpp:

Program Listing for File typedef.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_typedef.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/neighbor_search/typedef.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_NEIGHBOR_SEARCH_TYPEDEF_H
   #define MLPACK_NEIGHBOR_SEARCH_TYPEDEF_H
   
   // In case someone included this directly.
   #include "neighbor_search.hpp"
   
   #include <mlpack/core/metrics/lmetric.hpp>
   
   #include "sort_policies/nearest_neighbor_sort.hpp"
   #include "sort_policies/furthest_neighbor_sort.hpp"
   
   namespace mlpack {
   namespace neighbor {
   
   typedef NeighborSearch<NearestNeighborSort, metric::EuclideanDistance> KNN;
   
   typedef NeighborSearch<FurthestNeighborSort, metric::EuclideanDistance> KFN;
   
   template<template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType = tree::SPTree>
   using DefeatistKNN = NeighborSearch<
       NearestNeighborSort,
       metric::EuclideanDistance,
       arma::mat,
       TreeType,
       TreeType<metric::EuclideanDistance,
           NeighborSearchStat<NearestNeighborSort>,
           arma::mat>::template DefeatistDualTreeTraverser,
       TreeType<metric::EuclideanDistance,
           NeighborSearchStat<NearestNeighborSort>,
           arma::mat>::template DefeatistSingleTreeTraverser>;
   
   typedef DefeatistKNN<tree::SPTree> SpillKNN;
   
   } // namespace neighbor
   } // namespace mlpack
   
   #endif
