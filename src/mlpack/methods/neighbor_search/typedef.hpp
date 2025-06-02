/**
 * @file methods/neighbor_search/typedef.hpp
 * @author Ryan Curtin
 *
 * Simple typedefs describing template instantiations of the NeighborSearch
 * class which are commonly used.  This is meant to be included by
 * neighbor_search.h but is a separate file for simplicity.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_NEIGHBOR_SEARCH_TYPEDEF_HPP
#define MLPACK_NEIGHBOR_SEARCH_TYPEDEF_HPP

// In case someone included this directly.
#include "neighbor_search.hpp"

#include <mlpack/core/distances/lmetric.hpp>

#include "sort_policies/nearest_neighbor_sort.hpp"
#include "sort_policies/furthest_neighbor_sort.hpp"

namespace mlpack {

/**
 * The KNN class is the k-nearest-neighbors method.  It returns L2 distances
 * (Euclidean distances) for each of the k nearest neighbors.
 */
using KNN = NeighborSearch<NearestNeighborSort, EuclideanDistance>;

/**
 * The KFN class is the k-furthest-neighbors method.  It returns L2 distances
 * (Euclidean distances) for each of the k furthest neighbors.
 */
using KFN = NeighborSearch<FurthestNeighborSort, EuclideanDistance>;

/**
 * The DefeatistKNN class is the k-nearest-neighbors method considering
 * defeatist search. It returns L2 distances (Euclidean distances) for each of
 * the k nearest neighbors found.
 * @tparam TreeType The tree type to use; must adhere to the TreeType API,
 *     and implement Defeatist Traversers.
 */
template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType = SPTree>
using DefeatistKNN = NeighborSearch<
    NearestNeighborSort,
    EuclideanDistance,
    arma::mat,
    TreeType,
    TreeType<EuclideanDistance,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>::template DefeatistDualTreeTraverser,
    TreeType<EuclideanDistance,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>::template DefeatistSingleTreeTraverser>;

/**
 * The SpillKNN class is the k-nearest-neighbors method considering defeatist
 * search on SPTree.  It returns L2 distances (Euclidean distances) for each of
 * the k nearest neighbors found.
 */
using SpillKNN = DefeatistKNN<SPTree>;

} // namespace mlpack

#endif
