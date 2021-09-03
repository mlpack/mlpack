/**
 * @file methods/rann/ra_model_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the RAModel class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANN_RA_MODEL_IMPL_HPP
#define MLPACK_METHODS_RANN_RA_MODEL_IMPL_HPP

// In case it hasn't been included yet.
#include "ra_model.hpp"
#include <mlpack/core/math/random_basis.hpp>

namespace mlpack {
namespace neighbor {

template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RAWrapper<TreeType>::Train(util::Timers& timers,
                                arma::mat&& referenceSet,
                                const size_t /* leafSize */)
{
  if (!ra.Naive())
    timers.Start("tree_building");

  ra.Train(std::move(referenceSet));

  if (!ra.Naive())
    timers.Stop("tree_building");
}

template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RAWrapper<TreeType>::Search(util::Timers& timers,
                                 arma::mat&& querySet,
                                 const size_t k,
                                 arma::Mat<size_t>& neighbors,
                                 arma::mat& distances,
                                 const size_t /* leafSize */)
{
  if (!ra.Naive() && !ra.SingleMode())
  {
    // Build the tree manually, so we can time it.
    timers.Start("tree_building");
    typename decltype(ra)::Tree tree(std::move(querySet));
    timers.Stop("tree_building");

    timers.Start("computing_neighbors");
    ra.Search(&tree, k, neighbors, distances);
    timers.Stop("computing_neighbors");
  }
  else
  {
    timers.Start("computing_neighbors");
    ra.Search(querySet, k, neighbors, distances);
    timers.Stop("computing_neighbors");
  }
}

template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RAWrapper<TreeType>::Search(util::Timers& timers,
                                 const size_t k,
                                 arma::Mat<size_t>& neighbors,
                                 arma::mat& distances)
{
  timers.Start("computing_neighbors");
  ra.Search(k, neighbors, distances);
  timers.Stop("computing_neighbors");
}

template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void LeafSizeRAWrapper<TreeType>::Train(util::Timers& timers,
                                        arma::mat&& referenceSet,
                                        const size_t leafSize)
{
  // Build tree, if necessary.
  if (ra.Naive())
  {
    ra.Train(std::move(referenceSet));
  }
  else
  {
    timers.Start("tree_building");
    std::vector<size_t> oldFromNewReferences;
    typename decltype(ra)::Tree* tree =
        new typename decltype(ra)::Tree(std::move(referenceSet),
                                        oldFromNewReferences,
                                        leafSize);
    timers.Stop("tree_building");

    ra.Train(tree);

    // Give the model ownership of the tree and the mappings.
    ra.treeOwner = true;
    ra.oldFromNewReferences = std::move(oldFromNewReferences);
  }
}

template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void LeafSizeRAWrapper<TreeType>::Search(util::Timers& timers,
                                         arma::mat&& querySet,
                                         const size_t k,
                                         arma::Mat<size_t>& neighbors,
                                         arma::mat& distances,
                                         const size_t leafSize)
{
  if (!ra.Naive() && !ra.SingleMode())
  {
    // Build a second tree and search, taking the leaf size into account.
    timers.Start("tree_building");
    Log::Info << "Building query tree...."<< std::endl;
    std::vector<size_t> oldFromNewQueries;
    typename decltype(ra)::Tree queryTree(std::move(querySet),
                                          oldFromNewQueries,
                                          leafSize);
    Log::Info << "Tree built." << std::endl;
    timers.Stop("tree_building");

    arma::Mat<size_t> neighborsOut;
    arma::mat distancesOut;
    timers.Start("computing_neighbors");
    ra.Search(&queryTree, k, neighborsOut, distancesOut);
    timers.Stop("computing_neighbors");

    // Unmap the query points.
    distances.set_size(distancesOut.n_rows, distancesOut.n_cols);
    neighbors.set_size(neighborsOut.n_rows, neighborsOut.n_cols);
    for (size_t i = 0; i < oldFromNewQueries.size(); ++i)
    {
      neighbors.col(oldFromNewQueries[i]) = neighborsOut.col(i);
      distances.col(oldFromNewQueries[i]) = distancesOut.col(i);
    }
  }
  else
  {
    // Search without building a second tree.
    timers.Start("computing_neighbors");
    ra.Search(querySet, k, neighbors, distances);
    timers.Stop("computing_neighbors");
  }
}

template<typename Archive>
void RAModel::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(treeType));
  ar(CEREAL_NVP(randomBasis));
  ar(CEREAL_NVP(q));

  // This should never happen, but just in case, be clean with memory.
  if (cereal::is_loading<Archive>())
    InitializeModel(false, false); // Values will be overwritten.

  // Avoid polymorphic serialization by explicitly serializing the correct type.
  switch (treeType)
  {
    case KD_TREE:
      {
        LeafSizeRAWrapper<tree::KDTree>& typedSearch =
            dynamic_cast<LeafSizeRAWrapper<tree::KDTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case COVER_TREE:
      {
        RAWrapper<tree::StandardCoverTree>& typedSearch =
            dynamic_cast<RAWrapper<tree::StandardCoverTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case R_TREE:
      {
        RAWrapper<tree::RTree>& typedSearch =
            dynamic_cast<RAWrapper<tree::RTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case R_STAR_TREE:
      {
        RAWrapper<tree::RStarTree>& typedSearch =
            dynamic_cast<RAWrapper<tree::RStarTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case X_TREE:
      {
        RAWrapper<tree::XTree>& typedSearch =
            dynamic_cast<RAWrapper<tree::XTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case HILBERT_R_TREE:
      {
        RAWrapper<tree::HilbertRTree>& typedSearch =
            dynamic_cast<RAWrapper<tree::HilbertRTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case R_PLUS_TREE:
      {
        RAWrapper<tree::RPlusTree>& typedSearch =
            dynamic_cast<RAWrapper<tree::RPlusTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case R_PLUS_PLUS_TREE:
      {
        RAWrapper<tree::RPlusPlusTree>& typedSearch =
            dynamic_cast<RAWrapper<tree::RPlusPlusTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case UB_TREE:
      {
        LeafSizeRAWrapper<tree::UBTree>& typedSearch =
            dynamic_cast<LeafSizeRAWrapper<tree::UBTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case OCTREE:
      {
        LeafSizeRAWrapper<tree::Octree>& typedSearch =
            dynamic_cast<LeafSizeRAWrapper<tree::Octree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
  }
}

} // namespace neighbor
} // namespace mlpack

#endif
