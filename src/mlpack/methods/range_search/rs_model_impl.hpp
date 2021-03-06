/**
 * @file methods/range_search/rs_model_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of serialize() and inline functions for RSModel.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANGE_SEARCH_RS_MODEL_IMPL_HPP
#define MLPACK_METHODS_RANGE_SEARCH_RS_MODEL_IMPL_HPP

// In case it hasn't been included yet.
#include "rs_model.hpp"

#include <mlpack/core/math/random_basis.hpp>

namespace mlpack {
namespace range {

template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RSWrapper<TreeType>::Train(arma::mat&& referenceSet,
                                const size_t /* leafSize */)
{
  rs.Train(std::move(referenceSet));
}

template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RSWrapper<TreeType>::Search(arma::mat&& querySet,
                                 const math::Range& range,
                                 std::vector<std::vector<size_t>>& neighbors,
                                 std::vector<std::vector<double>>& distances,
                                 const size_t /* leafSize */)
{
  rs.Search(std::move(querySet), range, neighbors, distances);
}

template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RSWrapper<TreeType>::Search(const math::Range& range,
                                 std::vector<std::vector<size_t>>& neighbors,
                                 std::vector<std::vector<double>>& distances)
{
  rs.Search(range, neighbors, distances);
}

template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void LeafSizeRSWrapper<TreeType>::Train(arma::mat&& referenceSet,
                                        const size_t leafSize)
{
  if (rs.Naive())
  {
    rs.Train(std::move(referenceSet));
  }
  else
  {
    std::vector<size_t> oldFromNewReferences;
    typename decltype(rs)::Tree* tree =
        new typename decltype(rs)::Tree(std::move(referenceSet),
                                        oldFromNewReferences,
                                        leafSize);
    rs.Train(tree);

    // Give the model ownership of the tree and the mappings.
    rs.treeOwner = true;
    rs.oldFromNewReferences = std::move(oldFromNewReferences);
  }
}

template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void LeafSizeRSWrapper<TreeType>::Search(
    arma::mat&& querySet,
    const math::Range& range,
    std::vector<std::vector<size_t>>& neighbors,
    std::vector<std::vector<double>>& distances,
    const size_t leafSize)
{
  if (!rs.Naive() && !rs.SingleMode())
  {
    // Build a second tree and search.
    Timer::Start("tree_building");
    Log::Info << "Building query tree..." << std::endl;
    std::vector<size_t> oldFromNewQueries;
    typename decltype(rs)::Tree queryTree(std::move(querySet),
                                          oldFromNewQueries,
                                          leafSize);
    Log::Info << "Tree built." << std::endl;
    Timer::Stop("tree_building");

    std::vector<std::vector<size_t>> neighborsOut;
    std::vector<std::vector<double>> distancesOut;
    rs.Search(&queryTree, range, neighborsOut, distancesOut);

    // Remap the query points.
    neighbors.resize(queryTree.Dataset().n_cols);
    distances.resize(queryTree.Dataset().n_cols);
    for (size_t i = 0; i < queryTree.Dataset().n_cols; ++i)
    {
      neighbors[oldFromNewQueries[i]] = neighborsOut[i];
      distances[oldFromNewQueries[i]] = distancesOut[i];
    }
  }
  else
  {
    rs.Search(std::move(querySet), range, neighbors, distances);
  }
}

// Serialize the model.
template<typename Archive>
void RSModel::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(treeType));
  ar(CEREAL_NVP(randomBasis));
  ar(CEREAL_NVP(q));

  // This should never happen, but just in case...
  if (cereal::is_loading<Archive>())
    InitializeModel(false, false); // Values will be overwritten.

  // Avoid polymorphic serialization by explicitly serializing the correct type.
  switch (treeType)
  {
    case KD_TREE:
      {
        LeafSizeRSWrapper<tree::KDTree>& typedSearch =
            dynamic_cast<LeafSizeRSWrapper<tree::KDTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case COVER_TREE:
      {
        RSWrapper<tree::StandardCoverTree>& typedSearch =
            dynamic_cast<RSWrapper<tree::StandardCoverTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case R_TREE:
      {
        RSWrapper<tree::RTree>& typedSearch =
            dynamic_cast<RSWrapper<tree::RTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case R_STAR_TREE:
      {
        RSWrapper<tree::RStarTree>& typedSearch =
            dynamic_cast<RSWrapper<tree::RStarTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case BALL_TREE:
      {
        LeafSizeRSWrapper<tree::BallTree>& typedSearch =
            dynamic_cast<LeafSizeRSWrapper<tree::BallTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case X_TREE:
      {
        RSWrapper<tree::XTree>& typedSearch =
            dynamic_cast<RSWrapper<tree::XTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case HILBERT_R_TREE:
      {
        RSWrapper<tree::HilbertRTree>& typedSearch =
            dynamic_cast<RSWrapper<tree::HilbertRTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case R_PLUS_TREE:
      {
        RSWrapper<tree::RPlusTree>& typedSearch =
            dynamic_cast<RSWrapper<tree::RPlusTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case R_PLUS_PLUS_TREE:
      {
        RSWrapper<tree::RPlusPlusTree>& typedSearch =
            dynamic_cast<RSWrapper<tree::RPlusPlusTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case VP_TREE:
      {
        RSWrapper<tree::VPTree>& typedSearch =
            dynamic_cast<RSWrapper<tree::VPTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case RP_TREE:
      {
        RSWrapper<tree::RPTree>& typedSearch =
            dynamic_cast<RSWrapper<tree::RPTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case MAX_RP_TREE:
      {
        RSWrapper<tree::MaxRPTree>& typedSearch =
            dynamic_cast<RSWrapper<tree::MaxRPTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case UB_TREE:
      {
        RSWrapper<tree::UBTree>& typedSearch =
            dynamic_cast<RSWrapper<tree::UBTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case OCTREE:
      {
        LeafSizeRSWrapper<tree::Octree>& typedSearch =
            dynamic_cast<LeafSizeRSWrapper<tree::Octree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
  }
}

} // namespace range
} // namespace mlpack

#endif
