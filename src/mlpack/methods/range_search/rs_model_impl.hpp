/**
 * @file rs_model_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of Serialize() and inline functions for RSModel.
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

namespace mlpack {
namespace range {

//! Monochromatic range search on the given RSType instance.
template<typename RSType>
void MonoSearchVisitor::operator()(RSType* rs) const
{
  if (rs)
    return rs->Search(range, neighbors, distances);
  throw std::runtime_error("no range search model initialized");
}

//! Save parameters for bichromatic range search.
BiSearchVisitor::BiSearchVisitor(const arma::mat& querySet,
                                 const math::Range& range,
                                 std::vector<std::vector<size_t>>& neighbors,
                                 std::vector<std::vector<double>>& distances,
                                 const size_t leafSize):
    querySet(querySet),
    range(range),
    neighbors(neighbors),
    distances(distances),
    leafSize(leafSize)
{}

//! Default Bichromatic range search on the given RSType instance.
template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void BiSearchVisitor::operator()(RSTypeT<TreeType>* rs) const
{
  if (rs)
    return rs->Search(querySet, range, neighbors, distances);
  throw std::runtime_error("no range search model initialized");
}

//! Bichromatic range search on the given RSType specialized for KDTrees.
void BiSearchVisitor::operator()(RSTypeT<tree::KDTree>* rs) const
{
  if (rs)
    return SearchLeaf(rs);
  throw std::runtime_error("no range search model initialized");
}

//! Bichromatic range search on the given RSType specialized for BallTrees.
void BiSearchVisitor::operator()(RSTypeT<tree::BallTree>* rs) const
{
  if (rs)
    return SearchLeaf(rs);
  throw std::runtime_error("no range search model initialized");
}

//! Bichromatic range search specialized for Ocrees.
void BiSearchVisitor::operator()(RSTypeT<tree::Octree>* rs) const
{
  if (rs)
    return SearchLeaf(rs);
  throw std::runtime_error("no range search model initialized");
}

//! Bichromatic range search on the given RSType considering the leafSize.
template<typename RSType>
void BiSearchVisitor::SearchLeaf(RSType* rs) const
{
  if (!rs->Naive() && !rs->SingleMode())
  {
    // Build a second tree and search.
    Timer::Start("tree_building");
    Log::Info << "Building query tree..." << std::endl;
    std::vector<size_t> oldFromNewQueries;
    typename RSType::Tree queryTree(std::move(querySet), oldFromNewQueries,
        leafSize);
    Log::Info << "Tree built." << std::endl;
    Timer::Stop("tree_building");

    std::vector<std::vector<size_t>> neighborsOut;
    std::vector<std::vector<double>> distancesOut;
    rs->Search(&queryTree, range, neighborsOut, distancesOut);

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
    rs->Search(querySet, range, neighbors, distances);
}

//! Save parameters for Train.
TrainVisitor::TrainVisitor(arma::mat&& referenceSet,
                           const size_t leafSize) :
    referenceSet(std::move(referenceSet)),
    leafSize(leafSize)
{}

//! Default Train on the given RSType instance.
template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void TrainVisitor::operator()(RSTypeT<TreeType>* rs) const
{
  if (rs)
    return rs->Train(std::move(referenceSet));
  throw std::runtime_error("no range search model initialized");
}

//! Train on the given RSType specialized for KDTrees.
void TrainVisitor::operator()(RSTypeT<tree::KDTree>* rs) const
{
  if (rs)
    return TrainLeaf(rs);
  throw std::runtime_error("no range search model initialized");
}

//! Train on the given RSType specialized for BallTrees.
void TrainVisitor::operator()(RSTypeT<tree::BallTree>* rs) const
{
  if (rs)
    return TrainLeaf(rs);
  throw std::runtime_error("no range search model initialized");
}

//! Train specialized for Octrees.
void TrainVisitor::operator()(RSTypeT<tree::Octree>* rs) const
{
  if (rs)
    return TrainLeaf(rs);
  throw std::runtime_error("no range search model initialized");
}

//! Train on the given RSType considering the leafSize.
template<typename RSType>
void TrainVisitor::TrainLeaf(RSType* rs) const
{
  if (rs->Naive())
    rs->Train(std::move(referenceSet));
  else
  {
    std::vector<size_t> oldFromNewReferences;
    typename RSType::Tree* tree =
        new typename RSType::Tree(std::move(referenceSet), oldFromNewReferences,
        leafSize);
    rs->Train(tree);

    // Give the model ownership of the tree and the mappings.
    rs->treeOwner = true;
    rs->oldFromNewReferences = std::move(oldFromNewReferences);
  }
}

//! Expose the referenceSet of the given RSType.
template<typename RSType>
const arma::mat& ReferenceSetVisitor::operator()(RSType* rs) const
{
  if (rs)
    return rs->ReferenceSet();
  throw std::runtime_error("no range search model initialized");
}

//! For cleaning memory
template<typename RSType>
void DeleteVisitor::operator()(RSType* rs) const
{
  if (rs)
    delete rs;
}

//! Save parameters for serializing
template<typename Archive>
SerializeVisitor<Archive>::SerializeVisitor(Archive& ar,
                                            const std::string& name) :
    ar(ar),
    name(name)
{}

//! Serializes the given RSType instance.
template<typename Archive>
template<typename RSType>
void SerializeVisitor<Archive>::operator()(RSType* rs) const
{
  ar & data::CreateNVP(rs, name);
}

//! Return whether single mode enabled
template<typename RSType>
bool& SingleModeVisitor::operator()(RSType* rs) const
{
 if (rs)
   return rs->SingleMode();
 throw std::runtime_error("no range search model initialized");
}

//! Exposes Naive() function of given RSType
template<typename RSType>
bool& NaiveVisitor::operator()(RSType* rs) const
{
 if (rs)
   return rs->Naive();
 throw std::runtime_error("no range search model initialized");
}

// Serialize the model.
template<typename Archive>
void RSModel::Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(treeType, "treeType");
  ar & CreateNVP(randomBasis, "randomBasis");
  ar & CreateNVP(q, "q");

  // This should never happen, but just in case...
  if (Archive::is_loading::value)
    boost::apply_visitor(DeleteVisitor(), rSearch);

  // We'll only need to serialize one of the model objects, based on the type.
  const std::string& name = RSModelName::Name();
  SerializeVisitor<Archive> s(ar, name);
  boost::apply_visitor(s, rSearch);
}

inline const arma::mat& RSModel::Dataset() const
{
  return boost::apply_visitor(ReferenceSetVisitor(), rSearch);
}

inline bool RSModel::SingleMode() const
{
  return boost::apply_visitor(SingleModeVisitor(), rSearch);
}

inline bool& RSModel::SingleMode()
{
  return boost::apply_visitor(SingleModeVisitor(), rSearch);
}

inline bool RSModel::Naive() const
{
  return boost::apply_visitor(NaiveVisitor(), rSearch);
}

inline bool& RSModel::Naive()
{
  return boost::apply_visitor(NaiveVisitor(), rSearch);
}

} // namespace range
} // namespace mlpack

#endif
