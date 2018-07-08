/**
 * @file ra_model_impl.hpp
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
#include <boost/serialization/variant.hpp>

namespace mlpack {
namespace neighbor {

//! Monochromatic search for the given RAType instance.
template<typename RAType>
void MonoSearchVisitor::operator()(RAType* ra) const
{
  if (ra)
    return ra->Search(k, neighbors, distances);
  throw std::runtime_error("no rank-approximate model initialized");
}

//! Save the parameters for the rank-approximate search.
template<typename SortPolicy>
BiSearchVisitor<SortPolicy>::BiSearchVisitor(const arma::mat& querySet,
                                 const size_t k,
                                 arma::Mat<size_t>& neighbors,
                                 arma::mat& distances,
                                 const size_t leafSize) :
    querySet(querySet),
    k(k),
    neighbors(neighbors),
    distances(distances),
    leafSize(leafSize)
{};

//! Default Bichromatic search on the given RAType instance.
template<typename SortPolicy>
template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void BiSearchVisitor<SortPolicy>::operator()(RATypeT<TreeType>* ra) const
{
  if (ra)
    return ra->Search(querySet, k, neighbors, distances);
  throw std::runtime_error("no rank-approximate model initialized");
}

//! Bichromatic search on the given RAType specialized for KDTrees.
template<typename SortPolicy>
void BiSearchVisitor<SortPolicy>::operator()(RATypeT<tree::KDTree>* ra) const
{
  if (ra)
    return SearchLeaf(ra);
  throw std::runtime_error("no rank-approximate search model initialized");
}

//! Bichromatic search on the given RAType specialized for Octrees.
template<typename SortPolicy>
void BiSearchVisitor<SortPolicy>::operator()(RATypeT<tree::Octree>* ra) const
{
  if (ra)
    return SearchLeaf(ra);
  throw std::runtime_error("no rank-approximate search model initialized");
}

//! Bichromatic search on the given RAType considering the leafSize.
template<typename SortPolicy>
template<typename RAType>
void BiSearchVisitor<SortPolicy>::SearchLeaf(RAType* ra) const
{
  if (!ra->Naive() && !ra->SingleMode())
  {
    // Build a second tree and search
    Timer::Start("tree_building");
    Log::Info << "Building query tree...."<< std::endl;
    std::vector<size_t> oldFromNewQueries;
    typename RAType::Tree queryTree(std::move(querySet), oldFromNewQueries,
        leafSize);
    Log::Info << "Tree Built." << std::endl;
    Timer::Stop("tree_building");

    arma::Mat<size_t> neighborsOut;
    arma::mat distancesOut;
    ra->Search(&queryTree, k, neighborsOut, distancesOut);

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
    ra->Search(querySet, k, neighbors, distances);
  }
}

//! Save parameters for the Train.
template<typename SortPolicy>
TrainVisitor<SortPolicy>::TrainVisitor(arma::mat&& referenceSet,
                                       const size_t leafSize) :
    referenceSet(std::move(referenceSet)),
    leafSize(leafSize)
{};

//! Default Train on the given RAType instance.
template<typename SortPolicy>
template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void TrainVisitor<SortPolicy>::operator()(RATypeT<TreeType>* ra) const
{
  if (ra)
    return ra->Train(std::move(referenceSet));
  throw std::runtime_error("no rank-approximate search model initialized");
}

//! Train on the given RAType specialized for KDTrees.
template<typename SortPolicy>
void TrainVisitor<SortPolicy>::operator()(RATypeT<tree::KDTree>* ra) const
{
  if (ra)
    return TrainLeaf(ra);
  throw std::runtime_error("no rank-approximate search model initialized");
}

//! Train on the given RAType specialized for Octrees.
template<typename SortPolicy>
void TrainVisitor<SortPolicy>::operator()(RATypeT<tree::Octree>* ra) const
{
  if (ra)
    return TrainLeaf(ra);
  throw std::runtime_error("no rank-approximate search model is initialized");
}

//! Train on the given RAType considering the leafSize.
template<typename SortPolicy>
template<typename RAType>
void TrainVisitor<SortPolicy>::TrainLeaf(RAType* ra) const
{
  // Build tree, if necessary
  if (ra->Naive())
  {
    ra->Train(std::move(referenceSet));
  }
  else
  {
    std::vector<size_t> oldFromNewReferences;
    typename RAType::Tree* tree =
        new typename RAType::Tree(std::move(referenceSet), oldFromNewReferences,
        leafSize);
    ra->Train(tree);

    // Give the model ownership of the tree and the mappings.
    ra->treeOwner = true;
    ra->oldFromNewReferences = std::move(oldFromNewReferences);
  }
}

//! Exposes the SingleSampleLimit() method of the given RAType.
template<typename RAType>
size_t& SingleSampleLimitVisitor::operator()(RAType* ra) const
{
  if (ra)
    return ra->SingleSampleLimit();
  throw std::runtime_error("no rank-approximate search model is initialized");
}

//! Exposes the FirstLeafExact() method of the given RAType.
template<typename RAType>
bool& FirstLeafExactVisitor::operator()(RAType* ra) const
{
  if (ra)
    return ra->FirstLeafExact();
  throw std::runtime_error("no rank-approximate search model is initialized");
}

//! Exposes the SampleAtLeaves() method of the given RAType.
template<typename RAType>
bool& SampleAtLeavesVisitor::operator()(RAType* ra) const
{
  if (ra)
    return ra->SampleAtLeaves();
  throw std::runtime_error("no rank-approximate search model is initialized");
}

//! Exposes the Alpha() method of the given RAType instance.
template<typename RAType>
double& AlphaVisitor::operator()(RAType* ra) const
{
  if (ra)
    return ra->Alpha();
  throw std::runtime_error("no rank-approximate model is initialized");
}

//! Exposes the Tau() method of the given RAType instance.
template<typename RAType>
double& TauVisitor::operator()(RAType* ra) const
{
  if (ra)
    return ra->Tau();
  throw std::runtime_error("no rank-approximate model is initialized");
}

//! Exposes the SingleMode() method of the given RAType.
template<typename RAType>
bool& SingleModeVisitor::operator()(RAType* ra) const
{
  if (ra)
    return ra->SingleMode();
  throw std::runtime_error("no rank-approximate model is initialized");
}

//! Exposes the referenceSet of the given RAType.
template<typename RAType>
const arma::mat& ReferenceSetVisitor::operator()(RAType* ra) const
{
  if (ra)
    return ra->ReferenceSet();
  throw std::runtime_error("no rank-approximate model is initialized");
}

//! Exposes the Naive() method of the given RAType instance.
template<typename RAType>
bool& NaiveVisitor::operator()(RAType* ra) const
{
  if (ra)
    return ra->Naive();
  throw std::runtime_error("no rank-approximate search model is initialized");
}

//! For cleaning memory
template<typename RSType>
void DeleteVisitor::operator()(RSType* rs) const
{
  if (rs)
    delete rs;
}

template<typename SortPolicy>
RAModel<SortPolicy>::RAModel(const TreeTypes treeType, const bool randomBasis) :
    treeType(treeType),
    leafSize(20),
    randomBasis(randomBasis)
{
  // Nothing to do.
}

// Copy constructor.
template<typename SortPolicy>
RAModel<SortPolicy>::RAModel(const RAModel& other) :
    treeType(other.treeType),
    leafSize(other.leafSize),
    randomBasis(other.randomBasis),
    q(other.q),
    raSearch(other.raSearch)
{
  // Nothing to do.
}

// Move constructor.
template<typename SortPolicy>
RAModel<SortPolicy>::RAModel(RAModel&& other) :
    treeType(other.treeType),
    leafSize(other.leafSize),
    randomBasis(other.randomBasis),
    q(std::move(other.q)),
    raSearch(std::move(other.raSearch))
{
  // Clear other model.
  other.treeType = TreeTypes::KD_TREE;
  other.leafSize = 20;
  other.randomBasis = false;
  other.raSearch = decltype(other.raSearch)();
}

// Copy operator.
template<typename SortPolicy>
RAModel<SortPolicy>& RAModel<SortPolicy>::operator=(const RAModel& other)
{
  // Clear current model.
  boost::apply_visitor(DeleteVisitor(), raSearch);

  treeType = other.treeType;
  leafSize = other.leafSize;
  randomBasis = other.randomBasis;
  q = other.q;
  raSearch = other.raSearch;

  return *this;
}

template<typename SortPolicy>
RAModel<SortPolicy>& RAModel<SortPolicy>::operator=(RAModel&& other)
{
  boost::apply_visitor(DeleteVisitor(), raSearch);

  treeType = other.treeType;
  leafSize = other.leafSize;
  randomBasis = other.randomBasis;
  q = std::move(other.q);
  raSearch = std::move(other.raSearch);

  // Reset other model.
  other.treeType = TreeTypes::KD_TREE;
  other.leafSize = 20;
  other.randomBasis = false;
  other.raSearch = decltype(other.raSearch)();

  return *this;
}

// Clean memory, if necessary
template<typename SortPolicy>
RAModel<SortPolicy>::~RAModel()
{
  boost::apply_visitor(DeleteVisitor(), raSearch);
}

template<typename SortPolicy>
template<typename Archive>
void RAModel<SortPolicy>::serialize(Archive& ar,
                                    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(treeType);
  ar & BOOST_SERIALIZATION_NVP(randomBasis);
  ar & BOOST_SERIALIZATION_NVP(q);

  // This should never happen, but just in case, be clean with memory.
  if (Archive::is_loading::value)
  {
    boost::apply_visitor(DeleteVisitor(), raSearch);
  }

  // We only need to serialize one of the kRANN objects.
  ar & BOOST_SERIALIZATION_NVP(raSearch);
}

template<typename SortPolicy>
const arma::mat& RAModel<SortPolicy>::Dataset() const
{
  return boost::apply_visitor(ReferenceSetVisitor(), raSearch);
}

template<typename SortPolicy>
bool RAModel<SortPolicy>::Naive() const
{
  return boost::apply_visitor(NaiveVisitor(), raSearch);
}

template<typename SortPolicy>
bool& RAModel<SortPolicy>::Naive()
{
  return boost::apply_visitor(NaiveVisitor(), raSearch);
}

template<typename SortPolicy>
bool RAModel<SortPolicy>::SingleMode() const
{
  return boost::apply_visitor(SingleModeVisitor(), raSearch);
}

template<typename SortPolicy>
bool& RAModel<SortPolicy>::SingleMode()
{
  return boost::apply_visitor(SingleModeVisitor(), raSearch);
}

template<typename SortPolicy>
double RAModel<SortPolicy>::Tau() const
{
  return boost::apply_visitor(TauVisitor(), raSearch);
}

template<typename SortPolicy>
double& RAModel<SortPolicy>::Tau()
{
  return boost::apply_visitor(TauVisitor(), raSearch);
}

template<typename SortPolicy>
double RAModel<SortPolicy>::Alpha() const
{
  return boost::apply_visitor(AlphaVisitor(), raSearch);
}

template<typename SortPolicy>
double& RAModel<SortPolicy>::Alpha()
{
  return boost::apply_visitor(AlphaVisitor(), raSearch);
}

template<typename SortPolicy>
bool RAModel<SortPolicy>::SampleAtLeaves() const
{
  return boost::apply_visitor(SampleAtLeavesVisitor(), raSearch);
}

template<typename SortPolicy>
bool& RAModel<SortPolicy>::SampleAtLeaves()
{
  return boost::apply_visitor(SampleAtLeavesVisitor(), raSearch);
}

template<typename SortPolicy>
bool RAModel<SortPolicy>::FirstLeafExact() const
{
  return boost::apply_visitor(FirstLeafExactVisitor(), raSearch);
}

template<typename SortPolicy>
bool& RAModel<SortPolicy>::FirstLeafExact()
{
  return boost::apply_visitor(FirstLeafExactVisitor(), raSearch);
}

template<typename SortPolicy>
size_t RAModel<SortPolicy>::SingleSampleLimit() const
{
  return boost::apply_visitor(SingleSampleLimitVisitor(), raSearch);
}

template<typename SortPolicy>
size_t& RAModel<SortPolicy>::SingleSampleLimit()
{
  return boost::apply_visitor(SingleSampleLimitVisitor(), raSearch);
}

template<typename SortPolicy>
size_t RAModel<SortPolicy>::LeafSize() const
{
  return leafSize;
}

template<typename SortPolicy>
size_t& RAModel<SortPolicy>::LeafSize()
{
  return leafSize;
}

template<typename SortPolicy>
typename RAModel<SortPolicy>::TreeTypes RAModel<SortPolicy>::TreeType() const
{
  return treeType;
}

template<typename SortPolicy>
typename RAModel<SortPolicy>::TreeTypes& RAModel<SortPolicy>::TreeType()
{
  return treeType;
}

template<typename SortPolicy>
bool RAModel<SortPolicy>::RandomBasis() const
{
  return randomBasis;
}

template<typename SortPolicy>
bool& RAModel<SortPolicy>::RandomBasis()
{
  return randomBasis;
}

template<typename SortPolicy>
void RAModel<SortPolicy>::BuildModel(arma::mat&& referenceSet,
                                     const size_t leafSize,
                                     const bool naive,
                                     const bool singleMode)
{
  // Initialize random basis, if necessary.
  if (randomBasis)
  {
    Log::Info << "Creating random basis..." << std::endl;
    math::RandomBasis(q, referenceSet.n_rows);
  }

  // Clean memory, if necessary.
  boost::apply_visitor(DeleteVisitor(), raSearch);

  this->leafSize = leafSize;

  if (randomBasis)
    referenceSet = q * referenceSet;

  if (!naive)
  {
    Timer::Start("tree_building");
    Log::Info << "Building reference tree..." << std::endl;
  }

  switch (treeType)
  {
    case KD_TREE:
      raSearch = new RAType<SortPolicy, tree::KDTree>(naive, singleMode);
      break;
    case COVER_TREE:
      raSearch = new RAType<SortPolicy, tree::StandardCoverTree>(naive,
          singleMode);
      break;
    case R_TREE:
      raSearch = new RAType<SortPolicy, tree::RTree>(naive, singleMode);
      break;
    case R_STAR_TREE:
      raSearch = new RAType<SortPolicy, tree::RStarTree>(naive, singleMode);
      break;
    case X_TREE:
      raSearch = new RAType<SortPolicy, tree::XTree>(naive, singleMode);
      break;
    case HILBERT_R_TREE:
      raSearch = new RAType<SortPolicy, tree::HilbertRTree>(naive, singleMode);
      break;
    case R_PLUS_TREE:
      raSearch = new RAType<SortPolicy, tree::RPlusTree>(naive, singleMode);
      break;
    case R_PLUS_PLUS_TREE:
      raSearch = new RAType<SortPolicy, tree::RPlusPlusTree>(naive,
          singleMode);
      break;
    case UB_TREE:
      raSearch = new RAType<SortPolicy, tree::UBTree>(naive, singleMode);
      break;
    case OCTREE:
      raSearch = new RAType<SortPolicy, tree::Octree>(naive, singleMode);
      break;
  }

  TrainVisitor<SortPolicy> tn(std::move(referenceSet), leafSize);
  boost::apply_visitor(tn, raSearch);

  if (!naive)
  {
    Timer::Stop("tree_building");
    Log::Info << "Tree built." << std::endl;
  }
}

template<typename SortPolicy>
void RAModel<SortPolicy>::Search(arma::mat&& querySet,
                                 const size_t k,
                                 arma::Mat<size_t>& neighbors,
                                 arma::mat& distances)
{
  // Apply the random basis if necessary.
  if (randomBasis)
    querySet = q * querySet;

  Log::Info << "Searching for " << k << " approximate nearest neighbors with ";
  if (!Naive() && !SingleMode())
    Log::Info << "dual-tree rank-approximate " << TreeName() << " search...";
  else if (!Naive())
    Log::Info << "single-tree rank-approximate " << TreeName() << " search...";
  else
    Log::Info << "brute-force (naive) rank-approximate search...";
  Log::Info << std::endl;

  BiSearchVisitor<SortPolicy> search(querySet, k, neighbors, distances,
      leafSize);
  boost::apply_visitor(search, raSearch);
}

template<typename SortPolicy>
void RAModel<SortPolicy>::Search(const size_t k,
                                 arma::Mat<size_t>& neighbors,
                                 arma::mat& distances)
{
  Log::Info << "Searching for " << k << " approximate nearest neighbors with ";
  if (!Naive() && !SingleMode())
    Log::Info << "dual-tree rank-approximate " << TreeName() << " search...";
  else if (!Naive())
    Log::Info << "single-tree rank-approximate " << TreeName() << " search...";
  else
    Log::Info << "brute-force (naive) rank-approximate search...";
  Log::Info << std::endl;

  MonoSearchVisitor search(k, neighbors, distances);
  boost::apply_visitor(search, raSearch);
}

template<typename SortPolicy>
std::string RAModel<SortPolicy>::TreeName() const
{
  switch (treeType)
  {
    case KD_TREE:
      return "kd-tree";
    case COVER_TREE:
      return "cover tree";
    case R_TREE:
      return "R tree";
    case R_STAR_TREE:
      return "R* tree";
    case X_TREE:
      return "X tree";
    case HILBERT_R_TREE:
      return "Hilbert R tree";
    case R_PLUS_TREE:
      return "R+ tree";
    case R_PLUS_PLUS_TREE:
      return "R++ tree";
    case UB_TREE:
      return "UB tree";
    case OCTREE:
      return "octree";
    default:
      return "unknown tree";
  }
}

} // namespace neighbor
} // namespace mlpack

#endif
