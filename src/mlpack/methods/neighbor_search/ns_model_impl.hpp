/**
 * @file ns_model_impl.hpp
 * @author Ryan Curtin
 *
 * This is a model for nearest or furthest neighbor search.  It is useful in
 * that it provides an easy way to serialize a model, abstracts away the
 * different types of trees, and also reflects the NeighborSearch API and
 * automatically directs to the right tree type.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_IMPL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_IMPL_HPP

// In case it hasn't been included yet.
#include "ns_model.hpp"

#include <boost/serialization/variant.hpp>

namespace mlpack {
namespace neighbor {

//! Monochromatic neighbor search on the given NSType instance.
template<typename NSType>
void MonoSearchVisitor::operator()(NSType *ns) const
{
  if (ns)
    return ns->Search(k, neighbors, distances);
  throw std::runtime_error("no neighbor search model initialized");
}

//! Save parameters for bichromatic neighbor search.
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
{}

//! Default Bichromatic neighbor search on the given NSType instance.
template<typename SortPolicy>
template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void BiSearchVisitor<SortPolicy>::operator()(NSTypeT<TreeType>* ns) const
{
  if (ns)
    return ns->Search(querySet, k, neighbors, distances);
  throw std::runtime_error("no neighbor search model initialized");
}

//! Bichromatic neighbor search on the given NSType specialized for KDTrees.
template<typename SortPolicy>
void BiSearchVisitor<SortPolicy>::operator()(NSTypeT<tree::KDTree>* ns) const
{
  if (ns)
    return SearchLeaf(ns);
  throw std::runtime_error("no neighbor search model initialized");
}

//! Bichromatic neighbor search on the given NSType specialized for BallTrees.
template<typename SortPolicy>
void BiSearchVisitor<SortPolicy>::operator()(NSTypeT<tree::BallTree>* ns) const
{
  if (ns)
    return SearchLeaf(ns);
  throw std::runtime_error("no neighbor search model initialized");
}

//! Bichromatic neighbor search on the given NSType considering the leafSize.
template<typename SortPolicy>
template<typename NSType>
void BiSearchVisitor<SortPolicy>::SearchLeaf(NSType* ns) const
{
  if (!ns->Naive() && !ns->SingleMode())
  {
    std::vector<size_t> oldFromNewQueries;
    typename NSType::Tree queryTree(std::move(querySet), oldFromNewQueries,
        leafSize);

    arma::Mat<size_t> neighborsOut;
    arma::mat distancesOut;
    ns->Search(&queryTree, k, neighborsOut, distancesOut);

    // Unmap the query points.
    distances.set_size(distancesOut.n_rows, distancesOut.n_cols);
    neighbors.set_size(neighborsOut.n_rows, neighborsOut.n_cols);
    for (size_t i = 0; i < neighborsOut.n_cols; ++i)
    {
      neighbors.col(oldFromNewQueries[i]) = neighborsOut.col(i);
      distances.col(oldFromNewQueries[i]) = distancesOut.col(i);
    }
  }
  else
    ns->Search(querySet, k, neighbors, distances);
}

//! Save parameters for Train.
template<typename SortPolicy>
TrainVisitor<SortPolicy>::TrainVisitor(arma::mat&& referenceSet,
                                       const size_t leafSize) :
    referenceSet(std::move(referenceSet)),
    leafSize(leafSize)
{}

//! Default Train on the given NSType instance.
template<typename SortPolicy>
template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void TrainVisitor<SortPolicy>::operator()(NSTypeT<TreeType>* ns) const
{
  if (ns)
    return ns->Train(std::move(referenceSet));
  throw std::runtime_error("no neighbor search model initialized");
}

//! Train on the given NSType specialized for KDTrees.
template<typename SortPolicy>
void TrainVisitor<SortPolicy>::operator ()(NSTypeT<tree::KDTree>* ns) const
{
  if (ns)
    return TrainLeaf(ns);
  throw std::runtime_error("no neighbor search model initialized");
}

//! Train on the given NSType specialized for BallTrees.
template<typename SortPolicy>
void TrainVisitor<SortPolicy>::operator ()(NSTypeT<tree::BallTree>* ns) const
{
  if (ns)
    return TrainLeaf(ns);
  throw std::runtime_error("no neighbor search model initialized");
}

//! Train on the given NSType considering the leafSize.
template<typename SortPolicy>
template<typename NSType>
void TrainVisitor<SortPolicy>::TrainLeaf(NSType* ns) const
{
  if (ns->Naive())
    ns->Train(std::move(referenceSet));
  else
  {
    std::vector<size_t> oldFromNewReferences;
    typename NSType::Tree* tree =
        new typename NSType::Tree(std::move(referenceSet),
        oldFromNewReferences, leafSize);
    ns->Train(tree);

    // Give the model ownership of the tree and the mappings.
    ns->treeOwner = true;
    ns->oldFromNewReferences = std::move(oldFromNewReferences);
  }
}

//! Expose the SingleMode method of the given NSType.
template<typename NSType>
bool& SingleModeVisitor::operator()(NSType* ns) const
{
  if (ns)
    return ns->SingleMode();
  throw std::runtime_error("no neighbor search model initialized");
}

//! Expose the Naive method of the given NSType.
template<typename NSType>
bool& NaiveVisitor::operator()(NSType* ns) const
{
  if (ns)
    return ns->Naive();
  throw std::runtime_error("no neighbor search model initialized");
}

//! Expose the Epsilon method of the given NSType.
template<typename NSType>
double& EpsilonVisitor::operator()(NSType* ns) const
{
  if (ns)
    return ns->Epsilon();
  throw std::runtime_error("no neighbor search model initialized");
}

//! Expose the referenceSet of the given NSType.
template<typename NSType>
const arma::mat& ReferenceSetVisitor::operator()(NSType* ns) const
{
  if (ns)
    return ns->ReferenceSet();
  throw std::runtime_error("no neighbor search model initialized");
}

//! Clean memory, if necessary.
template<typename NSType>
void DeleteVisitor::operator()(NSType* ns) const
{
  if (ns)
    delete ns;
}

/**
 * Initialize the NSModel with the given type and whether or not a random
 * basis should be used.
 */
template<typename SortPolicy>
NSModel<SortPolicy>::NSModel(TreeTypes treeType, bool randomBasis) :
    treeType(treeType),
    randomBasis(randomBasis)
{
  // Nothing to do.
}

//! Clean memory, if necessary.
template<typename SortPolicy>
NSModel<SortPolicy>::~NSModel()
{
  boost::apply_visitor(DeleteVisitor(), nSearch);
}

/**
 * Non-intrusive serialization for Neighbor Search class. We need this
 * definition because we are going to use the serialize function for boost
 * variant, which will look for a serialize function for its member types.
 */
template<typename Archive,
         typename SortPolicy,
         typename MetrType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename RuleType> class TraversalType>
void serialize(
    Archive& ar,
    NeighborSearch<SortPolicy, MetrType, MatType, TreeType, TraversalType>& ns,
    const unsigned int version)
{
  ns.Serialize(ar, version);
}

//! Serialize the kNN model.
template<typename SortPolicy>
template<typename Archive>
void NSModel<SortPolicy>::Serialize(Archive& ar,
                                    const unsigned int /* version */)
{
  ar & data::CreateNVP(treeType, "treeType");
  ar & data::CreateNVP(randomBasis, "randomBasis");
  ar & data::CreateNVP(q, "q");

  // This should never happen, but just in case, be clean with memory.
  if (Archive::is_loading::value)
    boost::apply_visitor(DeleteVisitor(), nSearch);

  const std::string& name = NSModelName<SortPolicy>::Name();
  ar & data::CreateNVP(nSearch, name);
}

//! Expose the dataset.
template<typename SortPolicy>
const arma::mat& NSModel<SortPolicy>::Dataset() const
{
  return boost::apply_visitor(ReferenceSetVisitor(), nSearch);
}

//! Expose singleMode.
template<typename SortPolicy>
bool NSModel<SortPolicy>::SingleMode() const
{
  return boost::apply_visitor(SingleModeVisitor(), nSearch);
}

template<typename SortPolicy>
bool& NSModel<SortPolicy>::SingleMode()
{
  return boost::apply_visitor(SingleModeVisitor(), nSearch);
}

//! Expose Naive.
template<typename SortPolicy>
bool NSModel<SortPolicy>::Naive() const
{
  return boost::apply_visitor(NaiveVisitor(), nSearch);
}

template<typename SortPolicy>
bool& NSModel<SortPolicy>::Naive()
{
  return boost::apply_visitor(NaiveVisitor(), nSearch);
}

template<typename SortPolicy>
double NSModel<SortPolicy>::Epsilon() const
{
  return boost::apply_visitor(EpsilonVisitor(), nSearch);
}

template<typename SortPolicy>
double& NSModel<SortPolicy>::Epsilon()
{
  return boost::apply_visitor(EpsilonVisitor(), nSearch);
}

//! Build the reference tree.
template<typename SortPolicy>
void NSModel<SortPolicy>::BuildModel(arma::mat&& referenceSet,
                                     const size_t leafSize,
                                     const bool naive,
                                     const bool singleMode,
                                     const double epsilon)
{
  // Initialize random basis if necessary.
  if (randomBasis)
  {
    Log::Info << "Creating random basis..." << std::endl;
    while (true)
    {
      // [Q, R] = qr(randn(d, d));
      // Q = Q * diag(sign(diag(R)));
      arma::mat r;
      if (arma::qr(q, r, arma::randn<arma::mat>(referenceSet.n_rows,
              referenceSet.n_rows)))
      {
        arma::vec rDiag(r.n_rows);
        for (size_t i = 0; i < rDiag.n_elem; ++i)
        {
          if (r(i, i) < 0)
            rDiag(i) = -1;
          else if (r(i, i) > 0)
            rDiag(i) = 1;
          else
            rDiag(i) = 0;
        }

        q *= arma::diagmat(rDiag);

        // Check if the determinant is positive.
        if (arma::det(q) >= 0)
          break;
      }
    }
  }

  // Clean memory, if necessary.
  boost::apply_visitor(DeleteVisitor(), nSearch);

  // Do we need to modify the reference set?
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
      nSearch = new NSType<SortPolicy, tree::KDTree>(naive, singleMode,
          epsilon);
      break;
    case COVER_TREE:
      nSearch = new NSType<SortPolicy, tree::StandardCoverTree>(naive,
          singleMode, epsilon);
      break;
    case R_TREE:
      nSearch = new NSType<SortPolicy, tree::RTree>(naive, singleMode, epsilon);
      break;
    case R_STAR_TREE:
      nSearch = new NSType<SortPolicy, tree::RStarTree>(naive, singleMode,
          epsilon);
      break;
    case BALL_TREE:
      nSearch = new NSType<SortPolicy, tree::BallTree>(naive, singleMode,
          epsilon);
      break;
    case X_TREE:
      nSearch = new NSType<SortPolicy, tree::XTree>(naive, singleMode, epsilon);
      break;
    case HILBERT_R_TREE:
      nSearch = new NSType<SortPolicy, tree::HilbertRTree>(naive, singleMode,
          epsilon);
      break;
  }

  TrainVisitor<SortPolicy> tn(std::move(referenceSet), leafSize);
  boost::apply_visitor(tn, nSearch);

  if (!naive)
  {
    Timer::Stop("tree_building");
    Log::Info << "Tree built." << std::endl;
  }
}

//! Perform neighbor search.  The query set will be reordered.
template<typename SortPolicy>
void NSModel<SortPolicy>::Search(arma::mat&& querySet,
                                 const size_t k,
                                 arma::Mat<size_t>& neighbors,
                                 arma::mat& distances)
{
  // We may need to map the query set randomly.
  if (randomBasis)
    querySet = q * querySet;

  Log::Info << "Searching for " << k << " neighbors with ";
  if (!Naive() && !SingleMode())
    Log::Info << "dual-tree " << TreeName() << " search..." << std::endl;
  else if (!Naive())
    Log::Info << "single-tree " << TreeName() << " search..." << std::endl;
  else
    Log::Info << "brute-force (naive) search..." << std::endl;
  if (Epsilon() != 0 && !Naive())
    Log::Info << "Maximum of " << Epsilon() * 100 << "% relative error."
        << std::endl;

  BiSearchVisitor<SortPolicy> search(querySet, k, neighbors, distances,
      leafSize);
  boost::apply_visitor(search, nSearch);
}

//! Perform neighbor search.
template<typename SortPolicy>
void NSModel<SortPolicy>::Search(const size_t k,
                                 arma::Mat<size_t>& neighbors,
                                 arma::mat& distances)
{
  Log::Info << "Searching for " << k << " neighbors with ";
  if (!Naive() && !SingleMode())
    Log::Info << "dual-tree " << TreeName() << " search..." << std::endl;
  else if (!Naive())
    Log::Info << "single-tree " << TreeName() << " search..." << std::endl;
  else
    Log::Info << "brute-force (naive) search..." << std::endl;
  if (Epsilon() != 0 && !Naive())
    Log::Info << "Maximum of " << Epsilon() * 100 << "% relative error."
        << std::endl;

  MonoSearchVisitor search(k, neighbors, distances);
  boost::apply_visitor(search, nSearch);
}

//! Get the name of the tree type.
template<typename SortPolicy>
std::string NSModel<SortPolicy>::TreeName() const
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
    case BALL_TREE:
      return "ball tree";
    case X_TREE:
      return "X tree";
    case HILBERT_R_TREE:
      return "Hilbert R tree";
    default:
      return "unknown tree";
  }
}

} // namespace neighbor
} // namespace mlpack

#endif
