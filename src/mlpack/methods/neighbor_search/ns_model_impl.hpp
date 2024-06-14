/**
 * @file methods/neighbor_search/ns_model_impl.hpp
 * @author Ryan Curtin
 *
 * This is a model for nearest or furthest neighbor search.  It is useful in
 * that it provides an easy way to serialize a model, abstracts away the
 * different types of trees, and also reflects the NeighborSearch API and
 * automatically directs to the right tree type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_IMPL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_IMPL_HPP

// In case it hasn't been included yet.
#include "ns_model.hpp"

namespace mlpack {

//! Train the model with the given options.  For NSWrapper, we ignore the
//! extra parameters.
template<typename SortPolicy,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename RuleType> class DualTreeTraversalType,
         template<typename RuleType> class SingleTreeTraversalType>
void NSWrapper<
    SortPolicy, TreeType, DualTreeTraversalType, SingleTreeTraversalType
>::Train(util::Timers& timers,
         arma::mat&& referenceSet,
         const size_t /* leafSize */,
         const double /* tau */,
         const double /* rho */)
{
  if (ns.SearchMode() != NAIVE_MODE)
    timers.Start("tree_building");

  ns.Train(std::move(referenceSet));

  if (ns.SearchMode() != NAIVE_MODE)
    timers.Stop("tree_building");
}

//! Perform bichromatic neighbor search (i.e. search with a separate query
//! set).  For NSWrapper, we ignore the extra parameters.
template<typename SortPolicy,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename RuleType> class DualTreeTraversalType,
         template<typename RuleType> class SingleTreeTraversalType>
void NSWrapper<
    SortPolicy, TreeType, DualTreeTraversalType, SingleTreeTraversalType
>::Search(util::Timers& timers,
          arma::mat&& querySet,
          const size_t k,
          arma::Mat<size_t>& neighbors,
          arma::mat& distances,
          const size_t /* leafSize */,
          const double /* rho */)
{
  if (ns.SearchMode() == DUAL_TREE_MODE)
  {
    // We build the query tree manually, so that we can time how long it takes.
    timers.Start("tree_building");
    typename decltype(ns)::Tree queryTree(std::move(querySet));
    timers.Stop("tree_building");

    timers.Start("computing_neighbors");
    ns.Search(queryTree, k, neighbors, distances);
    timers.Stop("computing_neighbors");
  }
  else
  {
    timers.Start("computing_neighbors");
    ns.Search(std::move(querySet), k, neighbors, distances);
    timers.Stop("computing_neighbors");
  }
}

//! Perform monochromatic neighbor search (i.e. use the reference set as the
//! query set).
template<typename SortPolicy,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename RuleType> class DualTreeTraversalType,
         template<typename RuleType> class SingleTreeTraversalType>
void NSWrapper<
    SortPolicy, TreeType, DualTreeTraversalType, SingleTreeTraversalType
>::Search(util::Timers& timers,
          const size_t k,
          arma::Mat<size_t>& neighbors,
          arma::mat& distances)
{
  timers.Start("computing_neighbors");
  ns.Search(k, neighbors, distances);
  timers.Stop("computing_neighbors");
}

//! Train a model with the given parameters.  This overload uses leafSize but
//! ignores the other parameters.
template<typename SortPolicy,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename RuleType> class DualTreeTraversalType,
         template<typename RuleType> class SingleTreeTraversalType>
void LeafSizeNSWrapper<
    SortPolicy, TreeType, DualTreeTraversalType, SingleTreeTraversalType
>::Train(util::Timers& timers,
         arma::mat&& referenceSet,
         const size_t leafSize,
         const double /* tau */,
         const double /* rho */)
{
  if (ns.SearchMode() == NAIVE_MODE)
  {
    ns.Train(std::move(referenceSet));
  }
  else
  {
    // Build the tree with the specified leaf size.
    timers.Start("tree_building");
    std::vector<size_t> oldFromNewReferences;
    typename decltype(ns)::Tree referenceTree(std::move(referenceSet),
        oldFromNewReferences, leafSize);
    ns.Train(std::move(referenceTree));
    ns.oldFromNewReferences = std::move(oldFromNewReferences);
    timers.Stop("tree_building");
  }
}

//! Perform bichromatic search (e.g. search with a separate query set).  This
//! overload uses the leaf size, but ignores the other parameters.
template<typename SortPolicy,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename RuleType> class DualTreeTraversalType,
         template<typename RuleType> class SingleTreeTraversalType>
void LeafSizeNSWrapper<
    SortPolicy, TreeType, DualTreeTraversalType, SingleTreeTraversalType
>::Search(util::Timers& timers,
          arma::mat&& querySet,
          const size_t k,
          arma::Mat<size_t>& neighbors,
          arma::mat& distances,
          const size_t leafSize,
          const double /* rho */)
{
  if (ns.SearchMode() == DUAL_TREE_MODE)
  {
    // We actually have to do the mapping of query points ourselves, since the
    // NeighborSearch class does not provide a way for us to specify the leaf
    // size when building the query tree.  (Therefore we must also build the
    // query tree manually.)
    timers.Start("tree_building");
    std::vector<size_t> oldFromNewQueries;
    typename decltype(ns)::Tree queryTree(std::move(querySet),
        oldFromNewQueries, leafSize);
    timers.Stop("tree_building");

    arma::Mat<size_t> neighborsOut;
    arma::mat distancesOut;
    timers.Start("computing_neighbors");
    ns.Search(queryTree, k, neighborsOut, distancesOut);
    timers.Stop("computing_neighbors");

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
  {
    timers.Start("computing_neighbors");
    ns.Search(querySet, k, neighbors, distances);
    timers.Stop("computing_neighbors");
  }
}

//! Train the model using the given parameters.
template<typename SortPolicy>
void SpillNSWrapper<SortPolicy>::Train(util::Timers& timers,
                                       arma::mat&& referenceSet,
                                       const size_t leafSize,
                                       const double tau,
                                       const double rho)
{
  timers.Start("tree_building");
  typename decltype(ns)::Tree tree(std::move(referenceSet), tau, leafSize,
      rho);
  timers.Stop("tree_building");

  ns.Train(std::move(tree));
}

//! Perform bichromatic search (i.e. search with a different query set) using
//! the given parameters.
template<typename SortPolicy>
void SpillNSWrapper<SortPolicy>::Search(util::Timers& timers,
                                        arma::mat&& querySet,
                                        const size_t k,
                                        arma::Mat<size_t>& neighbors,
                                        arma::mat& distances,
                                        const size_t leafSize,
                                        const double rho)
{
  if (ns.SearchMode() == DUAL_TREE_MODE)
  {
    // For Dual Tree Search on SpillTrees, the queryTree must be built with
    // non overlapping (tau = 0).
    timers.Start("tree_building");
    typename decltype(ns)::Tree queryTree(std::move(querySet), 0 /* tau */,
        leafSize, rho);
    timers.Stop("tree_building");

    timers.Start("computing_neighbors");
    ns.Search(queryTree, k, neighbors, distances);
    timers.Stop("computing_neighbors");
  }
  else
  {
    timers.Start("computing_neighbors");
    ns.Search(querySet, k, neighbors, distances);
    timers.Stop("computing_neighbors");
  }
}

/**
 * Initialize the NSModel with the given type and whether or not a random
 * basis should be used.
 */
template<typename SortPolicy>
NSModel<SortPolicy>::NSModel(TreeTypes treeType, bool randomBasis) :
    treeType(treeType),
    randomBasis(randomBasis),
    leafSize(20),
    tau(0.0),
    rho(0.7),
    nSearch(NULL)
{
  // Nothing to do.
}

template<typename SortPolicy>
NSModel<SortPolicy>::NSModel(const NSModel& other) :
    treeType(other.treeType),
    randomBasis(other.randomBasis),
    q(other.q),
    leafSize(other.leafSize),
    tau(other.tau),
    rho(other.rho),
    nSearch(other.nSearch->Clone())
{
  // Nothing to do.
}

template<typename SortPolicy>
NSModel<SortPolicy>::NSModel(NSModel&& other) :
    treeType(other.treeType),
    randomBasis(other.randomBasis),
    q(std::move(other.q)),
    leafSize(other.leafSize),
    tau(other.tau),
    rho(other.rho),
    nSearch(other.nSearch)
{
  // Reset parameters of the other model.
  other.treeType = TreeTypes::KD_TREE;
  other.randomBasis = false;
  other.leafSize = 20;
  other.tau = 0.0;
  other.rho = 0.7;
  other.nSearch = NULL;
}

template<typename SortPolicy>
NSModel<SortPolicy>& NSModel<SortPolicy>::operator=(const NSModel& other)
{
  if (this != &other)
  {
    delete nSearch;

    treeType = other.treeType;
    randomBasis = other.randomBasis;
    q = other.q;
    leafSize = other.leafSize;
    tau = other.tau;
    rho = other.rho;
    nSearch = other.nSearch->Clone();
  }

  return *this;
}

template<typename SortPolicy>
NSModel<SortPolicy>& NSModel<SortPolicy>::operator=(NSModel&& other)
{
  if (this != &other)
  {
    delete nSearch;

    treeType = other.treeType;
    randomBasis = other.randomBasis;
    q = std::move(other.q);
    leafSize = other.leafSize;
    tau = other.tau;
    rho = other.rho;
    nSearch = other.nSearch;

    // Reset parameters of the other model.
    other.treeType = TreeTypes::KD_TREE;
    other.randomBasis = false;
    other.leafSize = 20;
    other.tau = 0.0;
    other.rho = 0.7;
    other.nSearch = NULL;
  }

  return *this;
}

//! Clean memory, if necessary.
template<typename SortPolicy>
NSModel<SortPolicy>::~NSModel()
{
  delete nSearch;
}

//! Serialize the kNN model.
template<typename SortPolicy>
template<typename Archive>
void NSModel<SortPolicy>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(treeType));
  ar(CEREAL_NVP(randomBasis));
  ar(CEREAL_NVP(q));
  ar(CEREAL_NVP(leafSize));
  ar(CEREAL_NVP(tau));
  ar(CEREAL_NVP(rho));

  // This should never happen, but just in case, be clean with memory.
  if (cereal::is_loading<Archive>())
    InitializeModel(DUAL_TREE_MODE, 0.0); // Values will be overwritten.

  // Avoid polymorphic serialization by explicitly serializing the correct type.
  switch (treeType)
  {
    case KD_TREE:
      {
        LeafSizeNSWrapper<SortPolicy, KDTree>& typedSearch =
            dynamic_cast<LeafSizeNSWrapper<SortPolicy, KDTree>&>(*nSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case COVER_TREE:
      {
        NSWrapper<SortPolicy, StandardCoverTree>& typedSearch =
            dynamic_cast<NSWrapper<SortPolicy, StandardCoverTree>&>(*nSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case R_TREE:
      {
        NSWrapper<SortPolicy, RTree>& typedSearch =
            dynamic_cast<NSWrapper<SortPolicy, RTree>&>(*nSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case R_STAR_TREE:
      {
        NSWrapper<SortPolicy, RStarTree>& typedSearch =
            dynamic_cast<NSWrapper<SortPolicy, RStarTree>&>(*nSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case BALL_TREE:
      {
        LeafSizeNSWrapper<SortPolicy, BallTree>& typedSearch =
            dynamic_cast<LeafSizeNSWrapper<SortPolicy, BallTree>&>(*nSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case X_TREE:
      {
        NSWrapper<SortPolicy, XTree>& typedSearch =
            dynamic_cast<NSWrapper<SortPolicy, XTree>&>(*nSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case HILBERT_R_TREE:
      {
        NSWrapper<SortPolicy, HilbertRTree>& typedSearch =
            dynamic_cast<NSWrapper<SortPolicy, HilbertRTree>&>(*nSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case R_PLUS_TREE:
      {
        NSWrapper<SortPolicy, RPlusTree>& typedSearch =
            dynamic_cast<NSWrapper<SortPolicy, RPlusTree>&>(*nSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case R_PLUS_PLUS_TREE:
      {
        NSWrapper<SortPolicy, RPlusPlusTree>& typedSearch =
            dynamic_cast<NSWrapper<SortPolicy, RPlusPlusTree>&>(*nSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case SPILL_TREE:
      {
        SpillNSWrapper<SortPolicy>& typedSearch =
            dynamic_cast<SpillNSWrapper<SortPolicy>&>(*nSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case VP_TREE:
      {
        LeafSizeNSWrapper<SortPolicy, VPTree>& typedSearch =
            dynamic_cast<LeafSizeNSWrapper<SortPolicy, VPTree>&>(*nSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case RP_TREE:
      {
        LeafSizeNSWrapper<SortPolicy, RPTree>& typedSearch =
            dynamic_cast<LeafSizeNSWrapper<SortPolicy, RPTree>&>(*nSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case MAX_RP_TREE:
      {
        LeafSizeNSWrapper<SortPolicy, MaxRPTree>& typedSearch =
            dynamic_cast<LeafSizeNSWrapper<SortPolicy, MaxRPTree>&>(*nSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case UB_TREE:
      {
        LeafSizeNSWrapper<SortPolicy, UBTree>& typedSearch =
            dynamic_cast<LeafSizeNSWrapper<SortPolicy, UBTree>&>(*nSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case OCTREE:
      {
        LeafSizeNSWrapper<SortPolicy, Octree>& typedSearch =
            dynamic_cast<LeafSizeNSWrapper<SortPolicy, Octree>&>(*nSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
  }
}

//! Expose the dataset.
template<typename SortPolicy>
const arma::mat& NSModel<SortPolicy>::Dataset() const
{
  return nSearch->Dataset();
}

//! Access the search mode.
template<typename SortPolicy>
NeighborSearchMode NSModel<SortPolicy>::SearchMode() const
{
  return nSearch->SearchMode();
}

//! Modify the search mode.
template<typename SortPolicy>
NeighborSearchMode& NSModel<SortPolicy>::SearchMode()
{
  return nSearch->SearchMode();
}

template<typename SortPolicy>
double NSModel<SortPolicy>::Epsilon() const
{
  return nSearch->Epsilon();
}

template<typename SortPolicy>
double& NSModel<SortPolicy>::Epsilon()
{
  return nSearch->Epsilon();
}

//! Initialize a model given the tree type.  (No training happens here.)
template<typename SortPolicy>
void NSModel<SortPolicy>::InitializeModel(const NeighborSearchMode searchMode,
                                          const double epsilon)
{
  // Clear existing memory.
  if (nSearch)
    delete nSearch;

  switch (treeType)
  {
    case KD_TREE:
      nSearch = new LeafSizeNSWrapper<SortPolicy, KDTree>(searchMode, epsilon);
      break;
    case COVER_TREE:
      nSearch = new NSWrapper<SortPolicy, StandardCoverTree>(searchMode,
          epsilon);
      break;
    case R_TREE:
      nSearch = new NSWrapper<SortPolicy, RTree>(searchMode, epsilon);
      break;
    case R_STAR_TREE:
      nSearch = new NSWrapper<SortPolicy, RStarTree>(searchMode, epsilon);
      break;
    case BALL_TREE:
      nSearch = new LeafSizeNSWrapper<SortPolicy, BallTree>(searchMode,
          epsilon);
      break;
    case X_TREE:
      nSearch = new NSWrapper<SortPolicy, XTree>(searchMode, epsilon);
      break;
    case HILBERT_R_TREE:
      nSearch = new NSWrapper<SortPolicy, HilbertRTree>(searchMode, epsilon);
      break;
    case R_PLUS_TREE:
      nSearch = new NSWrapper<SortPolicy, RPlusTree>(searchMode, epsilon);
      break;
    case R_PLUS_PLUS_TREE:
      nSearch = new NSWrapper<SortPolicy, RPlusPlusTree>(searchMode, epsilon);
      break;
    case VP_TREE:
      nSearch = new LeafSizeNSWrapper<SortPolicy, VPTree>(searchMode, epsilon);
      break;
    case RP_TREE:
      nSearch = new LeafSizeNSWrapper<SortPolicy, RPTree>(searchMode, epsilon);
      break;
    case MAX_RP_TREE:
      nSearch = new LeafSizeNSWrapper<SortPolicy, MaxRPTree>(searchMode,
          epsilon);
      break;
    case SPILL_TREE:
      nSearch = new SpillNSWrapper<SortPolicy>(searchMode, epsilon);
      break;
    case UB_TREE:
      nSearch = new LeafSizeNSWrapper<SortPolicy, UBTree>(searchMode, epsilon);
      break;
    case OCTREE:
      nSearch = new LeafSizeNSWrapper<SortPolicy, Octree>(searchMode, epsilon);
      break;
  }
}

//! Build the reference tree.
template<typename SortPolicy>
void NSModel<SortPolicy>::BuildModel(util::Timers& timers,
                                     arma::mat&& referenceSet,
                                     const NeighborSearchMode searchMode,
                                     const double epsilon)
{
  // Initialize random basis if necessary.
  if (randomBasis)
  {
    timers.Start("computing_random_basis");
    Log::Info << "Creating random basis..." << std::endl;
    while (true)
    {
      // [Q, R] = qr(randn(d, d));
      // Q = Q * diag(sign(diag(R)));
      arma::mat r;
      if (arma::qr(q, r, randn<arma::mat>(referenceSet.n_rows,
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

    referenceSet = q * referenceSet;
    timers.Stop("computing_random_basis");
  }

  if (searchMode != NAIVE_MODE)
    Log::Info << "Building reference tree..." << std::endl;

  InitializeModel(searchMode, epsilon);
  nSearch->Train(timers, std::move(referenceSet), leafSize, tau, rho);

  if (searchMode != NAIVE_MODE)
    Log::Info << "Tree built." << std::endl;
}

//! Perform neighbor search.  The query set will be reordered.
template<typename SortPolicy>
void NSModel<SortPolicy>::Search(util::Timers& timers,
                                 arma::mat&& querySet,
                                 const size_t k,
                                 arma::Mat<size_t>& neighbors,
                                 arma::mat& distances)
{
  // We may need to map the query set randomly.
  if (randomBasis)
  {
    timers.Start("applying_random_basis");
    querySet = q * querySet;
    timers.Stop("applying_random_basis");
  }

  Log::Info << "Searching for " << k << " neighbors with ";

  switch (SearchMode())
  {
    case NAIVE_MODE:
      Log::Info << "brute-force (naive) search..." << std::endl;
      break;
    case SINGLE_TREE_MODE:
      Log::Info << "single-tree " << TreeName() << " search..." << std::endl;
      break;
    case DUAL_TREE_MODE:
      Log::Info << "dual-tree " << TreeName() << " search..." << std::endl;
      break;
    case GREEDY_SINGLE_TREE_MODE:
      Log::Info << "greedy single-tree " << TreeName() << " search..."
          << std::endl;
      break;
  }

  nSearch->Search(timers, std::move(querySet), k, neighbors, distances,
      leafSize, rho);
}

//! Perform neighbor search.
template<typename SortPolicy>
void NSModel<SortPolicy>::Search(util::Timers& timers,
                                 const size_t k,
                                 arma::Mat<size_t>& neighbors,
                                 arma::mat& distances)
{
  Log::Info << "Searching for " << k << " neighbors with ";

  switch (SearchMode())
  {
    case NAIVE_MODE:
      Log::Info << "brute-force (naive) search..." << std::endl;
      break;
    case SINGLE_TREE_MODE:
      Log::Info << "single-tree " << TreeName() << " search..." << std::endl;
      break;
    case DUAL_TREE_MODE:
      Log::Info << "dual-tree " << TreeName() << " search..." << std::endl;
      break;
    case GREEDY_SINGLE_TREE_MODE:
      Log::Info << "greedy single-tree " << TreeName() << " search..."
          << std::endl;
      break;
  }

  if (Epsilon() != 0 && SearchMode() != NAIVE_MODE)
    Log::Info << "Maximum of " << Epsilon() * 100 << "% relative error."
        << std::endl;

  nSearch->Search(timers, k, neighbors, distances);
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
    case R_PLUS_TREE:
      return "R+ tree";
    case R_PLUS_PLUS_TREE:
      return "R++ tree";
    case SPILL_TREE:
      return "Spill tree";
    case VP_TREE:
      return "vantage point tree";
    case RP_TREE:
      return "random projection tree (mean split)";
    case MAX_RP_TREE:
      return "random projection tree (max split)";
    case UB_TREE:
      return "UB tree";
    case OCTREE:
      return "octree";
    default:
      return "unknown tree";
  }
}

} // namespace mlpack

#endif
