/**
 * @file spill_search_impl.hpp
 * @author Ryan Curtin
 * @author Marcos Pividori
 *
 * Implementation of SpillSearch class, which performs a Hybrid sp-tree search
 * on two datasets.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_SPILL_SEARCH_IMPL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_SPILL_SEARCH_IMPL_HPP

// In case it hasn't been included yet.
#include "spill_search.hpp"

namespace mlpack {
namespace neighbor {

// Construct the object.
template<typename MetricType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType>
SpillSearch<MetricType, MatType, HyperplaneType, SplitType>::SpillSearch(
    const MatType& referenceSetIn,
    const bool naive,
    const bool singleMode,
    const double tau,
    const double leafSize,
    const double rho,
    const double epsilon,
    const MetricType metric) :
    neighborSearch(naive, singleMode, epsilon, metric),
    tau(tau),
    rho(rho),
    leafSize(leafSize)
{
  if (tau < 0)
    throw std::invalid_argument("tau must be non-negative");
  if (rho < 0 || rho > 1)
    throw std::invalid_argument("rho must be in the range [0,1]");
  Train(referenceSetIn);
}

// Construct the object.
template<typename MetricType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType>
SpillSearch<MetricType, MatType, HyperplaneType, SplitType>::SpillSearch(
    MatType&& referenceSetIn,
    const bool naive,
    const bool singleMode,
    const double tau,
    const double leafSize,
    const double rho,
    const double epsilon,
    const MetricType metric) :
    neighborSearch(naive, singleMode, epsilon, metric),
    tau(tau),
    rho(rho),
    leafSize(leafSize)
{
  if (tau < 0)
    throw std::invalid_argument("tau must be non-negative");
  if (rho < 0 || rho > 1)
    throw std::invalid_argument("rho must be in the range [0,1]");
  Train(std::move(referenceSetIn));
}

// Construct the object.
template<typename MetricType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType>
SpillSearch<MetricType, MatType, HyperplaneType, SplitType>::SpillSearch(
    Tree* referenceTree,
    const bool singleMode,
    const double tau,
    const double leafSize,
    const double rho,
    const double epsilon,
    const MetricType metric) :
    neighborSearch(singleMode, epsilon, metric),
    tau(tau),
    rho(rho),
    leafSize(leafSize)
{
  if (tau < 0)
    throw std::invalid_argument("tau must be non-negative");
  if (rho < 0 || rho > 1)
    throw std::invalid_argument("rho must be in the range [0,1]");
  Train(referenceTree);
}

// Construct the object without a reference dataset.
template<typename MetricType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType>
SpillSearch<MetricType, MatType, HyperplaneType, SplitType>::SpillSearch(
    const bool naive,
    const bool singleMode,
    const double tau,
    const double leafSize,
    const double rho,
    const double epsilon,
    const MetricType metric) :
    neighborSearch(naive, singleMode, epsilon, metric),
    tau(tau),
    rho(rho),
    leafSize(leafSize)
{
  if (tau < 0)
    throw std::invalid_argument("tau must be non-negative");
  if (rho < 0 || rho > 1)
    throw std::invalid_argument("rho must be in the range [0,1]");
}

// Clean memory.
template<typename MetricType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType>
SpillSearch<MetricType, MatType, HyperplaneType, SplitType>::
~SpillSearch()
{
  /* Nothing to do */
}

template<typename MetricType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType>
void SpillSearch<MetricType, MatType, HyperplaneType, SplitType>::
Train(const MatType& referenceSet)
{
  if (Naive())
    neighborSearch.Train(referenceSet);
  else
  {
    // Build reference tree with proper value for tau.
    Tree* tree = new Tree(referenceSet, tau, leafSize, rho);
    neighborSearch.Train(tree);
    // Give the model ownership of the tree.
    neighborSearch.treeOwner = true;
  }
}

template<typename MetricType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType>
void SpillSearch<MetricType, MatType, HyperplaneType, SplitType>::
Train(MatType&& referenceSetIn)
{
  if (Naive())
    neighborSearch.Train(std::move(referenceSetIn));
  else
  {
    // Build reference tree with proper value for tau.
    Tree* tree = new Tree(std::move(referenceSetIn), tau, leafSize, rho);
    neighborSearch.Train(tree);
    // Give the model ownership of the tree.
    neighborSearch.treeOwner = true;
  }
}

template<typename MetricType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType>
void SpillSearch<MetricType, MatType, HyperplaneType, SplitType>::
Train(Tree* referenceTree)
{
  neighborSearch.Train(referenceTree);
}

template<typename MetricType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType>
void SpillSearch<MetricType, MatType, HyperplaneType, SplitType>::
Search(const MatType& querySet,
       const size_t k,
       arma::Mat<size_t>& neighbors,
       arma::mat& distances)
{
  if (Naive() || SingleMode())
    neighborSearch.Search(querySet, k, neighbors, distances);
  else
  {
    // For Dual Tree Search on SpillTrees, the queryTree must be built with non
    // overlapping (tau = 0).
    Tree queryTree(querySet, 0 /* tau */, leafSize, rho);
    neighborSearch.Search(&queryTree, k, neighbors, distances);
  }
}

template<typename MetricType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType>
void SpillSearch<MetricType, MatType, HyperplaneType, SplitType>::
Search(Tree* queryTree,
       const size_t k,
       arma::Mat<size_t>& neighbors,
       arma::mat& distances)
{
  neighborSearch.Search(queryTree, k, neighbors, distances);
}

template<typename MetricType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType>
void SpillSearch<MetricType, MatType, HyperplaneType, SplitType>::
Search(const size_t k,
       arma::Mat<size_t>& neighbors,
       arma::mat& distances)
{
  if (tau == 0 || Naive() || SingleMode())
    neighborSearch.Search(k, neighbors, distances);
  else
  {
    // For Dual Tree Search on SpillTrees, the queryTree must be built with non
    // overlapping (tau = 0). If the referenceTree was built with a non-zero
    // value for tau, we need to build a new queryTree.
    Tree queryTree(ReferenceSet(), 0 /* tau */, leafSize, rho);
    neighborSearch.Search(&queryTree, k, neighbors, distances, true);
  }
}

//! Serialize SpillSearch.
template<typename MetricType,
         typename MatType,
         template<typename HyperplaneMetricType> class HyperplaneType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType>
template<typename Archive>
void SpillSearch<MetricType, MatType, HyperplaneType, SplitType>::
    Serialize(Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(neighborSearch, "neighborSearch");
  ar & data::CreateNVP(tau, "tau");
  ar & data::CreateNVP(rho, "rho");
  ar & data::CreateNVP(leafSize, "leafSize");
}

} // namespace neighbor
} // namespace mlpack

#endif
