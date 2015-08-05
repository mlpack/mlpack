/**
 * @file fastmks_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the FastMKS class (fast max-kernel search).
 */
#ifndef __MLPACK_METHODS_FASTMKS_FASTMKS_IMPL_HPP
#define __MLPACK_METHODS_FASTMKS_FASTMKS_IMPL_HPP

// In case it hasn't yet been included.
#include "fastmks.hpp"

#include "fastmks_rules.hpp"

#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <queue>

namespace mlpack {
namespace fastmks {

// No instantiated kernel.
template<typename KernelType,
         typename MatType,
         template<typename TMetricType, typename StatisticType, typename TMatType>
             class TreeType>
FastMKS<KernelType, MatType, TreeType>::FastMKS(
    const MatType& referenceSet,
    const bool singleMode,
    const bool naive) :
    referenceSet(referenceSet),
    referenceTree(NULL),
    treeOwner(true),
    singleMode(singleMode),
    naive(naive)
{
  Timer::Start("tree_building");

  if (!naive)
    referenceTree = new Tree(referenceSet);

  Timer::Stop("tree_building");
}

// Instantiated kernel.
template<typename KernelType,
         typename MatType,
         template<typename TMetricType, typename StatisticType, typename TMatType>
             class TreeType>
FastMKS<KernelType, MatType, TreeType>::FastMKS(const MatType& referenceSet,
                                                KernelType& kernel,
                                                const bool singleMode,
                                                const bool naive) :
    referenceSet(referenceSet),
    referenceTree(NULL),
    treeOwner(true),
    singleMode(singleMode),
    naive(naive),
    metric(kernel)
{
  Timer::Start("tree_building");

  // If necessary, the reference tree should be built.  There is no query tree.
  if (!naive)
    referenceTree = new Tree(referenceSet, metric);

  Timer::Stop("tree_building");
}

// One dataset, pre-built tree.
template<typename KernelType,
         typename MatType,
         template<typename TMetricType, typename StatisticType, typename TMatType>
             class TreeType>
FastMKS<KernelType, MatType, TreeType>::FastMKS(Tree* referenceTree,
                                                const bool singleMode) :
    referenceSet(referenceTree->Dataset()),
    referenceTree(referenceTree),
    treeOwner(false),
    singleMode(singleMode),
    naive(false),
    metric(referenceTree->Metric())
{
  // Nothing to do.
}

template<typename KernelType,
         typename MatType,
         template<typename TMetricType, typename StatisticType, typename TMatType>
             class TreeType>
FastMKS<KernelType, MatType, TreeType>::~FastMKS()
{
  // If we created the trees, we must delete them.
  if (treeOwner && referenceTree)
    delete referenceTree;
}

template<typename KernelType,
         typename MatType,
         template<typename TMetricType, typename StatisticType, typename TMatType>
             class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Search(
    const MatType& querySet,
    const size_t k,
    arma::Mat<size_t>& indices,
    arma::mat& kernels)
{
  Timer::Start("computing_products");

  // No remapping will be necessary because we are using the cover tree.
  indices.set_size(k, querySet.n_cols);
  kernels.set_size(k, querySet.n_cols);

  // Naive implementation.
  if (naive)
  {
    // Fill kernels.
    kernels.fill(-DBL_MAX);

    // Simple double loop.  Stupid, slow, but a good benchmark.
    for (size_t q = 0; q < querySet.n_cols; ++q)
    {
      for (size_t r = 0; r < referenceSet.n_cols; ++r)
      {
        const double eval = metric.Kernel().Evaluate(querySet.col(q),
                                                     referenceSet.col(r));

        size_t insertPosition;
        for (insertPosition = 0; insertPosition < indices.n_rows;
            ++insertPosition)
          if (eval > kernels(insertPosition, q))
            break;

        if (insertPosition < indices.n_rows)
          InsertNeighbor(indices, kernels, q, insertPosition, r, eval);
      }
    }

    Timer::Stop("computing_products");

    return;
  }

  // Single-tree implementation.
  if (singleMode)
  {
    // Fill kernels.
    kernels.fill(-DBL_MAX);

    // Create rules object (this will store the results).  This constructor
    // precalculates each self-kernel value.
    typedef FastMKSRules<KernelType, Tree> RuleType;
    RuleType rules(referenceSet, querySet, indices, kernels, metric.Kernel());

    typename Tree::template SingleTreeTraverser<RuleType> traverser(rules);

    for (size_t i = 0; i < querySet.n_cols; ++i)
      traverser.Traverse(i, *referenceTree);

    Log::Info << rules.BaseCases() << " base cases." << std::endl;
    Log::Info << rules.Scores() << " scores." << std::endl;

    Timer::Stop("computing_products");
    return;
  }

  // Dual-tree implementation.  First, we need to build the query tree.  We are
  // assuming it doesn't map anything...
  Timer::Stop("computing_products");
  Timer::Start("tree_building");
  Tree queryTree(querySet);
  Timer::Stop("tree_building");

  Search(&queryTree, k, indices, kernels);
}

template<typename KernelType,
         typename MatType,
         template<typename TMetricType, typename StatisticType, typename TMatType>
             class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Search(
    Tree* queryTree,
    const size_t k,
    arma::Mat<size_t>& indices,
    arma::mat& kernels)
{
  // If either naive mode or single mode is specified, this must fail.
  if (naive || singleMode)
  {
    throw std::invalid_argument("can't call Search() with a query tree when "
        "single mode or naive search is enabled");
  }

  // No remapping will be necessary because we are using the cover tree.
  indices.set_size(k, queryTree->Dataset().n_cols);
  kernels.set_size(k, queryTree->Dataset().n_cols);
  kernels.fill(-DBL_MAX);

  Timer::Start("computing_products");
  typedef FastMKSRules<KernelType, Tree> RuleType;
  RuleType rules(referenceSet, queryTree->Dataset(), indices, kernels,
      metric.Kernel());

  typename Tree::template DualTreeTraverser<RuleType> traverser(rules);

  traverser.Traverse(*queryTree, *referenceTree);

  Log::Info << rules.BaseCases() << " base cases." << std::endl;
  Log::Info << rules.Scores() << " scores." << std::endl;

  Timer::Stop("computing_products");
}

template<typename KernelType,
         typename MatType,
         template<typename TMetricType, typename StatisticType, typename TMatType>
             class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Search(
    const size_t k,
    arma::Mat<size_t>& indices,
    arma::mat& kernels)
{
  // No remapping will be necessary because we are using the cover tree.
  Timer::Start("computing_products");
  indices.set_size(k, referenceSet.n_cols);
  kernels.set_size(k, referenceSet.n_cols);
  kernels.fill(-DBL_MAX);

  // Naive implementation.
  if (naive)
  {
    // Simple double loop.  Stupid, slow, but a good benchmark.
    for (size_t q = 0; q < referenceSet.n_cols; ++q)
    {
      for (size_t r = 0; r < referenceSet.n_cols; ++r)
      {
        if (q == r)
          continue; // Don't return the point as its own candidate.

        const double eval = metric.Kernel().Evaluate(referenceSet.col(q),
                                                     referenceSet.col(r));

        size_t insertPosition;
        for (insertPosition = 0; insertPosition < indices.n_rows;
            ++insertPosition)
          if (eval > kernels(insertPosition, q))
            break;

        if (insertPosition < indices.n_rows)
          InsertNeighbor(indices, kernels, q, insertPosition, r, eval);
      }
    }

    Timer::Stop("computing_products");

    return;
  }

  // Single-tree implementation.
  if (singleMode)
  {
    // Create rules object (this will store the results).  This constructor
    // precalculates each self-kernel value.
    typedef FastMKSRules<KernelType, Tree> RuleType;
    RuleType rules(referenceSet, referenceSet, indices, kernels,
        metric.Kernel());

    typename Tree::template SingleTreeTraverser<RuleType> traverser(rules);

    for (size_t i = 0; i < referenceSet.n_cols; ++i)
      traverser.Traverse(i, *referenceTree);

    // Save the number of pruned nodes.
    const size_t numPrunes = traverser.NumPrunes();

    Log::Info << "Pruned " << numPrunes << " nodes." << std::endl;

    Log::Info << rules.BaseCases() << " base cases." << std::endl;
    Log::Info << rules.Scores() << " scores." << std::endl;

    Timer::Stop("computing_products");
    return;
  }

  // Dual-tree implementation.
  Timer::Stop("computing_products");

  Search(referenceTree, k, indices, kernels);
}

/**
 * Helper function to insert a point into the neighbors and distances matrices.
 *
 * @param queryIndex Index of point whose neighbors we are inserting into.
 * @param pos Position in list to insert into.
 * @param neighbor Index of reference point which is being inserted.
 * @param distance Distance from query point to reference point.
 */
template<typename KernelType,
         typename MatType,
         template<typename TMetricType, typename StatisticType, typename TMatType>
             class TreeType>
void FastMKS<KernelType, MatType, TreeType>::InsertNeighbor(
    arma::Mat<size_t>& indices,
    arma::mat& products,
    const size_t queryIndex,
    const size_t pos,
    const size_t neighbor,
    const double distance)
{
  // We only memmove() if there is actually a need to shift something.
  if (pos < (products.n_rows - 1))
  {
    int len = (products.n_rows - 1) - pos;
    memmove(products.colptr(queryIndex) + (pos + 1),
        products.colptr(queryIndex) + pos,
        sizeof(double) * len);
    memmove(indices.colptr(queryIndex) + (pos + 1),
        indices.colptr(queryIndex) + pos,
        sizeof(size_t) * len);
  }

  // Now put the new information in the right index.
  products(pos, queryIndex) = distance;
  indices(pos, queryIndex) = neighbor;
}

// Return string of object.
template<typename KernelType,
         typename MatType,
         template<typename TMetricType, typename StatisticType, typename TMatType>
             class TreeType>
std::string FastMKS<KernelType, MatType, TreeType>::ToString() const
{
  std::ostringstream convert;
  convert << "FastMKS [" << this << "]" << std::endl;
  convert << "  Naive: " << naive << std::endl;
  convert << "  Single: " << singleMode << std::endl;
  convert << "  Metric: " << std::endl;
  convert << mlpack::util::Indent(metric.ToString(),2);
  convert << std::endl;
  return convert.str();
}

}; // namespace fastmks
}; // namespace mlpack

#endif
