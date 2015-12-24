/**
 * @file fastmks_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the FastMKS class (fast max-kernel search).
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
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

// No data; create a model on an empty dataset.
template<typename KernelType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>::FastMKS(const bool singleMode,
                                                const bool naive) :
    referenceSet(new MatType()),
    referenceTree(NULL),
    treeOwner(true),
    setOwner(true),
    singleMode(singleMode),
    naive(naive)
{
  Timer::Start("tree_building");
  if (!naive)
    referenceTree = new Tree(*referenceSet);
  Timer::Stop("tree_building");
}

// No instantiated kernel.
template<typename KernelType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>::FastMKS(
    const MatType& referenceSet,
    const bool singleMode,
    const bool naive) :
    referenceSet(&referenceSet),
    referenceTree(NULL),
    treeOwner(true),
    setOwner(false),
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
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>::FastMKS(const MatType& referenceSet,
                                                KernelType& kernel,
                                                const bool singleMode,
                                                const bool naive) :
    referenceSet(&referenceSet),
    referenceTree(NULL),
    treeOwner(true),
    setOwner(false),
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
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>::FastMKS(Tree* referenceTree,
                                                const bool singleMode) :
    referenceSet(&referenceTree->Dataset()),
    referenceTree(referenceTree),
    treeOwner(false),
    setOwner(false),
    singleMode(singleMode),
    naive(false),
    metric(referenceTree->Metric())
{
  // Nothing to do.
}

template<typename KernelType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>::~FastMKS()
{
  // If we created the trees, we must delete them.
  if (treeOwner && referenceTree)
    delete referenceTree;
  if (setOwner)
    delete referenceSet;
}

template<typename KernelType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Train(const MatType& referenceSet)
{
  if (setOwner)
    delete this->referenceSet;

  this->referenceSet = &referenceSet;
  this->setOwner = false;

  if (!naive)
  {
    if (treeOwner && referenceTree)
      delete referenceTree;
    referenceTree = new Tree(referenceSet, metric);
    treeOwner = true;
  }
}

template<typename KernelType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Train(const MatType& referenceSet,
                                                   KernelType& kernel)
{
  if (setOwner)
    delete this->referenceSet;

  this->referenceSet = &referenceSet;
  this->metric = metric::IPMetric<KernelType>(kernel);
  this->setOwner = false;

  if (!naive)
  {
    if (treeOwner && referenceTree)
      delete referenceTree;
    referenceTree = new Tree(referenceSet, metric);
    treeOwner = true;
  }
}

template<typename KernelType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Train(Tree* tree)
{
  if (naive)
    throw std::invalid_argument("cannot call FastMKS::Train() with a tree when "
        "in naive search mode");

  if (setOwner)
    delete this->referenceSet;

  this->referenceSet = &tree->Dataset();
  this->metric = metric::IPMetric<KernelType>(tree->Metric().Kernel());
  this->setOwner = false;

  if (treeOwner && referenceTree)
    delete referenceTree;

  this->referenceTree = tree;
  this->treeOwner = true;
}

template<typename KernelType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Search(
    const MatType& querySet,
    const size_t k,
    arma::Mat<size_t>& indices,
    arma::mat& kernels)
{
  if (k > referenceSet->n_cols)
  {
    std::stringstream ss;
    ss << "requested value of k (" << k << ") is greater than the number of "
        << "points in the reference set (" << referenceSet->n_cols << ")";
    throw std::invalid_argument(ss.str());
  }

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
      for (size_t r = 0; r < referenceSet->n_cols; ++r)
      {
        const double eval = metric.Kernel().Evaluate(querySet.col(q),
                                                     referenceSet->col(r));

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
    RuleType rules(*referenceSet, querySet, indices, kernels, metric.Kernel());

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
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Search(
    Tree* queryTree,
    const size_t k,
    arma::Mat<size_t>& indices,
    arma::mat& kernels)
{
  if (k > referenceSet->n_cols)
  {
    std::stringstream ss;
    ss << "requested value of k (" << k << ") is greater than the number of "
        << "points in the reference set (" << referenceSet->n_cols << ")";
    throw std::invalid_argument(ss.str());
  }

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
  RuleType rules(*referenceSet, queryTree->Dataset(), indices, kernels,
      metric.Kernel());

  typename Tree::template DualTreeTraverser<RuleType> traverser(rules);

  traverser.Traverse(*queryTree, *referenceTree);

  Log::Info << rules.BaseCases() << " base cases." << std::endl;
  Log::Info << rules.Scores() << " scores." << std::endl;

  Timer::Stop("computing_products");
}

template<typename KernelType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Search(
    const size_t k,
    arma::Mat<size_t>& indices,
    arma::mat& kernels)
{
  // No remapping will be necessary because we are using the cover tree.
  Timer::Start("computing_products");
  indices.set_size(k, referenceSet->n_cols);
  kernels.set_size(k, referenceSet->n_cols);
  kernels.fill(-DBL_MAX);

  // Naive implementation.
  if (naive)
  {
    // Simple double loop.  Stupid, slow, but a good benchmark.
    for (size_t q = 0; q < referenceSet->n_cols; ++q)
    {
      for (size_t r = 0; r < referenceSet->n_cols; ++r)
      {
        if (q == r)
          continue; // Don't return the point as its own candidate.

        const double eval = metric.Kernel().Evaluate(referenceSet->col(q),
                                                     referenceSet->col(r));

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
    RuleType rules(*referenceSet, *referenceSet, indices, kernels,
        metric.Kernel());

    typename Tree::template SingleTreeTraverser<RuleType> traverser(rules);

    for (size_t i = 0; i < referenceSet->n_cols; ++i)
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
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
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

//! Serialize the model.
template<typename KernelType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
template<typename Archive>
void FastMKS<KernelType, MatType, TreeType>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  using data::CreateNVP;

  // Serialize preferences for search.
  ar & CreateNVP(naive, "naive");
  ar & CreateNVP(singleMode, "singleMode");

  // If we are doing naive search, serialize the dataset.  Otherwise we
  // serialize the tree.
  if (naive)
  {
    if (Archive::is_loading::value)
    {
      if (setOwner && referenceSet)
        delete referenceSet;

      setOwner = true;
    }

    ar & CreateNVP(referenceSet, "referenceSet");
    ar & CreateNVP(metric, "metric");
  }
  else
  {
    // Delete the current reference tree, if necessary.
    if (Archive::is_loading::value)
    {
      if (treeOwner && referenceTree)
        delete referenceTree;

      treeOwner = true;
    }

    ar & CreateNVP(referenceTree, "referenceTree");

    if (Archive::is_loading::value)
    {
      if (setOwner && referenceSet)
        delete referenceSet;

      referenceSet = &referenceTree->Dataset();
      metric = metric::IPMetric<KernelType>(referenceTree->Metric().Kernel());
      setOwner = false;
    }
  }
}

} // namespace fastmks
} // namespace mlpack

#endif
