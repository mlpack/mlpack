/**
 * @file fastmks_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the FastMKS class (fast max-kernel search).
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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

// Single dataset, no instantiated kernel.
template<typename KernelType, typename TreeType>
FastMKS<KernelType, TreeType>::FastMKS(const arma::mat& referenceSet,
                                       const bool single,
                                       const bool naive) :
    referenceSet(referenceSet),
    querySet(referenceSet),
    referenceTree(NULL),
    queryTree(NULL),
    treeOwner(true),
    single(single),
    naive(naive)
{
  Timer::Start("tree_building");

  if (!naive)
    referenceTree = new TreeType(referenceSet);

  if (!naive && !single)
    queryTree = new TreeType(referenceSet);

  Timer::Stop("tree_building");
}

// Two datasets, no instantiated kernel.
template<typename KernelType, typename TreeType>
FastMKS<KernelType, TreeType>::FastMKS(const arma::mat& referenceSet,
                                       const arma::mat& querySet,
                                       const bool single,
                                       const bool naive) :
    referenceSet(referenceSet),
    querySet(querySet),
    referenceTree(NULL),
    queryTree(NULL),
    treeOwner(true),
    single(single),
    naive(naive)
{
  Timer::Start("tree_building");

  // If necessary, the trees should be built.
  if (!naive)
    referenceTree = new TreeType(referenceSet);

  if (!naive && !single)
    queryTree = new TreeType(querySet);

  Timer::Stop("tree_building");
}

// One dataset, instantiated kernel.
template<typename KernelType, typename TreeType>
FastMKS<KernelType, TreeType>::FastMKS(const arma::mat& referenceSet,
                                       KernelType& kernel,
                                       const bool single,
                                       const bool naive) :
    referenceSet(referenceSet),
    querySet(referenceSet),
    referenceTree(NULL),
    queryTree(NULL),
    treeOwner(true),
    single(single),
    naive(naive),
    metric(kernel)
{
  Timer::Start("tree_building");

  // If necessary, the reference tree should be built.  There is no query tree.
  if (!naive)
    referenceTree = new TreeType(referenceSet, metric);

  if (!naive && !single)
    queryTree = new TreeType(referenceSet, metric);

  Timer::Stop("tree_building");
}

// Two datasets, instantiated kernel.
template<typename KernelType, typename TreeType>
FastMKS<KernelType, TreeType>::FastMKS(const arma::mat& referenceSet,
                                       const arma::mat& querySet,
                                       KernelType& kernel,
                                       const bool single,
                                       const bool naive) :
    referenceSet(referenceSet),
    querySet(querySet),
    referenceTree(NULL),
    queryTree(NULL),
    treeOwner(true),
    single(single),
    naive(naive),
    metric(kernel)
{
  Timer::Start("tree_building");

  // If necessary, the trees should be built.
  if (!naive)
    referenceTree = new TreeType(referenceSet, metric);

  if (!naive && !single)
    queryTree = new TreeType(querySet, metric);

  Timer::Stop("tree_building");
}

// One dataset, pre-built tree.
template<typename KernelType, typename TreeType>
FastMKS<KernelType, TreeType>::FastMKS(const arma::mat& referenceSet,
                                       TreeType* referenceTree,
                                       const bool single,
                                       const bool naive) :
    referenceSet(referenceSet),
    querySet(referenceSet),
    referenceTree(referenceTree),
    queryTree(NULL),
    treeOwner(false),
    single(single),
    naive(naive),
    metric(referenceTree->Metric())
{
  // The query tree cannot be the same as the reference tree.
  if (referenceTree)
    queryTree = new TreeType(*referenceTree);
}

// Two datasets, pre-built trees.
template<typename KernelType, typename TreeType>
FastMKS<KernelType, TreeType>::FastMKS(const arma::mat& referenceSet,
                                       TreeType* referenceTree,
                                       const arma::mat& querySet,
                                       TreeType* queryTree,
                                       const bool single,
                                       const bool naive) :
    referenceSet(referenceSet),
    querySet(querySet),
    referenceTree(referenceTree),
    queryTree(queryTree),
    treeOwner(false),
    single(single),
    naive(naive),
    metric(referenceTree->Metric())
{
  // Nothing to do.
}

template<typename KernelType, typename TreeType>
FastMKS<KernelType, TreeType>::~FastMKS()
{
  // If we created the trees, we must delete them.
  if (treeOwner)
  {
    if (queryTree)
      delete queryTree;
    if (referenceTree)
      delete referenceTree;
  }
  else if (&querySet == &referenceSet)
  {
    // The user passed in a reference tree which we needed to copy.
    if (queryTree)
      delete queryTree;
  }
}

template<typename KernelType, typename TreeType>
void FastMKS<KernelType, TreeType>::Search(const size_t k,
                                           arma::Mat<size_t>& indices,
                                           arma::mat& products)
{
  // No remapping will be necessary because we are using the cover tree.
  indices.set_size(k, querySet.n_cols);
  products.set_size(k, querySet.n_cols);
  products.fill(-DBL_MAX);

  Timer::Start("computing_products");

  // Naive implementation.
  if (naive)
  {
    // Simple double loop.  Stupid, slow, but a good benchmark.
    for (size_t q = 0; q < querySet.n_cols; ++q)
    {
      for (size_t r = 0; r < referenceSet.n_cols; ++r)
      {
        if ((&querySet == &referenceSet) && (q == r))
          continue;

        const double eval = metric.Kernel().Evaluate(querySet.unsafe_col(q),
            referenceSet.unsafe_col(r));

        size_t insertPosition;
        for (insertPosition = 0; insertPosition < indices.n_rows;
            ++insertPosition)
          if (eval > products(insertPosition, q))
            break;

        if (insertPosition < indices.n_rows)
          InsertNeighbor(indices, products, q, insertPosition, r, eval);
      }
    }

    Timer::Stop("computing_products");

    return;
  }

  // Single-tree implementation.
  if (single)
  {
    // Create rules object (this will store the results).  This constructor
    // precalculates each self-kernel value.
    typedef FastMKSRules<KernelType, TreeType> RuleType;
    RuleType rules(referenceSet, querySet, indices, products, metric.Kernel());

    typename TreeType::template SingleTreeTraverser<RuleType> traverser(rules);

    for (size_t i = 0; i < querySet.n_cols; ++i)
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
  typedef FastMKSRules<KernelType, TreeType> RuleType;
  RuleType rules(referenceSet, querySet, indices, products, metric.Kernel());

  typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);

  traverser.Traverse(*queryTree, *referenceTree);

  const size_t numPrunes = traverser.NumPrunes();

  Log::Info << "Pruned " << numPrunes << " nodes." << std::endl;
  Log::Info << rules.BaseCases() << " base cases." << std::endl;
  Log::Info << rules.Scores() << " scores." << std::endl;

  Timer::Stop("computing_products");
  return;
}

/**
 * Helper function to insert a point into the neighbors and distances matrices.
 *
 * @param queryIndex Index of point whose neighbors we are inserting into.
 * @param pos Position in list to insert into.
 * @param neighbor Index of reference point which is being inserted.
 * @param distance Distance from query point to reference point.
 */
template<typename KernelType, typename TreeType>
void FastMKS<KernelType, TreeType>::InsertNeighbor(arma::Mat<size_t>& indices,
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
template<typename KernelType, typename TreeType>
std::string FastMKS<KernelType, TreeType>::ToString() const
{
  std::ostringstream convert;
  convert << "FastMKS [" << this << "]" << std::endl;
  convert << "  Naive: " << naive << std::endl;
  convert << "  Single: " << single << std::endl;
  convert << "  Metric: " << std::endl;
  convert << mlpack::util::Indent(metric.ToString(),2);
  convert << std::endl;
  return convert.str();
}

// Specialized implementation for tighter bounds for Gaussian.
/*
template<>
void FastMKS<kernel::GaussianKernel>::Search(const size_t k,
                                             arma::Mat<size_t>& indices,
                                             arma::mat& products)
{
  Log::Warn << "Alternate implementation!" << std::endl;

  // Terrible copypasta.  Bad bad bad.
  // No remapping will be necessary.
  indices.set_size(k, querySet.n_cols);
  products.set_size(k, querySet.n_cols);
  products.fill(-1.0);

  Timer::Start("computing_products");

  size_t kernelEvaluations = 0;

  // Naive implementation.
  if (naive)
  {
    // Simple double loop.  Stupid, slow, but a good benchmark.
    for (size_t q = 0; q < querySet.n_cols; ++q)
    {
      for (size_t r = 0; r < referenceSet.n_cols; ++r)
      {
        const double eval = metric.Kernel().Evaluate(querySet.unsafe_col(q),
            referenceSet.unsafe_col(r));
        ++kernelEvaluations;

        size_t insertPosition;
        for (insertPosition = 0; insertPosition < indices.n_rows;
            ++insertPosition)
          if (eval > products(insertPosition, q))
            break;

        if (insertPosition < indices.n_rows)
          InsertNeighbor(indices, products, q, insertPosition, r, eval);
      }
    }

    Timer::Stop("computing_products");

    Log::Info << "Kernel evaluations: " << kernelEvaluations << "." << std::endl;
    return;
  }

  // Single-tree implementation.
  if (single)
  {
    // Calculate number of pruned nodes.
    size_t numPrunes = 0;

    // Precalculate query products ( || q || for all q).
    arma::vec queryProducts(querySet.n_cols);
    for (size_t queryIndex = 0; queryIndex < querySet.n_cols; ++queryIndex)
      queryProducts[queryIndex] = sqrt(metric.Kernel().Evaluate(
          querySet.unsafe_col(queryIndex), querySet.unsafe_col(queryIndex)));
    kernelEvaluations += querySet.n_cols;

    // Screw the CoverTreeTraverser, we'll implement it by hand.
    for (size_t queryIndex = 0; queryIndex < querySet.n_cols; ++queryIndex)
    {
      // Use an array of priority queues?
      std::priority_queue<
          SearchFrame<tree::CoverTree<IPMetric<kernel::GaussianKernel> > >,
          std::vector<SearchFrame<tree::CoverTree<IPMetric<
              kernel::GaussianKernel> > > >,
          SearchFrameCompare<tree::CoverTree<IPMetric<
              kernel::GaussianKernel> > > >
          frameQueue;

      // Add initial frame.
      SearchFrame<tree::CoverTree<IPMetric<kernel::GaussianKernel> > >
          nextFrame;
      nextFrame.node = referenceTree;
      nextFrame.eval = metric.Kernel().Evaluate(querySet.unsafe_col(queryIndex),
          referenceSet.unsafe_col(referenceTree->Point()));
      Log::Assert(nextFrame.eval <= 1);
      ++kernelEvaluations;

      // The initial evaluation will be the best so far.
      indices(0, queryIndex) = referenceTree->Point();
      products(0, queryIndex) = nextFrame.eval;

      frameQueue.push(nextFrame);

      tree::CoverTree<IPMetric<kernel::GaussianKernel> >* referenceNode;
      double eval;
      double maxProduct;

      while (!frameQueue.empty())
      {
        // Get the information for this node.
        const SearchFrame<tree::CoverTree<IPMetric<kernel::GaussianKernel> > >&
            frame = frameQueue.top();

        referenceNode = frame.node;
        eval = frame.eval;

        // Loop through the children, seeing if we can prune them; if not, add
        // them to the queue.  The self-child is different -- it has the same
        // parent (and therefore the same kernel evaluation).
        if (referenceNode->NumChildren() > 0)
        {
          SearchFrame<tree::CoverTree<IPMetric<kernel::GaussianKernel> > >
              childFrame;

          // We must handle the self-child differently, to avoid adding it to
          // the results twice.
          childFrame.node = &(referenceNode->Child(0));
          childFrame.eval = eval;

          // Alternate pruning rule.
          const double mdd = childFrame.node->FurthestDescendantDistance();
          if (eval >= (1 - std::pow(mdd, 2.0) / 2.0))
            maxProduct = 1;
          else
            maxProduct = eval * (1 - std::pow(mdd, 2.0) / 2.0) + sqrt(1 -
                std::pow(eval, 2.0)) * mdd * sqrt(1 - std::pow(mdd, 2.0) / 4.0);

          // Add self-child if we can't prune it.
          if (maxProduct > products(products.n_rows - 1, queryIndex))
          {
            // But only if it has children of its own.
            if (childFrame.node->NumChildren() > 0)
              frameQueue.push(childFrame);
          }
          else
            ++numPrunes;

          for (size_t i = 1; i < referenceNode->NumChildren(); ++i)
          {
            // Before we evaluate the child, let's see if it can possibly have
            // a better evaluation.
            const double mpdd = std::min(
                referenceNode->Child(i).ParentDistance() +
                referenceNode->Child(i).FurthestDescendantDistance(), 2.0);
            double maxChildEval = 1;
            if (eval < (1 - std::pow(mpdd, 2.0) / 2.0))
              maxChildEval = eval * (1 - std::pow(mpdd, 2.0) / 2.0) + sqrt(1 -
                  std::pow(eval, 2.0)) * mpdd * sqrt(1 - std::pow(mpdd, 2.0)
                  / 4.0);

            if (maxChildEval > products(products.n_rows - 1, queryIndex))
            {
              // Evaluate child.
              childFrame.node = &(referenceNode->Child(i));
              childFrame.eval = metric.Kernel().Evaluate(
                  querySet.unsafe_col(queryIndex),
                  referenceSet.unsafe_col(referenceNode->Child(i).Point()));
              ++kernelEvaluations;

              // Can we prune it?  If we can, we can avoid putting it in the
              // queue (saves time).
              const double cmdd = childFrame.node->FurthestDescendantDistance();
              if (childFrame.eval > (1 - std::pow(cmdd, 2.0) / 2.0))
                maxProduct = 1;
              else
                maxProduct = childFrame.eval * (1 - std::pow(cmdd, 2.0) / 2.0)
                    + sqrt(1 - std::pow(eval, 2.0)) * cmdd * sqrt(1 -
                    std::pow(cmdd, 2.0) / 4.0);

              if (maxProduct > products(products.n_rows - 1, queryIndex))
              {
                // Good enough to recurse into.  While we're at it, check the
                // actual evaluation and see if it's an improvement.
                if (childFrame.eval > products(products.n_rows - 1, queryIndex))
                {
                  // This is a better result.  Find out where to insert.
                  size_t insertPosition = 0;
                  for ( ; insertPosition < products.n_rows - 1;
                      ++insertPosition)
                    if (childFrame.eval > products(insertPosition, queryIndex))
                      break;

                  // Insert into the correct position; we are guaranteed that
                  // insertPosition is valid.
                  InsertNeighbor(indices, products, queryIndex, insertPosition,
                      childFrame.node->Point(), childFrame.eval);
                }

                // Now add this to the queue (if it has any children which may
                // need to be recursed into).
                if (childFrame.node->NumChildren() > 0)
                  frameQueue.push(childFrame);
              }
              else
                ++numPrunes;
            }
            else
              ++numPrunes;
          }
        }

        frameQueue.pop();
      }
    }

    Log::Info << "Pruned " << numPrunes << " nodes." << std::endl;
    Log::Info << "Kernel evaluations: " << kernelEvaluations << "."
        << std::endl;
    Log::Info << "Distance evaluations: " << distanceEvaluations << "."
        << std::endl;

    Timer::Stop("computing_products");
    return;
  }

  // Double-tree implementation.
  Log::Fatal << "Dual-tree search not implemented yet... oops..." << std::endl;

}
*/

}; // namespace fastmks
}; // namespace mlpack

#endif
