/**
 * @file fastmks_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the FastMKS class (fast max-kernel search).
 *
 * This file is part of MLPACK 1.0.4.
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

template<typename TreeType>
void RecurseTreeCountLeaves(const TreeType& node, arma::vec& counts)
{
  for (size_t i = 0; i < node.NumChildren(); ++i)
  {
    if (node.Child(i).NumChildren() == 0)
      counts[node.Child(i).Point()]++;
    else
      RecurseTreeCountLeaves<TreeType>(node.Child(i), counts);
  }
}

template<typename TreeType>
void CheckSelfChild(const TreeType& node)
{
  if (node.NumChildren() == 0)
    return; // No self-child applicable here.

  bool found = false;
  for (size_t i = 0; i < node.NumChildren(); ++i)
  {
    if (node.Child(i).Point() == node.Point())
      found = true;

    // Recursively check the children.
    CheckSelfChild(node.Child(i));
  }

  // Ensure this has its own self-child.
  Log::Assert(found == true);
}

template<typename TreeType, typename MetricType>
void CheckCovering(const TreeType& node)
{
  // Return if a leaf.  No checking necessary.
  if (node.NumChildren() == 0)
    return;

  const arma::mat& dataset = node.Dataset();
  const size_t nodePoint = node.Point();

  // To ensure that this node satisfies the covering principle, we must ensure
  // that the distance to each child is less than pow(expansionConstant, scale).
  double maxDistance = pow(node.Base(), node.Scale());
  for (size_t i = 0; i < node.NumChildren(); ++i)
  {
    const size_t childPoint = node.Child(i).Point();

    double distance = MetricType::Evaluate(dataset.col(nodePoint),
        dataset.col(childPoint));

    Log::Assert(distance <= maxDistance);

    // Check the child.
    CheckCovering<TreeType, MetricType>(node.Child(i));
  }
}

template<typename TreeType, typename MetricType>
void CheckIndividualSeparation(const TreeType& constantNode,
                               const TreeType& node)
{
  // Don't check points at a lower scale.
  if (node.Scale() < constantNode.Scale())
    return;

  // If at a higher scale, recurse.
  if (node.Scale() > constantNode.Scale())
  {
    for (size_t i = 0; i < node.NumChildren(); ++i)
    {
      // Don't recurse into leaves.
      if (node.Child(i).NumChildren() > 0)
        CheckIndividualSeparation<TreeType, MetricType>(constantNode,
            node.Child(i));
    }

    return;
  }

  // Don't compare the same point against itself.
  if (node.Point() == constantNode.Point())
    return;

  // Now we know we are at the same scale, so make the comparison.
  const arma::mat& dataset = constantNode.Dataset();
  const size_t constantPoint = constantNode.Point();
  const size_t nodePoint = node.Point();

  // Make sure the distance is at least the following value (in accordance with
  // the separation principle of cover trees).
  double minDistance = pow(constantNode.ExpansionConstant(),
      constantNode.Scale());

  double distance = MetricType::Evaluate(dataset.col(constantPoint),
      dataset.col(nodePoint));

}

template<typename TreeType, typename MetricType>
void CheckSeparation(const TreeType& node, const TreeType& root)
{
  // Check the separation between this point and all other points on this scale.
  CheckIndividualSeparation<TreeType, MetricType>(node, root);

  // Check the children, but only if they are not leaves.  Leaves don't need to
  // be checked.
  for (size_t i = 0; i < node.NumChildren(); ++i)
    if (node.Child(i).NumChildren() > 0)
      CheckSeparation<TreeType, MetricType>(node.Child(i), root);
}

template<typename TreeType, typename MetricType>
void GetMaxDistance(TreeType& node,
                    TreeType& constantNode,
                    double& best,
                    size_t& index)
{
  const arma::mat& dataset = node.Dataset();
  const double eval = MetricType::Evaluate(dataset.unsafe_col(node.Point()),
      dataset.unsafe_col(constantNode.Point()));
  if (eval > best)
  {
    best = eval;
    index = node.Point();
  }

  // Recurse into children.
  for (size_t i = 0; i < node.NumChildren(); ++i)
    GetMaxDistance<TreeType, MetricType>(node.Child(i), constantNode, best,
        index);
}

template<typename TreeType, typename MetricType>
void CheckMaxDistances(TreeType& node)
{
  // Check child distances.
  for (size_t i = 0; i < node.NumChildren(); ++i)
  {
    const arma::mat& dataset = node.Dataset();
    double eval = MetricType::Evaluate(dataset.unsafe_col(node.Point()),
        dataset.unsafe_col(node.Child(i).Point()));

    Log::Assert(std::abs(eval - node.Child(i).ParentDistance()) < 1e-10);
  }

  // Check all descendants.
  double maxDescendantDistance = 0;
  size_t maxIndex = 0;
  GetMaxDistance<TreeType, MetricType>(node, node, maxDescendantDistance,
      maxIndex);

  Log::Assert(std::abs(maxDescendantDistance -
      node.FurthestDescendantDistance()) < 1e-10);

  for (size_t i = 0; i < node.NumChildren(); ++i)
    CheckMaxDistances<TreeType, MetricType>(node.Child(i));
}

template<typename TreeType>
struct SearchFrame
{
  TreeType* node;
  double eval;
};

template<typename TreeType>
class SearchFrameCompare
{
 public:
  bool operator()(const SearchFrame<TreeType>& lhs,
                  const SearchFrame<TreeType>& rhs)
  {
    // Compare scale.
    if (lhs.node->Scale() != rhs.node->Scale())
      return (lhs.node->Scale() < rhs.node->Scale());
    else
    {
      // Now we have to compare by evaluation.
      return (lhs.eval < rhs.eval);
    }
  }
};

template<typename KernelType>
FastMKS<KernelType>::FastMKS(const arma::mat& referenceSet,
                             KernelType& kernel,
                             bool single,
                             bool naive,
                             double expansionConstant) :
    referenceSet(referenceSet),
    querySet(referenceSet), // Gotta point it somewhere...
    queryTree(NULL),
    single(single),
    naive(naive),
    metric(kernel)
{

  Timer::Start("tree_building");

  // Build the tree.  Could we do this in the initialization list?
  if (naive)
    referenceTree = NULL;
  else
    referenceTree = new tree::CoverTree<IPMetric<KernelType> >(referenceSet,
        expansionConstant, &metric);

  Timer::Stop("tree_building");
}

template<typename KernelType>
FastMKS<KernelType>::FastMKS(const arma::mat& referenceSet,
                             const arma::mat& querySet,
                             KernelType& kernel,
                             bool single,
                             bool naive,
                             double expansionConstant) :
    referenceSet(referenceSet),
    querySet(querySet),
    single(single),
    naive(naive),
    metric(kernel)
{
  Timer::Start("tree_building");

  // Build the trees.  Could we do this in the initialization lists?
  if (naive)
    referenceTree = NULL;
  else
    referenceTree = new tree::CoverTree<IPMetric<KernelType> >(referenceSet,
        expansionConstant, &metric);

  if (single || naive)
    queryTree = NULL;
  else
    queryTree = new tree::CoverTree<IPMetric<KernelType> >(querySet,
        expansionConstant, &metric);

/*  if (referenceTree != NULL)
  {
    Log::Debug << "Check counts" << std::endl;
    // Now loop through the tree and ensure that each leaf is only created once.
    arma::vec counts;
    counts.zeros(referenceSet.n_elem);
    RecurseTreeCountLeaves(*referenceTree, counts);

    // Each point should only have one leaf node representing it.
    for (size_t i = 0; i < 20; ++i)
      Log::Assert(counts[i] == 1);

    Log::Debug << "Check self child\n";
    // Each non-leaf should have a self-child.
    CheckSelfChild<tree::CoverTree<IPMetric<KernelType> > >(*referenceTree);

    Log::Debug << "Check covering\n";
    // Each node must satisfy the covering principle (its children must be less
    // than or equal to a certain distance apart).
    CheckCovering<tree::CoverTree<IPMetric<KernelType> >, IPMetric<KernelType> >(*referenceTree);

    Log::Debug << "Check max distances\n";
    // Check maximum distance of children and grandchildren.
    CheckMaxDistances<tree::CoverTree<IPMetric<KernelType> >, IPMetric<KernelType> >(*referenceTree);
    Log::Debug << "Done\n";
  }*/

  Timer::Stop("tree_building");
}

template<typename KernelType>
FastMKS<KernelType>::~FastMKS()
{
  if (queryTree)
    delete queryTree;
  if (referenceTree)
    delete referenceTree;
}

template<typename KernelType>
void FastMKS<KernelType>::Search(const size_t k,
                               arma::Mat<size_t>& indices,
                               arma::mat& products)
{
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

    Log::Info << "Kernel evaluations: " << kernelEvaluations << "."
        << std::endl;
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
          SearchFrame<tree::CoverTree<IPMetric<KernelType> > >,
          std::vector<SearchFrame<tree::CoverTree<IPMetric<KernelType> > > >,
          SearchFrameCompare<tree::CoverTree<IPMetric<KernelType> > > >
          frameQueue;

      // Add initial frame.
      SearchFrame<tree::CoverTree<IPMetric<KernelType> > > nextFrame;
      nextFrame.node = referenceTree;
      nextFrame.eval = metric.Kernel().Evaluate(querySet.unsafe_col(queryIndex),
          referenceSet.unsafe_col(referenceTree->Point()));
      ++kernelEvaluations;

      // The initial evaluation will be the best so far.
      indices(0, queryIndex) = referenceTree->Point();
      products(0, queryIndex) = nextFrame.eval;

      frameQueue.push(nextFrame);

      tree::CoverTree<IPMetric<KernelType> >* referenceNode;
      double eval;
      double maxProduct;

      while (!frameQueue.empty())
      {
        // Get the information for this node.
        const SearchFrame<tree::CoverTree<IPMetric<KernelType> > >& frame =
            frameQueue.top();

        referenceNode = frame.node;
        eval = frame.eval;

        // Loop through the children, seeing if we can prune them; if not, add
        // them to the queue.  The self-child is different -- it has the same
        // parent (and therefore the same kernel evaluation).
        if (referenceNode->NumChildren() > 0)
        {
          SearchFrame<tree::CoverTree<IPMetric<KernelType> > > childFrame;

          // We must handle the self-child differently, to avoid adding it to
          // the results twice.
          childFrame.node = &(referenceNode->Child(0));
          childFrame.eval = eval;

//          maxProduct = eval + std::pow(childFrame.node->ExpansionConstant(),
//              childFrame.node->Scale() + 1) * queryProducts[queryIndex];
          // Alternate pruning rule.
          maxProduct = eval + childFrame.node->FurthestDescendantDistance() *
              queryProducts[queryIndex];

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
            double maxChildEval = eval + queryProducts[queryIndex] *
                (referenceNode->Child(i).ParentDistance() +
                referenceNode->Child(i).FurthestDescendantDistance());

            if (maxChildEval <= products(products.n_rows - 1, queryIndex))
            {
              ++numPrunes;
              continue; // Skip this child; it can't be any better.
            }

            // Evaluate child.
            childFrame.node = &(referenceNode->Child(i));
            childFrame.eval = metric.Kernel().Evaluate(
                querySet.unsafe_col(queryIndex),
                referenceSet.unsafe_col(referenceNode->Child(i).Point()));
            ++kernelEvaluations;

            // Can we prune it?  If we can, we can avoid putting it in the queue
            // (saves time).
//            double maxProduct = childFrame.eval +
//                std::pow(childFrame.node->ExpansionConstant(),
//                childFrame.node->Scale() + 1) * queryProducts[queryIndex];
            // Alternate pruning rule.
            maxProduct = childFrame.eval + queryProducts[queryIndex] *
                childFrame.node->FurthestDescendantDistance();

            if (maxProduct > products(products.n_rows - 1, queryIndex))
            {
              // Good enough to recurse into.  While we're at it, check the
              // actual evaluation and see if it's an improvement.
              if (childFrame.eval > products(products.n_rows - 1, queryIndex))
              {
                // This is a better result.  Find out where to insert.
                size_t insertPosition = 0;
                for ( ; insertPosition < products.n_rows - 1; ++insertPosition)
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
            {
              ++numPrunes;
            }
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

/**
 * Helper function to insert a point into the neighbors and distances matrices.
 *
 * @param queryIndex Index of point whose neighbors we are inserting into.
 * @param pos Position in list to insert into.
 * @param neighbor Index of reference point which is being inserted.
 * @param distance Distance from query point to reference point.
 */
template<typename KernelType>
void FastMKS<KernelType>::InsertNeighbor(arma::Mat<size_t>& indices,
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

// Specialized implementation for tighter bounds for Gaussian.
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

}; // namespace fastmks
}; // namespace mlpack

#endif
