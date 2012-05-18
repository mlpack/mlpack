/**
 * @file max_ip_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the MaxIP class (maximum inner product search).
 */
#ifndef __MLPACK_METHODS_MAXIP_MAX_IP_IMPL_HPP
#define __MLPACK_METHODS_MAXIP_MAX_IP_IMPL_HPP

// In case it hasn't yet been included.
#include "max_ip.hpp"

#include "max_ip_rules.hpp"

namespace mlpack {
namespace maxip {

template<typename KernelType>
MaxIP<KernelType>::MaxIP(const arma::mat& referenceSet,
                         bool single,
                         bool naive) :
    referenceSet(referenceSet),
    querySet(referenceSet), // Gotta point it somewhere...
    queryTree(NULL),
    single(single),
    naive(naive)
{
  Timer::Start("tree_building");

  // Build the tree.  Could we do this in the initialization list?
  if (naive)
    referenceTree = NULL;
  else
    referenceTree = new tree::CoverTree<IPMetric<KernelType> >(referenceSet);

  Timer::Stop("tree_building");
}

template<typename KernelType>
MaxIP<KernelType>::MaxIP(const arma::mat& referenceSet,
                         const arma::mat& querySet,
                         bool single,
                         bool naive) :
    referenceSet(referenceSet),
    querySet(querySet),
    single(single),
    naive(naive)
{
  Timer::Start("tree_building");

  // Build the trees.  Could we do this in the initialization lists?
  if (naive)
    referenceTree = NULL;
  else
    referenceTree = new tree::CoverTree<IPMetric<KernelType> >(referenceSet);

  if (single || naive)
    queryTree = NULL;
  else
    queryTree = new tree::CoverTree<IPMetric<KernelType> >(querySet);

  Timer::Stop("tree_building");
}

template<typename KernelType>
MaxIP<KernelType>::~MaxIP()
{
  if (queryTree)
    delete queryTree;
  if (referenceTree)
    delete referenceTree;
}

template<typename KernelType>
void MaxIP<KernelType>::Search(const size_t k,
                               arma::Mat<size_t>& indices,
                               arma::mat& products)
{
  // No remapping will be necessary.
  indices.set_size(k, querySet.n_cols);
  products.set_size(k, querySet.n_cols);

  Timer::Start("computing_products");

  // Naive implementation.
  if (naive)
  {
    // Simple double loop.  Stupid, slow, but a good benchmark.
    for (size_t q = 0; q < querySet.n_cols; ++q)
    {
      for (size_t r = 0; r < referenceSet.n_cols; ++r)
      {
        const double eval = KernelType::Evaluate(querySet.unsafe_col(q),
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
    // Calculate number of pruned nodes.
    size_t numPrunes = 0;

    // Precalculate query products ( || q || for all q).
    arma::vec queryProducts(querySet.n_cols);
    for (size_t queryIndex = 0; queryIndex < querySet.n_cols; ++queryIndex)
      queryProducts[queryIndex] = KernelType::Evaluate(
          querySet.unsafe_col(queryIndex), querySet.unsafe_col(queryIndex));

    // Screw the CoverTreeTraverser, we'll implement it by hand.
    for (size_t queryIndex = 0; queryIndex < querySet.n_cols; ++queryIndex)
    {
      std::queue<tree::CoverTree<IPMetric<KernelType> >*> pointQueue;
      std::queue<size_t> parentQueue;
      std::queue<double> parentEvalQueue;
      pointQueue.push(referenceTree);
      parentQueue.push(size_t() - 1); // Has no parent.
      parentEvalQueue.push(0); // No possible parent evaluation.

      tree::CoverTree<IPMetric<KernelType> >* referenceNode;
      size_t currentParent;
      double currentParentEval;
      double eval; // Kernel evaluation.

      while (!pointQueue.empty())
      {
        // Get the information for this node.
        referenceNode = pointQueue.front();
        currentParent = parentQueue.front();
        currentParentEval = parentEvalQueue.front();

        pointQueue.pop();
        parentQueue.pop();
        parentEvalQueue.pop();

        // See if this has the same parent.
        if (referenceNode->Point() == currentParent)
        {
          // We don't have to evaluate the kernel again.
          eval = currentParentEval;
        }
        else
        {
          // Evaluate the kernel.  Then see if it is a result to keep.
          eval = KernelType::Evaluate(querySet.unsafe_col(queryIndex),
              referenceSet.unsafe_col(referenceNode->Point()));

          // Is the result good enough to be saved?
          if (eval > products(products.n_rows - 1, queryIndex))
          {
            // Figure out where to insert.
            size_t insertPosition = 0;
            for ( ; insertPosition < products.n_rows - 1; ++insertPosition)
              if (eval > products(insertPosition, queryIndex))
                break;

            // We are guaranteed that insertPosition is valid.
            InsertNeighbor(indices, products, queryIndex, insertPosition,
                referenceNode->Point(), eval);
          }
        }

        // Now discover if we can prune this node or not.
        double maxProduct = eval + std::pow(referenceNode->ExpansionConstant(),
            referenceNode->Scale() + 1) * queryProducts[queryIndex];

        if (maxProduct > products(products.n_rows - 1, queryIndex))
        {
          // We can't prune.  So add our children.
          for (size_t i = 0; i < referenceNode->NumChildren(); ++i)
          {
            pointQueue.push(&(referenceNode->Child(i)));
            parentQueue.push(referenceNode->Point());
            parentEvalQueue.push(eval);
          }
        }
        else
        {
          numPrunes++;
        }
      }
    }

    Log::Info << "Pruned " << numPrunes << " nodes." << std::endl;

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
void MaxIP<KernelType>::InsertNeighbor(arma::Mat<size_t>& indices,
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

}; // namespace maxip
}; // namespace mlpack

#endif
