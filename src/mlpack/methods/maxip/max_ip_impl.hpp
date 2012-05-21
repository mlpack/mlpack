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
  products.fill(DBL_MIN);

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
        const double eval = KernelType::Evaluate(querySet.unsafe_col(q),
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
      queryProducts[queryIndex] = sqrt(KernelType::Evaluate(
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
      nextFrame.eval = KernelType::Evaluate(querySet.unsafe_col(queryIndex),
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

          maxProduct = eval + std::pow(childFrame.node->ExpansionConstant(),
              childFrame.node->Scale() + 1) * queryProducts[queryIndex];

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
            // Evaluate child.
            childFrame.node = &(referenceNode->Child(i));
            childFrame.eval = KernelType::Evaluate(
                querySet.unsafe_col(queryIndex),
                referenceSet.unsafe_col(referenceNode->Child(i).Point()));
            ++kernelEvaluations;

            // Can we prune it?  If we can, we can avoid putting it in the queue
            // (saves time).
            double maxProduct = childFrame.eval +
                std::pow(childFrame.node->ExpansionConstant(),
                childFrame.node->Scale() + 1) * queryProducts[queryIndex];

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
