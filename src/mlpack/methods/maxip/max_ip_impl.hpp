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

#include <mlpack/core/tree/traversers/single_tree_breadth_first_traverser.hpp>
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
        const double eval = KernelType::Evaluate(querySet.col(q),
                                                 referenceSet.col(r));

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
    MaxIPRules<IPMetric<KernelType> > rules(referenceSet, querySet, indices,
        products);

    tree::SingleTreeBreadthFirstTraverser<
        tree::CoverTree<IPMetric<KernelType> >,
        MaxIPRules<IPMetric<KernelType> > > traverser(rules);

    for (size_t i = 0; i < querySet.n_cols; ++i)
      traverser.Traverse(i, *referenceTree);

    Log::Info << "Pruned " << traverser.NumPrunes() << " nodes." << std::endl;

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
