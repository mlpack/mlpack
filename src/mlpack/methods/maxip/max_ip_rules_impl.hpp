/**
 * @file max_ip_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of MaxIPRules for cover tree search.
 */
#ifndef __MLPACK_METHODS_MAXIP_MAX_IP_RULES_IMPL_HPP
#define __MLPACK_METHODS_MAXIP_MAX_IP_RULES_IMPL_HPP

// In case it hasn't already been included.
#include "max_ip_rules.hpp"

namespace mlpack {
namespace maxip {

template<typename MetricType>
MaxIPRules<MetricType>::MaxIPRules(const arma::mat& referenceSet,
                                   const arma::mat& querySet,
                                   arma::Mat<size_t>& indices,
                                   arma::mat& products) :
    referenceSet(referenceSet),
    querySet(querySet),
    indices(indices),
    products(products)
{ /* Nothing left to do. */ }

template<typename MetricType>
void MaxIPRules<MetricType>::BaseCase(const size_t queryIndex,
                                      const size_t referenceIndex)
{
  const double eval = MetricType::Kernel::Evaluate(querySet.col(queryIndex),
      referenceSet.col(referenceIndex));

  if (eval > products(products.n_rows - 1, queryIndex))
  {
    size_t insertPosition;
    for (insertPosition = 0; insertPosition < indices.n_rows; ++insertPosition)
      if (eval > products(insertPosition, queryIndex))
        break;

    // We are guaranteed insertPosition is in the valid range.
    InsertNeighbor(queryIndex, insertPosition, referenceIndex, eval);
  }
}

template<typename MetricType>
bool MaxIPRules<MetricType>::CanPrune(const size_t queryIndex,
    tree::CoverTree<MetricType>& referenceNode)
{
  // The maximum possible inner product is given by
  //   <q, p_0> + R_p || q ||
  // and since we are using cover trees, p_0 is the point referred to by the
  // node, and R_p will be the expansion constant to the power of the scale plus
  // one.
  double maxProduct = MetricType::Kernel::Evaluate(querySet.col(queryIndex),
      referenceSet.col(referenceNode.Point()));

  maxProduct += std::pow(referenceNode.ExpansionConstant(),
      referenceNode.Scale() + 1) *
      sqrt(MetricType::Kernel::Evaluate(querySet.col(queryIndex),
      querySet.col(queryIndex)));

  if (maxProduct > products(products.n_rows - 1, queryIndex))
    return false;
  else
    return true;
}

/**
 * Helper function to insert a point into the neighbors and distances matrices.
 *
 * @param queryIndex Index of point whose neighbors we are inserting into.
 * @param pos Position in list to insert into.
 * @param neighbor Index of reference point which is being inserted.
 * @param distance Distance from query point to reference point.
 */
template<typename MetricType>
void MaxIPRules<MetricType>::InsertNeighbor(const size_t queryIndex,
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
