/**
 * @file fastmks_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of FastMKSRules for cover tree search.
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
#ifndef __MLPACK_METHODS_FASTMKS_FASTMKS_RULES_IMPL_HPP
#define __MLPACK_METHODS_FASTMKS_FASTMKS_RULES_IMPL_HPP

// In case it hasn't already been included.
#include "fastmks_rules.hpp"

namespace mlpack {
namespace fastmks {

template<typename MetricType>
FastMKSRules<MetricType>::FastMKSRules(const arma::mat& referenceSet,
                                       const arma::mat& querySet,
                                       arma::Mat<size_t>& indices,
                                       arma::mat& products) :
    referenceSet(referenceSet),
    querySet(querySet),
    indices(indices),
    products(products)
{
  // Precompute each self-kernel.
//  queryKernels.set_size(querySet.n_cols);
//  for (size_t i = 0; i < querySet.n_cols; ++i)
//   queryKernels[i] = sqrt(MetricType::Kernel::Evaluate(querySet.unsafe_col(i),
//        querySet.unsafe_col(i)));
}

template<typename MetricType>
bool FastMKSRules<MetricType>::CanPrune(const size_t queryIndex,
    tree::CoverTree<MetricType>& referenceNode,
    const size_t parentIndex)
{
  // The maximum possible inner product is given by
  //   <q, p_0> + R_p || q ||
  // and since we are using cover trees, p_0 is the point referred to by the
  // node, and R_p will be the expansion constant to the power of the scale plus
  // one.
  const double eval = MetricType::Kernel::Evaluate(querySet.col(queryIndex),
      referenceSet.col(referenceNode.Point()));

  // See if base case can be added.
  if (eval > products(products.n_rows - 1, queryIndex))
  {
    size_t insertPosition;
    for (insertPosition = 0; insertPosition < indices.n_rows; ++insertPosition)
      if (eval > products(insertPosition, queryIndex))
        break;

    // We are guaranteed insertPosition is in the valid range.
    InsertNeighbor(queryIndex, insertPosition, referenceNode.Point(), eval);
  }

  double maxProduct = eval + std::pow(referenceNode.ExpansionConstant(),
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
void FastMKSRules<MetricType>::InsertNeighbor(const size_t queryIndex,
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

}; // namespace fastmks
}; // namespace mlpack

#endif
