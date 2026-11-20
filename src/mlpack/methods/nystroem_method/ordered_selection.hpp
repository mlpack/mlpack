/**
 * @file methods/nystroem_method/ordered_selection.hpp
 * @author Ryan Curtin
 *
 * Select the first points of the dataset for use in the Nystroem method of
 * kernel matrix approximation. This is mostly for testing, but might have
 * other uses.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NYSTROEM_METHOD_ORDERED_SELECTION_HPP
#define MLPACK_METHODS_NYSTROEM_METHOD_ORDERED_SELECTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

class OrderedSelection
{
 public:
  /**
   * Select the specified number of points in the dataset.
   *
   * @param * (data) Dataset to sample from.
   * @param m Number of points to select.
   * @return Indices of selected points from the dataset.
   */
  static const arma::Col<size_t> Select(const arma::mat& /* data */,
                                        const size_t m)
  {
    // This generates [0 1 2 3 ... (m - 1)].
    return arma::linspace<arma::Col<size_t> >(0, m - 1, m);
  }
};

} // namespace mlpack

#endif
