/**
 * @file random_selection.hpp
 * @author Ryan Curtin
 *
 * Randomly select some points (with replacement) to use for the Nystroem
 * method.  Replacement is suboptimal, but for rank << number of points, this is
 * unlikely.
 */
#ifndef __MLPACK_METHODS_NYSTROEM_METHOD_RANDOM_SELECTION_HPP
#define __MLPACK_METHODS_NYSTROEM_METHOD_RANDOM_SELECTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

class RandomSelection
{
 public:
  /**
   * Randomly select the specified number of points in the dataset.
   *
   * @param data Dataset to sample from.
   * @param m Number of points to select.
   * @return Indices of selected points from the dataset.
   */
  const static arma::Col<size_t> Select(const arma::mat& data, const size_t m)
  {
    arma::Col<size_t> selectedPoints(m);
    for (size_t i = 0; i < m; ++i)
      selectedPoints(i) = math::RandInt(0, data.n_cols);

    return selectedPoints;
  }
};

} // namespace kernel
} // namespace mlpack

#endif
