/**
 * @file ordered_selection.hpp
 * @author Ryan Curtin
 *
 * Select the first points of the dataset for use in the Nystroem method of
 * kernel matrix approximation. This is mostly for testing, but might have
 * other uses.
 */
#ifndef MLPACK_METHODS_NYSTROEM_METHOD_ORDERED_SELECTION_HPP
#define MLPACK_METHODS_NYSTROEM_METHOD_ORDERED_SELECTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

class OrderedSelection
{
 public:
  /**
   * Select the specified number of points in the dataset.
   *
   * @param data Dataset to sample from.
   * @param m Number of points to select.
   * @return Indices of selected points from the dataset.
   */
  const static arma::Col<size_t> Select(const arma::mat& /* unused */,
                                        const size_t m)
  {
    // This generates [0 1 2 3 ... (m - 1)].
    return arma::linspace<arma::Col<size_t> >(0, m - 1, m);
  }
};

} // namespace kernel
} // namespace mlpack

#endif
