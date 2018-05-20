/**
 * @file constraints.hpp
 * @author Manish Kumar
 *
 * Declaration of the Constraints class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMNN_CONSTRAINTS_HPP
#define MLPACK_METHODS_LMNN_CONSTRAINTS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace lmnn {

class Constraints
{
 public:
  /**
   * Initialize Constraints with dataset and labels.
   *
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param k Number of target neigbors.
   */
  Constraints(const arma::mat& dataset,
              const arma::Row<size_t>& labels,
              size_t k);

  /**
   * Calculates k similar labeled nearest neighbors and stores them into the
   * passed matrix.
   */
  void TargetNeighbors(arma::Mat<size_t>& outputMatrix);

  /**
   * Calculates k differently labeled nearest neighbors for each datapoint and
   * writes them back to passed matrix
   */
  void Impostors(arma::Mat<size_t>& outputMatrix);

  /**
   * Generate triplets {i, j, l} for each datapoint i and writes back generated
   * triplets to matrix passed.
   */
  void Triplets(arma::Mat<size_t>& outputMatrix);

  //! Access the value of k.
  const size_t& K() const { return k; }
  //! Modify the value of k.
  size_t& K() { return k; }

 private:
  //! An alias of dataset.
  arma::mat dataset;

  //! An alias of Labels.
  arma::Row<size_t> labels;

  //! Number of target neighbors & impostors to calulate.
  size_t k;
};

} // namespace lmnn
} // namespace mlpack

// Include implementation.
#include "constraints_impl.hpp"

#endif
