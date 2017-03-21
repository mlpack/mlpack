/**
 * @file random_init.hpp
 * @author Marcus Edel
 *
 * Intialization rule for the neural networks. This simple initialization is
 * performed by assigning a random matrix to the weight matrix.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_RANDOM_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_RANDOM_INIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize randomly the weight matrix.
 */
class RandomInitialization
{
 public:
  /**
   * Initialize the random initialization rule with the given lower bound and
   * upper bound.
   *
   * @param lowerBound The number used as lower bound.
   * @param upperBound The number used as upper bound.
   */
  RandomInitialization(const double lowerBound = -1,
                       const double upperBound = 1) :
      lowerBound(lowerBound), upperBound(upperBound) { }

  /**
   * Initialize the random initialization rule with the given bound.
   * Using the negative of the bound as lower bound and the positive bound as
   * upper bound.
   *
   * @param bound The number used as lower bound
   */
  RandomInitialization(const double bound) :
      lowerBound(-std::abs(bound)), upperBound(std::abs(bound)) { }

  /**
   * Initialize randomly the elements of the specified weight matrix.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename eT>
  void Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols)
  {
    W = lowerBound + arma::randu<arma::Mat<eT>>(rows, cols) *
        (upperBound - lowerBound);
  }

  /**
   * Initialize randomly the elements of the specified weight 3rd order tensor.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename eT>
  void Initialize(arma::Cube<eT>& W,
                  const size_t rows,
                  const size_t cols,
                  const size_t slices)
  {
    W = arma::Cube<eT>(rows, cols, slices);

    for (size_t i = 0; i < slices; i++)
      Initialize(W.slice(i), rows, cols);
  }

 private:
  //! The number used as lower bound.
  const double lowerBound;

  //! The number used as upper bound.
  const double upperBound;
}; // class RandomInitialization

} // namespace ann
} // namespace mlpack

#endif
