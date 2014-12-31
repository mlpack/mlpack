/**
 * @file random_init.hpp
 * @author Marcus Edel
 *
 * Intialization rule for the neural networks. This simple initialization is
 * performed by assigning a random matrix to the weight matrix.
 */
#ifndef __MLPACK_METHOS_ANN_INIT_RULES_RANDOM_INIT_HPP
#define __MLPACK_METHOS_ANN_INIT_RULES_RANDOM_INIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize randomly the weight matrix.
 *
 * @tparam MatType Type of matrix (should be arma::mat or arma::spmat).
 */
template<typename MatType = arma::mat>
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
  RandomInitialization(const double lowerBound = -0.05,
                       const double upperBound = 0.05) :
      lowerBound(lowerBound), upperBound(upperBound) { }

  /**
   * Initialize the random initialization rule with the given bound.
   * Using the negative of the bound as lower bound and the postive bound as
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
   * @param n_rows Number of rows.
   * @param n_cols Number of columns.
   */
  void Initialize(MatType& W, const size_t n_rows, const size_t n_cols)
  {
    W = lowerBound + arma::randu<MatType>(n_rows, n_cols) *
        (upperBound - lowerBound);
  }

 private:
  //! The number used as lower bound.
  const double lowerBound;

  //! The number used as upper bound.
  const double upperBound;
}; // class RandomInitialization

}; // namespace ann
}; // namespace mlpack

#endif
