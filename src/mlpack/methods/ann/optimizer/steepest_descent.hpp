/**
 * @file steepest_descent.hpp
 * @author Marcus Edel
 *
 * Intialization rule for the neural networks. This simple initialization is
 * performed by assigning a random matrix to the weight matrix. 
 */
#ifndef __MLPACK_METHOS_ANN_OPTIMIZER_STEEPEST_DESCENT_HPP
#define __MLPACK_METHOS_ANN_OPTIMIZER_STEEPEST_DESCENT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize randomly the weight matrix.
 *
 * @tparam MatType Type of matrix (should be arma::mat or arma::spmat).
 */
template<typename MatType = arma::mat, typename VecType = arma::colvec>
class SteepestDescent
{
 public:
  /**
   * Initialize the random initialization rule with the given lower bound and
   * upper bound.
   *
   * @param lowerBound The number used as lower bound.
   * @param upperBound The number used as upper bound.
   */
  SteepestDescent(const size_t cols,
                  const size_t rows,
                  const double lr = 1, 
                  const double mom = 0.1) : 
      lr(lr), mom(mom)
  {
    if (mom > 0)
      momWeights = arma::zeros<MatType>(rows, cols);
  }

  void UpdateWeights(MatType& weights,
                     const MatType& gradient,
                     const double /* unused */)
  {
    if (mom > 0)
    {
      momWeights *= mom;
      momWeights += lr * gradient;
      weights -= momWeights;
    }
    else
      weights -= lr * gradient;
  }


 private:
  //! The number used as learning rate.
  const double lr;

  //! The number used as momentum.
  const double mom;

  //! weight momentum
  MatType momWeights;
}; // class SteepestDescent

}; // namespace ann
}; // namespace mlpack

#endif


