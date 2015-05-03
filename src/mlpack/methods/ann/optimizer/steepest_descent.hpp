/**
 * @file steepest_descent.hpp
 * @author Marcus Edel
 *
 * Implmentation of the steepest descent optimizer. The method of steepest
 * descent, also called the gradient descent method, is used to find the
 * nearest local minimum of a function which the assumtion that the gradient of
 * the function can be computed.
 */
#ifndef __MLPACK_METHOS_ANN_OPTIMIZER_STEEPEST_DESCENT_HPP
#define __MLPACK_METHOS_ANN_OPTIMIZER_STEEPEST_DESCENT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to update the weights using steepest descent.
 *
 * @tparam DataType Type of input data (should be arma::mat,
 * arma::spmat or arma::cube).
 */
template<typename DataType = arma::mat>
class SteepestDescent
{
 public:
  /*
   * Construct the optimizer object, which will be used to update the weights.
   *
   * @param lr The value used as learning rate (Default: 1).
   */
  SteepestDescent(const double lr = 1) : lr(lr), mom(0)
  {
    // Nothing to do here.
  }

  /**
   * Construct the optimizer object, which will be used to update the weights.
   *
   * @param cols The number of cols to initilize the momentum matrix.
   * @param rows The number of rows to initilize the momentum matrix.
   * @param lr The value used as learning rate (Default: 1).
   * @param mom The value used as momentum (Default: 0.1).
   */
  SteepestDescent(const size_t cols,
                  const size_t rows,
                  const double lr = 1,
                  const double mom = 0.1) :
      lr(lr), mom(mom)
  {
    if (mom > 0)
      momWeights = arma::zeros<DataType>(rows, cols);
  }

  /**
   * Construct the optimizer object, which will be used to update the weights.
   *
   * @param cols The number of cols used to initilize the momentum matrix.
   * @param rows The number of rows used to initilize the momentum matrix.
   * @param slices The number of slices used to initilize the momentum matrix.
   * @param lr The value used as learning rate (Default: 1).
   * @param mom The value used as momentum (Default: 0.1).
   */
  SteepestDescent(const size_t cols,
                  const size_t rows,
                  const size_t slices,
                  const double lr,
                  const double mom) :
      lr(lr), mom(mom)
  {
    if (mom > 0)
      momWeights = arma::zeros<DataType>(rows, cols, slices);
  }

  /*
   * Update the specified weights using steepest descent.
   *
   * @param weights The weights that should be updated.
   * @param gradient The gradient used to update the weights.
   */
  template<typename WeightType, typename GradientType>
  void UpdateWeights(WeightType& weights,
                     const GradientType& gradient,
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
  //! The value used as learning rate.
  const double lr;

  //! The value used as momentum.
  const double mom;

  //! Momentum matrix.
  DataType momWeights;
}; // class SteepestDescent

}; // namespace ann
}; // namespace mlpack

#endif
