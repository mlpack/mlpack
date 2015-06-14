/**
 * @file rmsprop.hpp
 * @author Marcus Edel
 *
 * Implmentation of the RmsProp optimizer. RmsProp is an optimizer that utilizes
 * the magnitude of recent gradients to normalize the gradients.
 */
#ifndef __MLPACK_METHODS_ANN_OPTIMIZER_RMSPROP_HPP
#define __MLPACK_METHODS_ANN_OPTIMIZER_RMSPROP_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * RmsProp is an optimizer that utilizes the magnitude of recent gradients to
 * normalize the gradients.
 *
 * For more information, see the following.
 *
 * @code
 * @misc{[tieleman2012,
 *   title={Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine
 *   Learning},
 *   year={2012}
 * }
 * @endcode
 */
template<typename DecomposableFunctionType, typename DataType>
class RMSPROP
{
 public:
  /**
   * Construct the RMSPROP optimizer with the given function and parameters.
   *
   * @param function Function to be optimized (minimized).
   * @param lr The learning rate coefficient.
   * @param alpha Constant similar to that used in AdaDelta and Momentum methods.
   * @param eps The eps coefficient to avoid division by zero.
   */
  RMSPROP(DecomposableFunctionType& function,
          const double lr = 0.01,
          const double alpha = 0.99,
          const double eps = 1e-8) :
      function(function),
      lr(lr),
      alpha(alpha),
      eps(eps),
      meanSquareGad(function.Weights())
  {
    // Nothing to do here.
  }

  /**
   * Optimize the given function using RmsProp.
   */
  void Optimize()
  {
    if (meanSquareGad.n_elem == 0)
    {
      meanSquareGad = function.Weights();
      meanSquareGad.zeros();
    }

    DataType gradient;
    function.Gradient(gradient);

    Optimize(function.Weights(), gradient, meanSquareGad);
  }

 private:
  /**
   * Optimize the given function using RmsProp.
   *
   * @param weights The weights that should be updated.
   * @param gradient The gradient used to update the weights.
   * @param gradient The moving average over the root mean squared gradient used
   *    to update the weights.
   */
  template<typename eT>
  void Optimize(arma::Cube<eT>& weights,
                arma::Cube<eT>& gradient,
                arma::Cube<eT>& meanSquareGradient)
  {
    for (size_t s = 0; s < weights.n_slices; s++)
      Optimize(weights.slice(s), gradient.slice(s), meanSquareGradient.slice(s));
  }

  /**
   * Optimize the given function using RmsProp.
   *
   * @param weights The weights that should be updated.
   * @param gradient The gradient used to update the weights.
   * @param gradient The moving average over the root mean squared gradient used
   *    to update the weights.
   */
  template<typename eT>
  void Optimize(arma::Mat<eT>& weights,
                arma::Mat<eT>& gradient,
                arma::Mat<eT>& meanSquareGradient)
  {
    meanSquareGradient *= alpha;
    meanSquareGradient += (1 - alpha) * (gradient % gradient);
    weights -= lr * gradient / (arma::sqrt(meanSquareGradient) + eps);
  }

  //! The instantiated function.
  DecomposableFunctionType& function;

  //! The value used as learning rate.
  const double lr;

  //! The value used as alpha
  const double alpha;

  //! The value used as eps.
  const double eps;

  //! The current mean squared error.
  DataType meanSquareGad;
}; // class RMSPROP

}; // namespace ann
}; // namespace mlpack

#endif
