/**
 * @file adam.hpp
 * @author Marcus Edel
 *
 * Implementation of the Adam optimizer. Adam is an an algorithm for first-
 * order gradient-based optimization of stochastic objective functions, based on
 * adaptive estimates of lower-order moments.
 */
#ifndef __MLPACK_METHODS_ANN_OPTIMIZER_ADAM_HPP
#define __MLPACK_METHODS_ANN_OPTIMIZER_ADAM_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Adam is an optimizer that computes individual adaptive learning rates for
 * different parameters from estimates of first and second moments of the
 * gradients.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Kingma2014,
 *   author    = {Diederik P. Kingma and Jimmy Ba},
 *   title     = {Adam: {A} Method for Stochastic Optimization},
 *   journal   = {CoRR},
 *   year      = {2014}
 * }
 * @endcode
 */
template< typename DataType>
class Adam
{
 public:
  /**
   * Construct the Adam optimizer with the given function and parameters.
   *
   * @param function Function to be optimized (minimized).
   * @param lr The learning rate coefficient.
   * @param beta1 The first moment coefficient.
   * @param beta2 The second moment coefficient.
   * @param eps The eps coefficient to avoid division by zero (numerical
   *        stability).
   */
  Adam(
          const double lr = 0.001,
          const double beta1 = 0.9,
          const double beta2 = 0.999,
          const double eps = 1e-8) :
      lr(lr),
      beta1(beta1),
      beta2(beta2),
      eps(eps)
  {
    // Nothing to do here.
  }

  /**
   * Optimize the given function using Adam.
   */
  void Optimize(DataType& weights)
  {
    if (mean.n_elem == 0)
    {
	  mean = weights;
      mean.zeros();

      variance = mean;
    }

    Optimize(weights, gradient, mean, variance);
  }

  /*
   * Sum up all gradients and store the results in the gradients storage.
   */
  void Update(DataType const& function_gradients)
  {
    if (gradient.n_elem != 0)
    {
      gradient += function_gradients;
    }
    else
    {
      gradient = function_gradients;
    }
  }

  /*
   * Reset the gradient storage.
   */
  void Reset()
  {
    gradient.zeros();
  }

  //! Get the gradient.
  DataType& Gradient() const { return gradient; }
  //! Modify the gradient.
  DataType& Gradient() { return gradient; }

 private:
  /**
   * Optimize the given function using Adam.
   *
   * @param weights The weights that should be updated.
   * @param gradient The gradient used to update the weights.
   * @param mean The current mean parameter.
   * @param variance The current variance parameter.
   */
  template<typename eT>
  void Optimize(arma::Cube<eT>& weights,
                arma::Cube<eT>& gradient,
                arma::Cube<eT>& mean,
                arma::Cube<eT>& variance)
  {
    for (size_t s = 0; s < weights.n_slices; s++)
    {
      Optimize(weights.slice(s), gradient.slice(s), mean.slice(s),
          variance.slice(s));
    }
  }

  /**
   * Optimize the given function using Adam.
   *
   * @param weights The weights that should be updated.
   * @param gradient The gradient used to update the weights.
   * @param mean The current mean parameter.
   * @param variance The current variance parameter.
   */
  template<typename eT>
  void Optimize(arma::Mat<eT>& weights,
                arma::Mat<eT>& gradient,
                arma::Mat<eT>& mean,
                arma::Mat<eT>& variance)
  {
    // Accumulate updates.
    mean += (1 - beta1) * (gradient - mean);
    variance += (1 - beta2) * (gradient % gradient - variance);

    // Apply update.
    weights -= lr * mean / (arma::sqrt(variance) + eps);
  }


  //! The value used as learning rate.
  const double lr;

  //! The value used as first moment coefficient.
  const double beta1;

  //! The value used as second moment coefficient.
  const double beta2;

  //! The value used as eps.
  const double eps;

  //! The current gradient.
  DataType gradient;

  //! The current mean parameter.
  DataType mean;

  //! The current variance parameter.
  DataType variance;
}; // class Adam

} // namespace ann
} // namespace mlpack

#endif
