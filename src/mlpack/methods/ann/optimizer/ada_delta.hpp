/**
 * @file ada_delta.hpp
 * @author Marcus Edel
 *
 * Implementation of the Adadelta optimizer. Adadelta is an optimizer that
 * dynamically adapts over time using only first order information.
 * Additionally, Adadelta requires no manual tuning of a learning rate.
 */
#ifndef MLPACK_METHODS_ANN_OPTIMIZER_ADA_DELTA_HPP
#define MLPACK_METHODS_ANN_OPTIMIZER_ADA_DELTA_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Adadelta is an optimizer that uses two ideas to improve upon the two main
 * drawbacks of the Adagrad method:
 *
 *  - Accumulate Over Window
 *  - Correct Units with Hessian Approximation
 *
 * For more information, see the following.
 *
 * @code
 * @article{Zeiler2012,
 *   author    = {Matthew D. Zeiler},
 *   title     = {{ADADELTA:} An Adaptive Learning Rate Method},
 *   journal   = {CoRR},
 *   year      = {2012}
 * }
 * @endcode
 */
template<typename DecomposableFunctionType, typename DataType>
class AdaDelta
{
 public:
  /**
   * Construct the AdaDelta optimizer with the given function and parameters.
   *
   * @param function Function to be optimized (minimized).
   * @param rho Constant interpolation parameter similar to that used in
   *        Momentum methods.
   * @param eps The eps coefficient to avoid division by zero (numerical
   *        stability).
   */
  AdaDelta(DecomposableFunctionType& function,
          const double rho = 0.95,
          const double eps = 1e-6) :
      function(function),
      rho(rho),
      eps(eps)
  {
    // Nothing to do here.
  }

  /**
   * Optimize the given function using AdaDelta.
   */
  void Optimize()
  {
    if (meanSquaredGradient.n_elem == 0)
    {
      meanSquaredGradient = function.Weights();
      meanSquaredGradient.zeros();

      meanSquaredGradientDx = meanSquaredGradient;
    }

    Optimize(function.Weights(), gradient, meanSquaredGradient,
        meanSquaredGradientDx);
  }

  /*
   * Sum up all gradients and store the results in the gradients storage.
   */
  void Update()
  {
    if (gradient.n_elem != 0)
    {
      gradient += function.Gradient();
    }
    else
    {
      gradient = function.Gradient();
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
   * Optimize the given function using AdaDelta.
   *
   * @param weights The weights that should be updated.
   * @param gradient The gradient used to update the weights.
   * @param meanSquaredGradient The current mean squared gradient.
   * @param meanSquaredGradientDx The current mean squared Dx gradient.
   */
  template<typename eT>
  void Optimize(arma::Cube<eT>& weights,
                arma::Cube<eT>& gradient,
                arma::Cube<eT>& meanSquaredGradient,
                arma::Cube<eT>& meanSquaredGradientDx)
  {
    for (size_t s = 0; s < weights.n_slices; s++)
    {
      Optimize(weights.slice(s), gradient.slice(s), meanSquaredGradient.slice(s),
          meanSquaredGradientDx.slice(s));
    }
  }

  /**
   * Optimize the given function using AdaDelta.
   *
   * @param weights The weights that should be updated.
   * @param gradient The gradient used to update the weights.
   * @param meanSquaredGradient The current mean squared gradient.
   * @param meanSquaredGradientDx The current mean squared Dx gradient.
   */
  template<typename eT>
  void Optimize(arma::Mat<eT>& weights,
                arma::Mat<eT>& gradient,
                arma::Mat<eT>& meanSquaredGradient,
                arma::Mat<eT>& meanSquaredGradientDx)
  {
    // Accumulate gradient.
    meanSquaredGradient *= rho;
    meanSquaredGradient += (1 - rho) * (gradient % gradient);
    arma::Mat<eT> dx = arma::sqrt((meanSquaredGradientDx + eps) /
        (meanSquaredGradient + eps)) % gradient;

    // Accumulate updates.
    meanSquaredGradientDx *= rho;
    meanSquaredGradientDx += (1 - rho) * (dx % dx);

    // Apply update.
    weights -= dx;
  }

  //! The instantiated function.
  DecomposableFunctionType& function;

  //! The value used as interpolation parameter.
  const double rho;

  //! The value used as eps.
  const double eps;

  //! The current gradient.
  DataType gradient;

  //! The current mean squared gradient.
  DataType meanSquaredGradient;

  //! The current mean squared gradient.
  DataType meanSquaredGradientDx;
}; // class AdaDelta

} // namespace ann
} // namespace mlpack

#endif
