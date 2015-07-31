/**
 * @file ada_delta.hpp
 * @author Marcus Edel
 *
 * Implmentation of the RmsProp optimizer. Adadelta is an optimizer that uses
 * the magnitude of recent gradients and steps to obtain an adaptive step rate.
 */
#ifndef __MLPACK_METHODS_ANN_OPTIMIZER_ADA_DELTA_HPP
#define __MLPACK_METHODS_ANN_OPTIMIZER_ADA_DELTA_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Adadelta is an optimizer that uses the magnitude of recent gradients and
 * steps to obtain an adaptive step rate. In its basic form, given a step rate
 * \f$ \gamma \f$ and a decay term \f$ \alpha \f$ we perform the following
 * updates:
 *
 * \f[
 *  g_t &=& (1 - \gamma)f'(\Delta_t)^2 + \gammag_{t - 1} \\
 *  \vec{\Delta} \Delta_t = \alpha \frac{\sqrt(s_{t-1} +
 *  \epsilon)}{\sqrt{g_t + \epsilon}} f'(\Delta_t) \\
 *  \Delta_{t + 1} &=& \Delta_t - \vec{\Delta} \Delta_t \\
 *  s_t &=& (1 - \gamma) \vec{\Delta} \Delta_t^2 + \gammas_{t - 1}
 * \f]
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
   * @param rho Constant similar to that used in AdaDelta and Momentum methods.
   * @param eps The eps coefficient to avoid division by zero.
   */
  AdaDelta(DecomposableFunctionType& function,
          const double rho = 0.95,
          const double eps = 1e-6) :
      function(function),
      rho(rho),
      eps(eps),
      iteration(0)
  {
    // Nothing to do here.
  }

  /**
   * Optimize the given function using RmsProp.
   */
  void Optimize()
  {
    if (meanSquaredGradient.n_elem == 0)
    {
      meanSquaredGradient = function.Weights();
      meanSquaredGradient.zeros();

      meanSquaredGradientDx = meanSquaredGradient;
    }

    if (iteration > 1)
      gradient /= iteration;

    Optimize(function.Weights(), gradient, meanSquaredGradient,
        meanSquaredGradientDx);
  }

  /*
   * Sum up all gradients and store the results in the gradients storage.
   */
  void Update()
  {
    iteration++;

    if (gradient.n_elem != 0)
    {
      DataType outputGradient;
      function.Gradient(outputGradient);
      gradient += outputGradient;
    }
    else
    {
      function.Gradient(gradient);
    }
  }

  /*
   * Reset the gradient storage.
   */
  void Reset()
  {
    iteration = 0;
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
   * @param meanSquaredGradient The current mean squared gradient Dx
   * @param meanSquaredGradientDx The current mean squared gradient.
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
   * @param meanSquaredGradient The current mean squared gradient Dx
   * @param meanSquaredGradientDx The current mean squared gradient.
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

  //! The value used as learning rate.
  const double rho;

  //! The value used as eps.
  const double eps;

  //! The current gradient.
  DataType gradient;

  //! The current mean squared gradient.
  DataType meanSquaredGradient;

  //! The current mean squared gradient Dx
  DataType meanSquaredGradientDx;

  //! The locally stored number of iterations.
  size_t iteration;
}; // class AdaDelta

}; // namespace ann
}; // namespace mlpack

#endif
