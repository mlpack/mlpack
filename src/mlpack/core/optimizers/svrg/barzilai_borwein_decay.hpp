/**
 * @file barzilai_borwein_decay.hpp
 * @author Marcus Edel
 *
 * Barzilai-Borwein decay policy.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SVRG_BARZILIA_BORWEIN_DECAY_HPP
#define MLPACK_CORE_OPTIMIZERS_SVRG_BARZILIA_BORWEIN_DECAY_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Barzilai-Borwein decay policy for Stochastic variance reduced gradient
 * (SVRG).
 *
 * For more information, please refer to:
 *
 * @code
 * @incollection{Tan2016,
 *   title     = {Barzilai-Borwein Step Size for Stochastic Gradient Descent},
 *   author    = {Tan, Conghui and Ma, Shiqian and Dai, Yu-Hong
 *                and Qian, Yuqiu},
 *   booktitle = {Advances in Neural Information Processing Systems 29},
 *   editor    = {D. D. Lee and M. Sugiyama and U. V. Luxburg and I. Guyon
 *                and R. Garnett},
 *   pages     = {685--693},
 *   year      = {2016},
 *   publisher = {Curran Associates, Inc.}
 * }
 * @endcode
 */
class BarzilaiBorweinDecay
{
 public:
  /*
   * Construct the Barzilai-Borwein decay policy.
   *
   * @param maxStepSize The maximum step size.
   * @param eps The eps coefficient to avoid division by zero (numerical
   *    stability).
   */
  BarzilaiBorweinDecay(const double maxStepSize = DBL_MAX,
                       const double eps = 1e-7) :
      eps(eps),
      maxStepSize(maxStepSize)
  { /* Nothing to do. */}

  /**
   * The Initialize method is called by SVRG Optimizer method before the start
   * of the iteration update process.
   *
   * @param rows Number of rows in the gradient matrix.
   * @param cols Number of columns in the gradient matrix.
   */
  void Initialize(const size_t /* rows */, const size_t /* cols */)
  { /* Do nothing. */ }

  /**
   * Barzilai-Borwein update step for SVRG.
   *
   * @param iterate The current function parameter at time t.
   * @param iterate0 The last function parameters at time t - 1.
   * @param gradient The current gradient matrix at time t.
   * @param fullGradient The computed full gradient.
   * @param numBatches The number of batches.
   * @param stepSize Step size to be used for the given iteration.
   */
  void Update(const arma::mat& iterate,
              const arma::mat& iterate0,
              const arma::mat& /* gradient */,
              const arma::mat& fullGradient,
              const size_t numBatches,
              double& stepSize)
  {
    if (!fullGradient0.is_empty())
    {
      // Step size selection based on Barzilai-Borwein (BB).
      stepSize = std::pow(arma::norm(iterate - iterate0), 2.0) /
          (arma::dot(iterate - iterate0, fullGradient - fullGradient0) + eps) /
          (double) numBatches;

      stepSize = std::min(stepSize, maxStepSize);
    }

    fullGradient0 = std::move(fullGradient);
  }

 private:
  //! Locally-stored full gradient.
  arma::mat fullGradient0;

  //! The value used for numerical stability.
  double eps;

  //! The maximum step size.
  double maxStepSize;
};

} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_SVRG_BARZILIA_BORWEIN_DECAY_HPP
