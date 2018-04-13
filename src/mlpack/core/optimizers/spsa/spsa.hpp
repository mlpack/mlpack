/**
 * @file spsa_update.hpp
 * @author N Rajiv Vaidyanathan
 *
 * SPSA update for faster convergence.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SPSA_SPSA_HPP
#define MLPACK_CORE_OPTIMIZERS_SPSA_SPSA_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>

namespace mlpack {
namespace optimization {

/**
 * Implementation of the SPSA update policy. SPSA update policy improves
 * the rate of optimization by performing simultaneously on all parameters.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Spall1998,
 *   author  = {Spall, J. C.},
 *   title   = {An Overview of the Simultaneous Perturbation Method for
 *              Efficient Optimization},
 *   journal = {Johns Hopkins APL Technical Digest},
 *   volume  = {19},
 *   number  = {4},
 *   pages   = {482--492},
 *   year    = {1998}
 * }
 * @endcode
 *
 */
class SPSA
{
 public:
  // Specifies the maximum number of iterations.
  size_t maxIter;
  arma::vec spVector;

  // Control the amount of gradient update.
  double a;
  double A;

  // Non-negative co-efficients controlling the optimizer.
  double alpha;
  double gamma;
  double c;

  // Gain sequences.
  double ak;
  double ck;

  SPSA(const double& alpha = 0.602,
       const double& gamma = 0.101,
       const double& a = 0.16,
       const double& c = 0.3,
       const size_t& maxIterations = 100000):
    maxIter(maxIterations),
    a(a),
    A(0.001*maxIter),
    alpha(alpha),
    gamma(gamma),
    c(c),
    ak(0),
    ck(0)
  {
    // Nothing to do.
  }

  template<typename DecomposableFunctionType>
  double Optimize(DecomposableFunctionType& function, arma::mat& iterate)
  {
    arma::mat gradient = arma::zeros(iterate.n_rows, iterate.n_cols);

    for (size_t i = 0; i < maxIter; i++)
    {
      ak = a/std::pow((i + 1 + A), alpha);
      ck = c/std::pow((i + 1), gamma);

      gradient.zeros();
      for (size_t b = 0; b < 10; b++)
      {
        spVector = arma::conv_to<arma::vec>::from(
                    randi(iterate.n_elem, arma::distr_param(0, 1))) * 2 - 1;

        iterate += ck * spVector;
        double f_plus = function.Evaluate(iterate, 0, iterate.n_elem);

        iterate -= 2 * ck * spVector;
        double f_minus = function.Evaluate(iterate, 0, iterate.n_elem);
        iterate += ck * spVector;

        gradient += (f_plus - f_minus) * (1 / (2 * ck * spVector));
      }

        gradient /= 10;
        iterate -= ak*gradient;
    }

    return function.Evaluate(iterate, 0, iterate.n_elem);
  }

  const double& Alpha() const { return alpha; }

  const double& Gamma() const { return gamma; }

  const double& C() const { return c; }

};

} // namespace optimization
} // namespace mlpack

#endif
