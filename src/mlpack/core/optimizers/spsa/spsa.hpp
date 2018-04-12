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
  long long int max_iter;
  arma::vec sp_vector;

  // Control the amount of gradient update.
  float a;
  float A;

  // Non-negative co-efficients controlling the optimizer.
  float alpha;
  float gamma;
  float c;

  // Gain sequences.
  float ak;
  float ck;

  SPSA(const float& alpha = 0.602,
       const float& gamma = 0.101,
       const float& a = 0.16,
       const float& c = 0.3,
       const long long int& maxIterations = 100000):
    max_iter(maxIterations),
    a(a),
    A(0.001*max_iter),
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
    const int s = iterate.n_elem;
    const double epsilon = 1e-8;
    arma::mat gradient = arma::zeros(iterate.n_rows, iterate.n_cols);

    for (long long int i = 0; i < max_iter; i++)
    {
      ak = a/std::pow((i + 1 + A), alpha);
      ck = c/std::pow((i + 1), gamma);

      gradient.zeros();
      for (size_t b = 0; b < 10; b++)
      {
        sp_vector = arma::conv_to<arma::vec>::from(
                    randi(s, arma::distr_param(0, 1)))*2 - 1;

        iterate += ck * sp_vector;
        double f_plus = function.Evaluate(iterate, 0, s);

        iterate -= 2 * ck * sp_vector;
        double f_minus = function.Evaluate(iterate, 0, s);
        iterate += ck * sp_vector;

        gradient += (f_plus - f_minus) * (1 / (2 * ck * sp_vector));
      }

        gradient /= 10;
        iterate -= ak*gradient;
    }

    return function.Evaluate(iterate, 0, s);
  }
};

} // namespace optimization
} // namespace mlpack

#endif
