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

  // Specifies the maximum number of iterations.
  size_t max_iter;
  arma::vec sp_vector;
  SPSA(const float& alpha = 0.602,
       const float& gamma = 0.101,
       const float& a = 1e-6,
       const float& c = 0.01,
       const size_t& maxIterations = 100000):
    alpha(alpha),
    gamma(gamma),
    a(a),
    c(c),
    A(0.1*a),
    max_iter(maxIterations)
  {
    // Nothing to do.
  }

  template<typename DecomposableFunctionType>
  void Optimize(DecomposableFunctionType& function, arma::mat& iterate)
  {
    const int s = iterate.n_elem;

    for (size_t i = 0; i < max_iter; i++)
    {
      sp_vector = arma::conv_to<arma::vec>::from(
                  randi(s, arma::distr_param(-1, 1)));
      ak = a/std::pow((max_iter + 1 + A), alpha);
      ck = c/std::pow((max_iter + 1), gamma);
      arma::vec f_plus = function.Evaluate(iterate + ck*sp_vector, s);
      arma::vec f_minus = function.Evaluate(iterate - ck*sp_vector, s);
      arma::mat gradient = (f_plus - f_minus) % (1 / (2 * ck * sp_vector));
      iterate -= ak*gradient;
    }
  }
};

} // namespace optimization
} // namespace mlpack

#endif
