/**
 * @file spsa_update.hpp
 * @author N Rajiv Vaidyanathan
 *
 * SPSA update for Stochastic Gradient Descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SPSA_SPSA_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_SPSA_SPSA_IMPL_HPP

#include <mlpack/prereqs.hpp>

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
class SPSAUpdate
{
 public:
  SPSAUpdate(const float& alpha,
       const float& gamma,
       const float& a,
       const float& c,
       const size_t& maxIterations):
    alpha(alpha),
    gamma(gamma),
    a(a),
    c(c),
    max_iter(maxIterations)
  {
    Initialize();
  }

  void Initialize()
  {
    A = 0.1*a;
    ak = a/pow((max_iter + 1 + A), alpha);
    ck = c/pow((max_iter + 1), gamma);
  }

  template<typename DecomposableFunctionType>
  void Update(arma::mat& iterate,
          DecomposableFunctionType& function)
  {
    const int s = iterate.n_elem;

    for (size_t i = 0; i < max_iter; i++)
    {
      sp_vector = arma::conv_to<arma::vec>::from(
                  randi(s, arma::distr_param(-1, 1)));
      arma::vec f_plus = function.Evaluate(iterate + ck*sp_vector, s);
      arma::vec f_minus = function.Evaluate(iterate - ck*sp_vector, s);
      arma::mat gradient = ((f_plus - f_minus)/2)*ck*sp_vector.i();

      iterate -= ak*gradient;
    }
  }

 private:
  float a;
  float A;
  float alpha;
  float gamma;
  float c;
  float ak;
  float ck;
  size_t max_iter;
  arma::vec sp_vector;
};

} // namespace optimization
} // namespace mlpack

#endif
