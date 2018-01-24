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
  SPSAUpdate(const size_t& n_params = 100000,
         const float& alpha = 0.602,
         const float& gamma = 0.101,
         const float& a = 1e-6,
         const float& c = 0.01,
         const size_t& max_iter = 5000000):
    alpha(alpha),
    gamma(gamma),
    a(a),
    c(c),
    ak(0),
    ck(0),
    A(0.1*a),
    max_iter(max_iter),
    p(n_params)
  {
    // Nothing to do.
  }

  void Initialize()
  {
    ak = a/pow((max_iter + 1 + A), alpha);
    ck = c/pow((max_iter + 1), gamma);
  }

  template<typename DecomposableFunctionType>
  void Update(arma::mat& iterate,
          DecomposableFunctionType& function)
  {
    for (size_t i = 0; i < max_iter; i++)
    {
      sp_vector = randi(p, arma::distr_param(-1, 1));
      arma::vec f_plus = function.Evaluate(iterate + ck*sp_vector);
      arma::vec f_minus = function.Evaluate(iterate - ck*sp_vector);
      float gradient = (f_plus - f_minus)/(2*ck*sp_vector);

      iterate -= ak*gradient;
    }
  }

  float Alpha() const { return alpha; }

  float& Alpha() { return alpha; }

  float Gamma() const { return gamma; }

  float& Gamma() { return gamma; }

  float Gradient_scaling_parameter(const int& choice) const
  {
    if (choice == 0)
    {
      std::cout<<"Parameter -> a"<<std::endl;
      return a;
    }
    else if (choice == 1)
    {
      std::cout<<"Parameter -> A"<<std::endl;
      return A;
    }
    else
    {
      std::cout<<"No such parameter exists..."<<std::endl;
      return -1.0;
    }
  }

  float& Gradient_scaling_parameter(const int& choice)
  {
    if (choice == 0)
    {
      std::cout<<"Parameter -> a"<<std::endl;
      return a;
    }
    else if (choice == 1)
    {
      std::cout<<"Parameter -> A"<<std::endl;
      return A;
    }
    else
    {
      std::cout<<"No such parameter exists..."<<std::endl;
      static float none = -1;
      return none;
    }
  }

  float Noise_variance_parameter() const { return c; }

  float& Noise_variance_parameter() { return c; }

 private:
  float a;
  float A;
  float alpha;
  float gamma;
  float c;
  float ak;
  float ck;
  long long int max_iter, p;
  arma::vec sp_vector;
};

} // namespace optimization
} // namespace mlpack

#endif
