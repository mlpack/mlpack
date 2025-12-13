/**
 * @file methods/ann/activation_functions/gelu_exact_function.hpp
 * @author Kumar Utkarsh
 *
 * Definition and implementation of the exact Gaussian Error Linear Unit (GELU)
 * function.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GELU_EXACT_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GELU_EXACT_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The exact GELU function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = x * Phi(x) \\
 * Phi(x) = 0.5 * (1 + erf(x / sqrt(2))) \\
 * f'(x) = Phi(x) + x * phi(x) \\
 * phi(x) = (1 / sqrt(2\pi)) * exp(-x^2 / 2)
 * @f}
 */
class GELUExactFunction
{
 public:
  //! Compute the exact GELU function for a single value.
  static double Fn(const double x)
  {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
  }

  //! Compute the exact GELU function for matrices/vectors.
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y = 0.5 * x % (1.0 + erf(x / std::sqrt(2.0)));
  }

  // Compute the first derivative of the exact GELU function for a single value
  static double Deriv(const double x, const double y )
  {
    const double phi = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
    // Reuse y to avoid costly Phi(x) computation.
    return (x == 0.0) ? 0.5 : (y / x + x * phi);
  }

  //! Compute the first derivative for matrices/vectors.
  template<typename InputVecType, typename OutputVecType, typename DerivVecType>
  static void Deriv(const InputVecType& x,
                    const OutputVecType& y,
                    DerivVecType& dy)
  {
    dy.set_size(x.n_elem);
    // Reuse y to avoid costly Phi(x) computation.
    for (size_t i = 0; i < x.n_elem; ++i)
    {
      if (x[i] == 0.0) dy[i] = 0.5;
      else dy[i] = y[i] / x[i] +
          x[i] * std::exp(-0.5 * x[i] * x[i]) / std::sqrt(2.0 * M_PI);
    }
  }
}; // class GELUExactFunction

} // namespace mlpack

#endif
