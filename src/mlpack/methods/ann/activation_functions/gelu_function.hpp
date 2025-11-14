/**
 * @file methods/ann/activation_functions/gelu_function.hpp
 * @author Himanshu Pathak
 * @author Adam Kropp
 *
 * Definition and implementation of the Gaussian Error Linear Unit (GELU)
 * function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GELU_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GELU_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The GELU function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = 0.5 * x * {1 + tanh[(2/pi)^(1/2) * (x + 0.044715 * x^3)]} \\
 * f'(x) = 0.5 * tanh(0.0356774 * x^3 + 0.797885 * x) +
 *         (0.0535161x^3 + 0.398942 * x) * 
 *         sech^2(0.0356774 * x^3+0.797885 * x) + 0.5\\
 * @f}
 */
class GELUFunction
{
 public:
  /**
   * Computes the GELU function.
   *
   * @param x Input data.
   * @return f(x).
   */
  template<typename ElemType>
  static ElemType Fn(const ElemType x)
  {
    return (x / 2) *
        (1 + std::tanh(std::sqrt(2 / arma::Datum<ElemType>::pi) *
        (x + ElemType(0.044715) * std::pow(x, ElemType(3)))));
  }

  /**
   * Computes the GELU function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    typedef typename InputVecType::elem_type ElemType;

    y = (x / 2) %
        (1 + tanh(std::sqrt(2 / arma::Datum<ElemType>::pi) *
        (x + ElemType(0.044715) * pow(x, ElemType(3)))));
  }

  /**
   * Computes the first derivative of the GELU function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @return f'(x)
   */
  template<typename ElemType>
  static ElemType Deriv(const ElemType x, const ElemType /* y */)
  {
    if (x < -10) return 0; // catch overflows
    return ElemType(0.5) * std::tanh(ElemType(0.0356774) *
        std::pow(x, ElemType(3)) + ElemType(0.797885) * x) +
        (ElemType(0.0535161) * std::pow(x, ElemType(3)) +
         ElemType(0.398942) * x) *
         std::pow(1 / std::cosh(ElemType(0.0356774) * std::pow(x, 3) +
         ElemType(0.797885) * x), 2) + ElemType(0.5);
  }

  /**
   * Computes the first derivatives of the GELU function.
   *
   * @param x Input activation.
   * @param y Result of Fn(x).
   * @param dy The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType, typename DerivVecType>
  static void Deriv(const InputVecType& x,
                    const OutputVecType& /* y */,
                    DerivVecType& dy)
  {
    typedef typename InputVecType::elem_type ElemType;

    dy = ElemType(0.5) * tanh(ElemType(0.0356774) * pow(x, ElemType(3)) +
        ElemType(0.797885) * x) + (ElemType(0.0535161) * pow(x, ElemType(3)) +
        ElemType(0.398942) * x) %
        pow(1 / cosh(ElemType(0.0356774) * pow(x, ElemType(3)) +
        ElemType(0.797885) * x), 2) + ElemType(0.5);
    dy(find(x < -10)).fill(0); // catch overflows
  }
}; // class GELUFunction

} // namespace mlpack

#endif
