/**
 * @file methods/ann/activation_functions/shifted_softplus.hpp
 * @author Mayank Raj
 *
 * Definition and implementation of the Shifted-Softplus functionKristof T. Schütt,
 * Pieter-Jan Kindermans, Huziel E. Sauceda, Stefan Chmiela, Alexandre Tkatchenko, Klaus-Robert Müller.
 * For more information, see the following paper.
 *
 * @code
 * @misc{
 *   author = {Kristof T. Schütt, Pieter-Jan Kindermans, Huziel E. Sauceda,
 *   Stefan Chmiela, Alexandre Tkatchenko, Klaus-Robert Müller},
 *   title = {SchNet: A continuous-filter convolutional neural network for
 *   modeling quantum interactions},
 *   year = {2017}
 * }
 * @endcode
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SHIFTED_SOFTPLUS_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SHIFTED_SOFTPLUS_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The Shifted Softplus function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \ln(0.5 \cdot e^x + 0.5)
 * f'(x) &=& \frac{0.5 \cdot e^x}{0.5 \cdot e^x + 0.5}
 * @f}
 */
class ShiftedSoftplusFunction
{
 public:
  /**
   * Computes the Shifted-Softplus function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    return std::log(0.5 * std::exp(x) + 0.5);
  }

  /**
   * Computes the Shifted-Softplus function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y.set_size(x.size());
    for(size_t i = 0; i < x.n_elem; ++i)
    {
      y(i) = std::log(0.5 * std::exp(x(i)) + 0.5);
    }
  }

  /**
   * Computes the first derivative of the Shifted-Softplus function.
   *
   * @param y Input activation.
   * @return f'(x)
   */
  static double Deriv(const double y)
  {
    return 1 - 0.5 / std::exp(y);
   }

  /**
   * Computes the first derivatives of the Shifted-Softplus function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType& y, OutputVecType& x)
  {
    x.set_size(y.size());
    for(size_t i = 0; i < y.n_elem; ++i)
    {
      x(i) = Deriv(y(i));
     }
  }
  }; // class ShiftedSoftplusFunction
  } // namespace mlpack

  #endif