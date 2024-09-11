/**
 * @file methods/ann/activation_functions/hyper_sinh_function.hpp
 * @author Mayank Raj
 *
 * Definition of the hyper-sinh function as described by
 * L. Parisi et al. (2021) (full reference below).
 *
 * Original implementation in Python by Luca Parisi as per https://github.com/luca-parisi/hyper_sinh
 *
 * For more information, see the following paper.
 *
 * @code
 * @article{parisi2021hyper,
 *   title={hyper-sinh: An accurate and reliable function from shallow to deep learning in TensorFlow and Keras},
 *   author={Parisi, Luca and Ma, Renfei and RaviChandran, Narrendar and Lanzillotta, Matteo},
 *   journal={Machine Learning with Applications},
 *   volume={6},
 *   pages={100112},
 *   year={2021},
 *   publisher={Elsevier}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_HYPER_SINH_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_HYPER_SINH_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The Hyper-sinh function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \begin{cases}
 * \sinh(x) / 3, & \text{if } x > 0 \\
 * (x^3) / 4, & \text{if } x \leq 0
 * \end{cases} \\
 * f'(x) &=& \begin{cases}
 * \cosh(x) / 3, & \text{if } x > 0 \\
 * (3/4) x^2, & \text{if } x \leq 0
 * \end{cases} \\
 * @f}
 */
class HyperSinhFunction
{
 public:
  /**
   * Computes the Hyper-sinh function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x)
  {
    if (x > 0)
      return (std::sinh(x) / 3.0);
    else
      return (std::pow(x, 3.0) / 4.0);
  }

  /**
   * Computes the Hyper-sinh function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType& x, OutputVecType& y)
  {
    y.set_size(x.size());
    for (size_t i = 0; i < x.n_elem; ++i)
    {
      if (x(i) > 0)
      {
        y(i) = std::sinh(x(i)) / 3;
      }
      else
      {
        y(i) = std::pow(x(i), 3) / 4;
      }
    }
  }


  /**
   * Computes the first derivative of the Hyper-sinh function.
   *
   * @param y Input activation.
   * @return f'(x)
   */
  static double Deriv(const double /* x */, const double y)
  {
    if (y > 0)
      return (std::pow((1.0 / 9.0) + (y * y), 0.5));
    else
      return (3.0  * std::pow(std::pow(y, 2) / 4, 1.0 / 3.0));
  }

  /**
   * Computes the first derivatives of the Hyper-sinh function.
   *
   * @param y Input activations.
   * @param dy The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType, typename DerivVecType>
  static void Deriv(const InputVecType& /* x */,
                    const OutputVecType& y,
                    DerivVecType& dy)
  {
    dy.set_size(size(y));
    #pragma omp for
    for (size_t i = 0; i < y.n_elem; ++i)
    {
      if (y(i) > 0)
      {
        dy(i) = (std::pow((1.0 / 9.0) + (y(i) * y(i)), 0.5));
      }
      else
      {
        dy(i) = (3.0 * std::pow(std::pow(y(i), 2) / 4, 1.0 / 3.0));
      }
    }
  }
}; // class HyperSinhFunction
} // namespace mlpack

#endif
