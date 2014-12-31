/**
 * @file rectifier_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the rectifier function as described by
 * V. Nair and G. E. Hinton.
 *
 * For more information, see the following paper.
 *
 * @code
 * @misc{NairHinton2010,
 *   author = {Vinod Nair, Geoffrey E. Hinton},
 *   title = {Rectified Linear Units Improve Restricted Boltzmann Machines},
 *   year = {2010}
 * }
 * @endcode
 */
#ifndef __MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_RECTIFIER_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_RECTIFIER_FUNCTION_HPP

#include <mlpack/core.hpp>
#include <algorithm>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The rectifier function, defined by
 *
 * @f[
 * f(x) &=& \max(0, x) \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     1 & : x > 0 \\
 *     0 & : x \le 0
 *   \end{array}
 * \right
 * @f]
 */
class RectifierFunction
{
 public:
  /**
   * Computes the rectifier function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double fn(const double x)
  {
    return std::max(0.0, x);
  }

  /**
   * Computes the rectifier function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void fn(const InputVecType& x, OutputVecType& y)
  {
    y = x;
    y = arma::max(arma::zeros<OutputVecType>(x.n_elem), x);
  }

  /**
   * Computes the first derivative of the rectifier function.
   *
   * @param x Input data.
   * @return f'(x)
   */
  static double deriv(const double y)
  {
    return y > 0 ? 1 : 0;
  }

  /**
   * Computes the first derivatives of the rectifier function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void deriv(const InputVecType& y, OutputVecType& x)
  {
    x = y;
    x.transform( [](double y) { return deriv(y); } );
  }
}; // class RectifierFunction

}; // namespace ann
}; // namespace mlpack

#endif
