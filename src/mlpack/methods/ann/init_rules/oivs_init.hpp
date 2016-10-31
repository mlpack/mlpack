/**
 * @file oivs_init.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the Optimal Initial Value Setting method
 * (OIVS). This initialization rule is based on geometrical considerations as
 * described by H. Shimodaira.
 *
 * For more information, see the following paper.
 *
 * @code
 * @inproceedings{ShimodairaICTAI1994,
 *   title={A weight value initialization method for improving learning
 *   performance of the backpropagation algorithm in neural networks},
 *   author={Shimodaira, H.},
 *   booktitle={Tools with Artificial Intelligence, 1994. Proceedings.,
 *   Sixth International Conference on},
 *   year={1994}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_OIVS_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_OIVS_INIT_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

#include "random_init.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize the weight matrix with the oivs method. The
 * method is based on the equations representing the characteristics of the
 * information transformation mechanism of a node. The method is defined by
 *
 * @f{eqnarray*}{
 * b &=& |F^{-1}(1 - \epsilon) - f^{-1}(\epsilon)| \\
 * \hat{w} &=& \frac{b}{k \cdot n} \\
 * \gamma &\le& a_i \le \gamma \\
 * w_i &=& \hat{w} \cdot \sqrt{a_i + 1}
 * @f}
 *
 * Where f is the transfer function epsilon, k custom parameters, n the number of
 * neurons in the outgoing layer and gamma a parameter that defines the random
 * interval.
 *
 * @tparam ActivationFunction The activation function used for the oivs method.
 */
template<
    class ActivationFunction = LogisticFunction
>
class OivsInitialization
{
 public:
  /**
   * Initialize the random initialization rule with the given values.
   *
   * @param epsilon Parameter to control the activation region.
   * @param k Parameter to control the activation region width.
   * @param gamma Parameter to define the uniform random range.
   */
  OivsInitialization(const double epsilon = 0.1,
                     const int k = 5,
                     const double gamma = 0.9) :
      k(k), gamma(gamma),
      b(std::abs(ActivationFunction::inv(1 - epsilon) -
                 ActivationFunction::inv(epsilon)))
  {
  }

  /**
   * Initialize the elements of the specified weight matrix with the oivs method.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename eT>
  void Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols)
  {
    RandomInitialization randomInit(-gamma, gamma);
    randomInit.Initialize(W, rows, cols);

    W = (b / (k  * rows)) * arma::sqrt(W + 1);
  }

  /**
   * Initialize the elements of the specified weight 3rd order tensor with the
   * oivs method.
   *
   * @param W 3rd order tensor to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param slices Number of slices.
   */
  template<typename eT>
  void Initialize(arma::Cube<eT>& W,
                  const size_t rows,
                  const size_t cols,
                  const size_t slices)
  {
    W = arma::Cube<eT>(rows, cols, slices);

    for (size_t i = 0; i < slices; i++)
      Initialize(W.slice(i), rows, cols);
  }

 private:
  //! Parameter to control the activation region width.
  const int k;

  //! Parameter to define the uniform random range.
  const double gamma;

  //! Parameter to control the activation region.
  const double b;
}; // class OivsInitialization


} // namespace ann
} // namespace mlpack

#endif
