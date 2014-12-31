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
 */
#ifndef __MLPACK_METHOS_ANN_INIT_RULES_OIVS_INIT_HPP
#define __MLPACK_METHOS_ANN_INIT_RULES_OIVS_INIT_HPP

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
 * @f[
 * b = |f^{-1}(1 - \epsilon) - f^{-1}(\epsilon)| \\
 * \^w = \frac{b}{k \cdot n} \\
 * \gamma \le a_i \le \gamma \\
 * w_i = \^w \cdot \sqrt{a_i + 1}
 * @f]
 *
 * Where f is the transfer function epsilon, k custom parameters, n the number of
 * neurons in the outgoing layer and gamma a parameter that defines the random
 * interval.
 *
 * @tparam ActivationFunction The activation function used for the oivs method.
 * @tparam MatType Type of matrix (should be arma::mat or arma::spmat).
 */
template<
    class ActivationFunction = LogisticFunction,
    typename MatType = arma::mat
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
      epsilon(epsilon), k(k), gamma(gamma) { }

  /**
   * Initialize the elements of the specified weight matrix with the oivs method.
   *
   * @param W Weight matrix to initialize.
   * @param n_rows Number of rows.
   * @return n_cols Number of columns.
   */
  void Initialize(MatType& W, const size_t n_rows, const size_t n_cols)
  {
    double b = std::abs(ActivationFunction::inv(1 - epsilon) -
        ActivationFunction::inv(epsilon));

    RandomInitialization<MatType> randomInit(-gamma, gamma);
    randomInit.Initialize(W, n_rows, n_cols);

    W = (b / (k  * n_rows)) * arma::sqrt(W + 1);
  }

 private:
  //! Parameter to control the activation region.
  const double epsilon;

  //! Parameter to control the activation region width.
  const int k;

  //! Parameter to define the uniform random range.
  const double gamma;
}; // class OivsInitialization


}; // namespace ann
}; // namespace mlpack

#endif
