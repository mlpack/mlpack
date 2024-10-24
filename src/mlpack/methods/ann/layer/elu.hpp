/**
 * @file methods/ann/layer/elu.hpp
 * @author Vivek Pal
 * @author Dakshit Agrawal
 *
 * Definition of the ELU activation function as described by Djork-Arne Clevert,
 * Thomas Unterthiner and Sepp Hochreiter.
 *
 * Definition of the SELU function as introduced by Klambauer et. al. in Self
 * Neural Networks.  The SELU activation function keeps the mean and variance of
 * the input invariant.
 *
 * In short, SELU = lambda * ELU, with 'alpha' and 'lambda' fixed for
 * normalized inputs.
 *
 * Hence both ELU and SELU are implemented in the same file, with
 * lambda = 1 for ELU function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ELU_HPP
#define MLPACK_METHODS_ANN_LAYER_ELU_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * The ELU activation function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \left\{
 *   \begin{array}{lr}
 *    x & : x > 0 \\
 *    \alpha(e^x - 1) & : x \le 0
 *   \end{array}
 * \right. \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     1 & : x > 0 \\
 *     f(x) + \alpha & : x \le 0
 *   \end{array}
 * \right.
 * @f}
 *
 * For more information, read the following paper:
 *
 * @code
 * @article{Clevert2015,
 *   author  = {Djork{-}Arn{\'{e}} Clevert and Thomas Unterthiner and
 *              Sepp Hochreiter},
 *   title   = {Fast and Accurate Deep Network Learning by Exponential Linear
 *              Units (ELUs)},
 *   journal = {CoRR},
 *   year    = {2015},
 *   url     = {https://arxiv.org/abs/1511.07289}
 * }
 * @endcode
 *
 * The SELU activation function is defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \left\{
 *   \begin{array}{lr}
 *    \lambda * x & : x > 0 \\
 *    \lambda * \alpha(e^x - 1) & : x \le 0
 *   \end{array}
 * \right. \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     \lambda & : x > 0 \\
 *     f(x) + \lambda * \alpha & : x \le 0
 *   \end{array}
 * \right.
 * @f}
 *
 * For more information, read the following paper:
 *
 * @code
 * @article{Klambauer2017,
 *   author  = {Gunter Klambauer and Thomas Unterthiner and
 *              Andreas Mayr},
 *   title   = {Self-Normalizing Neural Networks},
 *   journal = {Advances in Neural Information Processing Systems},
 *   year    = {2017},
 *   url = {https://arxiv.org/abs/1706.02515}
 * }
 * @endcode
 *
 * In testing mode, there is no computation of the derivative.
 *
 * @note Make sure to use SELU activation function with normalized inputs and
 *       weights initialized with Lecun Normal Initialization.
 *
 * @tparam InputType The type of the layer's inputs. The layer automatically
 *     cast inputs to this type (Default: arma::mat).
 * @tparam OutputType The type of the computation which also causes the output
 *     to also be in this type. The type also allows the computation and weight
 *     type to differ from the input type (Default: arma::mat).
 */
template <typename MatType = arma::mat>
class ELUType : public Layer<MatType>
{
 public:
  /**
   * Create the ELU object.
   *
   * NOTE: Use this constructor for SELU activation function.
   */
  ELUType();

  /**
   * Create the ELU object using the specified parameter. The non zero
   * gradient for negative inputs can be adjusted by specifying the ELU
   * hyperparameter alpha (alpha > 0).
   *
   * @note Use this constructor for ELU activation function.
   * @param alpha Scale parameter for the negative factor.
   */
  ELUType(const double alpha);

  //! Clone the ELUType object. This handles polymorphism correctly.
  ELUType* Clone() const { return new ELUType(*this); }

  // Virtual destructor.
  virtual ~ELUType() { }

  // Copy constructor.
  ELUType(const ELUType& other);

  // Move Constructor.
  ELUType(ELUType&& other);

  // Copy assignment operator.
  ELUType& operator=(const ELUType& other);

  // Move assignement operator.
  ELUType& operator=(ELUType&& other);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& /* input */,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g);

  //! Get the non zero gradient.
  double const& Alpha() const { return alpha; }
  //! Modify the non zero gradient.
  double& Alpha() { return alpha; }

  //! Get the lambda parameter.
  double const& Lambda() const { return lambda; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally stored first derivative of the activation function.
  MatType derivative;

  //! ELU Hyperparameter (0 < alpha)
  //! SELU parameter fixed to 1.6732632423543774 for normalized inputs.
  double alpha;

  //! Lambda parameter used for multiplication of ELU function.
  //! For ELU activation function, lambda = 1.
  //! For SELU activation function, lambda = 1.0507009873554802 for normalized
  //! inputs.
  double lambda;
}; // class ELUType

// Convenience typedefs.

// ELU layer.
using ELU = ELUType<arma::mat>;

// SELU layer.
using SELU = ELUType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "elu_impl.hpp"

#endif
