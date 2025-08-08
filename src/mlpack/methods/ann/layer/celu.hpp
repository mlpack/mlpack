/**
 * @file methods/ann/layer/celu.hpp
 * @author Gaurav Singh
 *
 * Definition of the CELU activation function as described by Jonathan T. Barron.
 *
 * For more information, read the following paper.
 *
 * @code
 * @article{
 *   author  = {Jonathan T. Barron},
 *   title   = {Continuously Differentiable Exponential Linear Units},
 *   year    = {2017},
 *   url     = {https://arxiv.org/pdf/1704.07483}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CELU_HPP
#define MLPACK_METHODS_ANN_LAYER_CELU_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * The CELU activation function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \left\{
 *   \begin{array}{lr}
 *    x & : x \ge 0 \\
 *    \alpha(e^(\frac{x}{\alpha}) - 1) & : x < 0
 *   \end{array}
 * \right. \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     1 & : x \ge 0 \\
 *     (\frac{f(x)}{\alpha}) + 1 & : x < 0
 *   \end{array}
 * \right.
 * @f}
 *
 * When not in training mode, there is no computation of the derivative.
 *
 * @tparam InputType The type of the layer's inputs. The layer automatically
 *     cast inputs to this type (Default: arma::mat).
 * @tparam OutputType The type of the computation which also causes the output
 *     to also be in this type. The type also allows the computation and weight
 *     type to differ from the input type (Default: arma::mat).
 */
template<typename MatType = arma::mat>
class CELU : public Layer<MatType>
{
 public:
  // Convenience typedef to access the element type of the weights and data.
  using ElemType = typename MatType::elem_type;

  /**
   * Create the CELU object using the specified parameter. The non zero
   * gradient for negative inputs can be adjusted by specifying the CELU
   * hyperparameter alpha (alpha > 0).
   *
   * @param alpha Scale parameter for the negative factor (default = 1.0).
   */
  CELU(const double alpha = 1.0);

  //! Clone the CELU object. This handles polymorphism correctly.
  CELU* Clone() const { return new CELU(*this); }


  // Virtual destructor
  virtual ~CELU() { }

  //Copy constructor
  CELU(const CELU& other);

  //Move Constructor
  CELU(CELU&& other);

  //Copy assignment operator
  CELU& operator=(const CELU& other);

  //Move assignement operator
  CELU& operator=(CELU&& other);

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

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  // CELU Hyperparameter (alpha > 0).
  double alpha;
}; // class CELU

} // namespace mlpack

// Include implementation.
#include "celu_impl.hpp"

#endif
