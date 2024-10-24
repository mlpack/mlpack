/**
 * @file methods/ann/layer/relu6.hpp
 * @author Aakash kaushik
 *
 * For more information, kindly refer to the following paper.
 *
 * @code
 * @article{Andrew G2017,
 *  author = {Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko,
 *      Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam},
 *  title = {MobileNets: Efficient Convolutional Neural Networks for Mobile
 *      Vision Applications},
 *  year = {2017},
 *  url = {https://arxiv.org/pdf/1704.04861}
 * }
 * @endcode
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RELU6_HPP
#define MLPACK_METHODS_ANN_LAYER_RELU6_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * @tparam MatType Matrix representation to accept as input and allows the
 *         computation and weight type to differ from the input type
 *         (Default: arma::mat).
 */
template<typename MatType = arma::mat>
class ReLU6Type : public Layer<MatType>
{
 public:
  /**
   * Create the ReLU6Type object.
   */
  ReLU6Type();

  //! Clone the ReLU6Type object. This handles polymorphism correctly.
  ReLU6Type* Clone() const { return new ReLU6Type(*this); }

  // Virtual destructor.
  virtual ~ReLU6Type() { }

  //! Copy the given ReLU6Type.
  ReLU6Type(const ReLU6Type& other);
  //! Take ownership of the given ReLU6Type.
  ReLU6Type(ReLU6Type&& other);
  //! Copy the given ReLU6Type.
  ReLU6Type& operator=(const ReLU6Type& other);
  //! Take ownership of the given ReLU6Type.
  ReLU6Type& operator=(ReLU6Type&& other);

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
  void Backward(const MatType& input,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g);

  //! Get size of weights.
  size_t WeightSize() const { return 0; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */);
}; // class ReLU6

// Convenience typedefs.

// Standard ReLU6 layer.
using ReLU6 = ReLU6Type<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "relu6_impl.hpp"

#endif
