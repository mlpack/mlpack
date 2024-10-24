/**
 * @file methods/ann/layer/sequential.hpp
 * @author Marcus Edel
 *
 * Definition of the Sequential class, which acts as a feed-forward fully
 * connected network container.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SEQUENTIAL_HPP
#define MLPACK_METHODS_ANN_LAYER_SEQUENTIAL_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the Sequential class. The sequential class works as a
 * feed-forward fully connected network container which plugs various layers
 * together.
 *
 * This class can also be used as a container for a residual block. In that
 * case, the sizes of the input and output matrices of this class should be
 * equal. A typedef has been added for use as a Residual<> class.
 *
 * For more information, refer the following paper.
 *
 * @code
 * @article{He15,
 *   author    = {Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun},
 *   title     = {Deep Residual Learning for Image Recognition},
 *   year      = {2015},
 *   url       = {https://arxiv.org/abs/1512.03385},
 *   eprint    = {1512.03385},
 * }
 * @endcode
 *
 * Note: If this class is used as the first layer of a network, it should be
 *       preceded by IdentityLayer<>.
 *
 * Note: This class should at least have two layers for a call to its Gradient()
 *       function.
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam Residual If true, use the object as a Residual block.
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat,
    bool Residual = false
>
class SequentialType : public MultiLayer<InputType, OutputType>
{
 public:
  /**
   * Create the Sequential object.
   */
  SequentialType();

  /**
   * Create the Sequential object using the specified parameters.
   *
   * @param ownsLayers If true, then this module will delete its layers when
   *      deallocated.
   */
  SequentialType(const bool ownsLayers);

  //! Copy constructor.
  SequentialType(const SequentialType& layer);

  //! Copy assignment operator.
  SequentialType& operator=(const SequentialType& layer);

  //! Destroy the Sequential object.
  ~SequentialType();

  //! Clone the SequentialType object. This handles polymorphism correctly.
  SequentialType* Clone() const { return new SequentialType(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& /* gradient */);

  size_t InputShape() const;

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Indicator if we already initialized the model.
  bool reset;

  //! Whether we are responsible for deleting the layers held in this module.
  bool ownsLayers;
}; // class SequentialType

// Standard Sequential layer.
using Sequential = SequentialType<arma::mat, arma::mat, false>;

// Standard Residual layer.
using Residual = SequentialType<arma::mat, arma::mat, true>;

} // namespace mlpack

// Include implementation.
#include "sequential_impl.hpp"

#endif
