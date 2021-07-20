/**
 * @file methods/ann/layer/add.hpp
 * @author Marcus Edel
 *
 * Definition of the Add class that applies a bias term to the incoming data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADD_HPP
#define MLPACK_METHODS_ANN_LAYER_ADD_HPP

#include <mlpack/prereqs.hpp>
#include "layer.hpp"
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Add module class. The Add module applies a bias term
 * to the incoming data.
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class AddType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create the Add object.  The output size of the layer will be the same as
   * the input size.
   */
  AddType();

  //! Clone the AddType object. This handles polymorphism correctly.
  AddType* Clone() const { return new AddType(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
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
   * @param * (input) The propagated input.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& /* input */,
                const OutputType& error,
                OutputType& gradient);

  //! Return the weights of the network.
  const OutputType& Parameters() const { return weights; }
  //! Modify the weights of the network.
  OutputType& Parameters() { return weights; }

  //! Get the size of weights.
  size_t WeightSize() const { return outSize; }

  void ComputeOutputDimensions()
  {
    this->outputDimensions = this->inputDimensions;
    outSize = std::accumulate(this->outputDimensions.begin(),
        this->outputDimensions.end(), 0);
  }

  void SetWeights(typename OutputType::elem_type* weightPtr);

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight object.
  OutputType weights;
}; // class Add

// Standard Add layer.
typedef AddType<arma::mat, arma::mat> Add;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "add_impl.hpp"

#endif
