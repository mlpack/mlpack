/**
 * @file methods/ann/layer/linear3d.hpp
 * @author Mrityunjay Tripathi
 *
 * Definition of the Linear layer class which accepts 3D input.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_LINEAR3D_HPP
#define MLPACK_METHODS_ANN_LAYER_LINEAR3D_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/regularizer/no_regularizer.hpp>

#include "layer.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Linear3D layer class. The Linear class represents a
 * single layer of a neural network.
 *
 * Shape of input : (inSize * nPoints, batchSize)
 * Shape of output : (outSize * nPoints, batchSize)
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat,
    typename RegularizerType = NoRegularizer
>
class Linear3DType : public Layer<InputType, OutputType>
{
 public:
  //! Create the Linear3D object.
  Linear3DType();

  /**
   * Create the Linear3D layer object using the specified number of output
   * units.
   *
   * @param outSize The number of output units.
   * @param regularizer The regularizer to use, optional.
   */
  Linear3DType(const size_t outSize,
               RegularizerType regularizer = RegularizerType());

  //! Clone the Linear3DType object. This handles polymorphism correctly.
  Linear3DType* Clone() const { return new Linear3DType(*this); }

  /*
   * Reset the layer parameter.
   */
  void SetWeights(typename OutputType::elem_type* weightsPtr);

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
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient);

  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! Get the weight of the layer.
  OutputType const& Weight() const { return weight; }
  //! Modify the weight of the layer.
  OutputType& Weight() { return weight; }

  //! Get the bias of the layer.
  OutputType const& Bias() const { return bias; }
  //! Modify the bias weights of the layer.
  OutputType& Bias() { return bias; }

  size_t WeightSize() const { return outSize * (inSize + 1); }

  void ComputeOutputDimensions()
  {
    // The Linear3D layer shares weights for each row of the input, and
    // duplicates it across the columns.  Thus, we only change the number of
    // rows.
    inSize = std::accumulate(this->inputDimensions.begin(),
        this->inputDimensions.end(), 0);
    this->outputDimensions = this->inputDimensions;
    this->outputDimensions[0] = outSize;
  }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight object.
  OutputType weights;

  //! Locally-stored weight parameters.
  OutputType weight;

  //! Locally-stored bias term parameters.
  OutputType bias;

  //! Locally-stored regularizer object.
  RegularizerType regularizer;
}; // class Linear

// Standard Linear3D layer.
typedef Linear3DType<arma::mat, arma::mat, NoRegularizer> Linear3D;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "linear3d_impl.hpp"

#endif
