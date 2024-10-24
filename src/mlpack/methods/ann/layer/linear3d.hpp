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

/**
 * Implementation of the Linear3D layer class. The Linear class represents a
 * single layer of a neural network.
 *
 * Shape of input : (inSize * nPoints, batchSize)
 * Shape of output : (outSize * nPoints, batchSize)
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<
    typename MatType = arma::mat,
    typename RegularizerType = NoRegularizer
>
class Linear3DType : public Layer<MatType>
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

  // Virtual destructor.
  virtual ~Linear3DType() { }

  //! Copy the given Linear3DType (but not weights).
  Linear3DType(const Linear3DType& other);
  //! Take ownership of the given Linear3DType (but not weights).
  Linear3DType(Linear3DType&& other);
  //! Copy the given Linear3DType (but not weights).
  Linear3DType& operator=(const Linear3DType& other);
  //! Take ownership of the given Linear3DType (but not weights).
  Linear3DType& operator=(Linear3DType&& other);

  /*
   * Reset the layer parameter.
   */
  void SetWeights(const MatType& weightsIn);

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
   * f(x) by propagating x backwards trough f. Using the results from the feed
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

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& input,
                const MatType& error,
                MatType& gradient);

  //! Get the parameters.
  MatType const& Parameters() const { return weights; }
  //! Modify the parameters.
  MatType& Parameters() { return weights; }

  //! Get the weight of the layer.
  MatType const& Weight() const { return weight; }
  //! Modify the weight of the layer.
  MatType& Weight() { return weight; }

  //! Get the bias of the layer.
  MatType const& Bias() const { return bias; }
  //! Modify the bias weights of the layer.
  MatType& Bias() { return bias; }

  //! Return the number of weight elements.
  size_t WeightSize() const { return outSize * (this->inputDimensions[0] + 1); }

  //! Compute the output dimensions for the layer, using `InputDimensions()`.
  void ComputeOutputDimensions();

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight object.
  MatType weights;

  //! Locally-stored weight parameters.
  MatType weight;

  //! Locally-stored bias term parameters.
  MatType bias;

  //! Locally-stored regularizer object.
  RegularizerType regularizer;
}; // class Linear

// Standard Linear3D layer.
using Linear3D = Linear3DType<arma::mat, NoRegularizer>;

} // namespace mlpack

// Include implementation.
#include "linear3d_impl.hpp"

#endif
