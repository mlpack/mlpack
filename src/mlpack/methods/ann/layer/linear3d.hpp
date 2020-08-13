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
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/regularizer/no_regularizer.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Linear3D layer class. The Linear class represents a
 * single layer of a neural network.
 *
 * Shape of input : (inSize * nPoints, batchSize)
 * Shape of output : (outSize * nPoints, batchSize)
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat,
    typename RegularizerType = NoRegularizer
>
class Linear3D
{
 public:
  //! Create the Linear3D object.
  Linear3D();

  /**
   * Create the Linear3D layer object using the specified number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param regularizer The regularizer to use, optional.
   */
  Linear3D(const size_t inSize,
           const size_t outSize,
           RegularizerType regularizer = RegularizerType());

  /*
   * Reset the layer parameter.
   */
  void Reset();

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& input,
                const arma::Mat<eT>& error,
                arma::Mat<eT>& gradient);

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the input size.
  size_t InputSize() const { return inSize; }

  //! Get the output size.
  size_t OutputSize() const { return outSize; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  //! Get the weight of the layer.
  OutputDataType const& Weight() const { return weight; }
  //! Modify the weight of the layer.
  OutputDataType& Weight() { return weight; }

  //! Get the bias of the layer.
  OutputDataType const& Bias() const { return bias; }
  //! Modify the bias weights of the layer.
  OutputDataType& Bias() { return bias; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored weight parameters.
  OutputDataType weight;

  //! Locally-stored bias term parameters.
  OutputDataType bias;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored regularizer object.
  RegularizerType regularizer;
}; // class Linear

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "linear3d_impl.hpp"

#endif
