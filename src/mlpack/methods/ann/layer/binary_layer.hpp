/**
 * @file binary_rbm.hpp
 * @author Kris Singh
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RBM_BINARY_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_RBM_BINARY_LAYER_HPP

#include "layer_types.hpp"

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>


namespace mlpack{
namespace ann{

template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class BinaryLayer
{
 public:
  /* The visible layer of the rbm
   * network.
   *
   * @param: inSize: num of visible neurons
   * @param: outSize: num of hidden neurons
   */
  BinaryLayer(size_t inSize, size_t outSize,
      bool typeVisible = true);

  // Reset the variables
  void Reset();

  /**
    * Calculate the acivations and send to the hidden layer.
    * input if of the format datapoint + bias(other layer)
    *
    * @param input Input data used for evaluating the specified function.
    * @param output Resulting output activation.
    */
  void Forward(const InputDataType&& input, OutputDataType&& output);

  /**
   * Sample the output given the input parameters
   * The sample are obtained from a distribution 
   * specified by the sampler. 
   *
   * @param input the input parameters for 
   * @param output samples from the parameters
   */
  void Sample(InputDataType&& input, OutputDataType&& output);

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

  //! Get the parameters.
  OutputDataType const& Bias() const { return ownBias; }

  //! Get the weights of the network
  OutputDataType const& Weight() const { return weight; }
  //! Modify the parameters.
  OutputDataType& Bias() { return ownBias; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

  /**
    * Calculate the pre-acivations and send to the hidden layer.
    * input if of the format datapoint + bias(other layer)
    *
    * @param input Input data used for evaluating the specified function.
    * @param output Resulting output from the linear function.
    */
  void ForwardPreActivation(const InputDataType&& input,
      OutputDataType&& output);

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored type of layer
  bool typeVisible;

  //! Locally-stored weight object of the layer.
  OutputDataType weights;

  //! Locally-stored weight paramters of the layer.
  OutputDataType weight;

  //! Locally-stored bias paramters belonging to the layer
  OutputDataType ownBias;

  //! Locally-store bias paremeter belonging to the other layer.
  OutputDataType otherBias;
}; // class BinaryLayer
} // namespace ann
} // namespace mlpack
// Include implementation.
#include "binary_layer_impl.hpp"
#endif
