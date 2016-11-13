/**
 * @file one_hot_layer.hpp
 * @author Shangtong Zhang
 *
 * Definition of the OneHotLayer class, which implements a standard network
 * layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ONE_HOT_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_ONE_HOT_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a one hot classification layer that can be used as
 * output layer.
 */
class OneHotLayer
{
 public:
  /**
   * Create the OneHotLayer object.
   */
  OneHotLayer()
  {
    // Nothing to do here.
  }

  /*
   * Calculate the error using the specified input activation and the target.
   * The error is stored into the given error parameter.
   *
   * @param inputActivations Input data used for evaluating the network.
   * @param target Target data used for evaluating the network.
   * @param error The calculated error with respect to the input activation and
   * the given target.
   */
  template<typename DataType>
  void CalculateError(const DataType& inputActivations,
                      const DataType& target,
                      DataType& error)
  {
    error = inputActivations - target;
  }

  /*
   * Calculate the output class using the specified input activation.
   *
   * @param inputActivations Input data used to calculate the output class.
   * @param output Output class of the input activation.
   */
  template<typename DataType>
  void OutputClass(const DataType& inputActivations, DataType& output)
  {
    output = inputActivations;
    output.zeros();

    arma::uword maxIndex = 0;
    inputActivations.max(maxIndex);
    output(maxIndex) = 1;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& /* ar */, const unsigned int /* version */)
  {
    /* Nothing to do here */
  }
}; // class OneHotLayer

//! Layer traits for the one-hot class classification layer.
template <>
class LayerTraits<OneHotLayer>
{
 public:
  static const bool IsBinary = true;
  static const bool IsOutputLayer = true;
  static const bool IsBiasLayer = false;
  static const bool IsConnection = false;
};

} // namespace ann
} // namespace mlpack


#endif
