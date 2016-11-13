/**
 * @file binary_classification_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the BinaryClassificationLayer class, which implements a
 * binary class classification layer that can be used as output layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_BINARY_CLASSIFICATION_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_BINARY_CLASSIFICATION_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a binary classification layer that can be used as
 * output layer.
 */
class BinaryClassificationLayer
{
 public:
  /**
   * Create the BinaryClassificationLayer object.
   *
   * @param confidence The confidence used for the output class transformation.
   */
  BinaryClassificationLayer(const double confidence = 0.5) :
      confidence(confidence)
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

    for (size_t i = 0; i < output.n_elem; i++)
      output(i) = output(i) > confidence ? 1 : 0;
  }

  //! Get the confidence parameter.
  double const& Confidence() const { return confidence; }
  //! Modify the confidence parameter.
  double& Confidence() { return confidence; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(confidence, "confidence");
  }

 private:
   double confidence;

}; // class BinaryClassificationLayer

//! Layer traits for the binary class classification layer.
template <>
class LayerTraits<BinaryClassificationLayer>
{
 public:
  static const bool IsBinary = true;
  static const bool IsOutputLayer = true;
  static const bool IsBiasLayer = false;
  static const bool IsLSTMLayer = false;
  static const bool IsConnection = false;
};

} // namespace ann
} // namespace mlpack

#endif
