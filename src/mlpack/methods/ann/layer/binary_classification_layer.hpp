/**
 * @file binary_classification_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the BinaryClassificationLayer class, which implements a
 * binary class classification layer that can be used as output layer.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_BINARY_CLASSIFICATION_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_BINARY_CLASSIFICATION_LAYER_HPP

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
   */
  BinaryClassificationLayer()
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
      output(i) = output(i) > 0.5 ? 1 : 0;
  }
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
