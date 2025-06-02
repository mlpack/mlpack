/**
 * @file methods/ann/layer/concatenate.hpp
 * @author Atharva Khandait
 *
 * Definition of the Concatenate class that concatenate a constant matrix to
 * the incoming data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONCATENATE_HPP
#define MLPACK_METHODS_ANN_LAYER_CONCATENATE_HPP

#include <mlpack/prereqs.hpp>
#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the Concatenate module class. The Concatenate module
 * concatenates a constant given matrix to the incoming data.
 *
 * The Concat() function provides the concat matrix, or it can be passed to
 * the constructor.
 *
 * After this layer is applied, the shape of the data will be a vector.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class ConcatenateType : public Layer<MatType>
{
 public:
  /**
   * Create the ConcatenateType object using the given constant matrix as the
   * data to be concatenated to the output of the forward pass.
   */
  ConcatenateType(const MatType& concat = MatType());

  //! Clone the ConcatenateType object. This handles polymorphism correctly.
  ConcatenateType* Clone() const { return new ConcatenateType(*this); }

  // Virtual destructor.
  virtual ~ConcatenateType() { }

  //! Copy the given ConcatenateType layer.
  ConcatenateType(const ConcatenateType& other);
  //! Take ownership of the given ConcatenateType layer.
  ConcatenateType(ConcatenateType&& other);
  //! Copy the given ConcatenateType layer.
  ConcatenateType& operator=(const ConcatenateType& other);
  //! Take ownership of the given ConcatenateType layer.
  ConcatenateType& operator=(ConcatenateType&& other);

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

  //! Get the concat matrix.
  MatType const& Concat() const { return concat; }
  //! Modify the concat.
  MatType& Concat() { return concat; }

  //! Compute the output dimensions of the layer based on `InputDimensions()`.
  void ComputeOutputDimensions();

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Matrix to be concatenated to input.
  MatType concat;
}; // class Concatenate

// Standard Concatenate layer.
using Concatenate = ConcatenateType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "concatenate_impl.hpp"

#endif
