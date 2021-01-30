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
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Concatenate module class. The Concatenate module
 * concatenates a constant given matrix to the incoming data.
 * Note: Users need to use the Concat() function to provide the concat matrix.
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
class ConcatenateType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create the ConcatenateType object using the specified number of output units.
   */
  ConcatenateType(const InputType& concat = InputType());

  //! Copy constructor.
  ConcatenateType(const ConcatenateType& layer);

  //! Move constructor.
  ConcatenateType(ConcatenateType&& layer);

  //! Operator= copy constructor.
  ConcatenateType& operator=(const ConcatenateType& layer);

  //! Operator= move constructor.
  ConcatenateType& operator=(ConcatenateType&& layer);

  //! Clone the ConcatenateType object. This handles polymorphism correctly.
	ConcatenateType* Clone() const { return new ConcatenateType(*this); }

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

  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! Get the output parameter.
  OutputType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputType& Delta() { return delta; }

  //! Get the concat matrix.
  OutputType const& Concat() const { return concat; }
  //! Modify the concat.
  OutputType& Concat() { return concat; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */)
  {
    // Nothing to do here.
  }

 private:
  //! Locally-stored number of input rows.
  size_t inRows;

  //! Locally-stored weight object.
  OutputType weights;

  //! Locally-stored delta object.
  OutputType delta;

  //! Locally-stored output parameter object.
  OutputType outputParameter;

  //! Locally-stored matrix to be concatenated to input.
  InputType concat;
}; // class Concatenate

// Standard Concatenate layer.
typedef ConcatenateType<arma::mat, arma::mat> Concatenate;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "concatenate_impl.hpp"

#endif
