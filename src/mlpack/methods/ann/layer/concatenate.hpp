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
 *
 * The Concat() function to provide the concat matrix, or it can be passed to
 * the constructor.
 *
 * After this layer is applied, the shape of the data will be a vector.
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
   * Create the ConcatenateType object using the given constant matrix as the
   * data to be concatenated to the output of the forward pass.
   */
  ConcatenateType(const InputType& concat = InputType());

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

  //! Get the concat matrix.
  OutputType const& Concat() const { return concat; }
  //! Modify the concat.
  OutputType& Concat() { return concat; }

  void ComputeOutputDimensions()
  {
    // This flattens the input.
    const size_t inSize = std::accumulate(this->inputDimensions.begin(),
        this->inputDimensions.end(), 0);
    this->outputDimensions = std::vector<size_t>(this->inputDimensions.size(),
        1);
    this->outputDimensions[0] = inSize + concat.n_elem;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(cereal::base_class<Layer<InputType, OutputType>>(this));

    ar(CEREAL_NVP(concat));
  }

 private:
  //! Matrix to be concatenated to input.
  InputType concat;

}; // class Concatenate

// Standard Concatenate layer.
typedef ConcatenateType<arma::mat, arma::mat> Concatenate;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "concatenate_impl.hpp"

#endif
