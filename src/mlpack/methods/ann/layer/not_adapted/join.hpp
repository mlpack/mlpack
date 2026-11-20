/**
 * @file methods/ann/layer/join.hpp
 * @author Marcus Edel
 *
 * Definition of the Join module.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_JOIN_HPP
#define MLPACK_METHODS_ANN_LAYER_JOIN_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

// TODO: should we clarify the comments?  This seems to join together points of
// a different batch
// TODO: I don't understand this layer well enough to update it...
/**
 * Implementation of the Join module class. The Join class accumulates
 * the output of various modules.
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class JoinType : public Layer<InputType, OutputType>
{
 public:
  //! Create the JoinType object.
  JoinType();

  //! Clone the JoinType object. This handles polymorphism correctly.
  JoinType* Clone() const { return new JoinType(*this); }

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

  // This layer simply flattens its input into a vector.
  const std::vector<size_t> OutputDimensions() const
  {
    // TODO: it's not clear what to do here
    std::vector<size_t> result(inputDimensions.size(), 0);
    result[0] = std::accumulate(inputDimensions.begin(), inputDimensions.end(),
        0);
    return result;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of input rows.
  size_t inSizeRows;

  //! Locally-stored number of input cols.
  size_t inSizeCols;
}; // class JoinType

//Standard Join layer.
using Join = JoinType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "join_impl.hpp"

#endif
