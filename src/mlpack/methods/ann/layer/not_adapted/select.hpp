/**
 * @file methods/ann/layer/select.hpp
 * @author Marcus Edel
 *
 * Definition of the Select module.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SELECT_HPP
#define MLPACK_METHODS_ANN_LAYER_SELECT_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * The select module selects the specified dimensions from a given input point.
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
class SelectType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create the Select object.
   *
   * @param index The first dimension to extract from the input.
   * @param elements The number of elements that should be used.  If 0 is given,
   *      then all dimensions starting with index up to the number of dimensions
   *      are used.
   */
  SelectType(const size_t index = 0, const size_t elements = 0);

  //! Clone the SelectType object. This handles polymorphism correctly.
  SelectType* Clone() const { return new SelectType(*this); }

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

  //! Get the column index.
  const size_t& Index() const { return index; }

  //! Get the number of elements selected.
  const size_t& NumElements() const { return elements; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  const std::vector<size_t> OutputDimensions() const
  {
    std::vector<size_t> outputDimensions(inputDimensions.size(), 1);
    if (elements > 0)
    {
      outputDimensions[0] = elements;
    }
    else
    {
      // Compute the total number of dimensions.
      const size_t totalDims = std::accumulate(inputDimensions.begin(),
          inputDimensions.end(), 0);
      outputDimensions[0] = (totalDims - index);
    }

    return outputDimensions;
  }

 private:
  //! Locally-stored column index.
  size_t index;

  //! Locally-stored number of elements selected.
  size_t elements;
}; // class SelectType

// Standard Select layer.
using Select = SelectType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "select_impl.hpp"

#endif
