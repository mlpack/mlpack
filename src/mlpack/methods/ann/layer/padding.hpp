/**
 * @file padding.hpp
 * @author Saksham Bansal
 *
 * Definition of the Padding class that pads the incoming data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_PADDING_HPP
#define MLPACK_METHODS_ANN_LAYER_PADDING_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Padding module class. The Padding module applies a bias term
 * to the incoming data.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class Padding
{
 public:
  /**
   * Create the Padding object using the specified number of output units.
   *
   * @param padWLeft Left padding width of the input.
   * @param padWLeft Right padding width of the input.
   * @param padHTop Top padding height of the input.
   * @param padHBottom Bottom padding height of the input.
   * @param aW number of extra zeros added to the right of input.
   * @param aH number of extra zeros added to the top of input.
   */
  Padding(const size_t padWLeft = 0,
          const size_t padWRight = 0,
          const size_t padHTop = 0,
          const size_t padHBottom = 0);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output,
               size_t aW = 0,
               size_t aH = 0);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>&& /* input */,
                const arma::Mat<eT>&& gy,
                arma::Mat<eT>&& g);

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored left padding width.
  size_t padWLeft;

  //! Locally-stored right padding width.
  size_t padWRight;

  //! Locally-stored top padding height.
  size_t padHTop;

  //! Locally-stored bottom padding height.
  size_t padHBottom;

  //! Locally-stored number of rows and columns of input.
  size_t nRows, nCols;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class Padding

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "padding_impl.hpp"

#endif
