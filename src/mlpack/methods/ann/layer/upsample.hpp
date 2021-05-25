/**
 * @file methods/ann/layer/upsample.hpp
 * @author Abhinav Anand
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_BILINEAR_INTERPOLATION_HPP
#define MLPACK_METHODS_ANN_LAYER_BILINEAR_INTERPOLATION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Definition and Implementation of the Upsample Layer.
 *
 * Upsample is an mathematical technique, primarily used for
 * scaling purposes. The upsampling is done using one of the
 * following methods - nearest, bilinear and bicubic. The input
 * should be a 2D matrix and it can have a number of channels/units.
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
class Upsample
{
 public:
  //! Create the Upsample object.
  Upsample();

  /**
   * The constructor for the Upsample.
   *
   * @param inRowSize Number of input rows.
   * @param inColSize Number of input columns.
   * @param outRowSize Number of output rows.
   * @param outColSize Number of output columns.
   * @param depth Number of input slices.
   * @param mode Mode of interpolation.
   */
  Upsample(const size_t inRowSize,
           const size_t inColSize,
           const size_t outRowSize,
           const size_t outColSize,
           const size_t depth,
           const string mode = "nearest");

  /**
   * Forward pass through the layer. The layer interpolates
   * the matrix using the given Bilinear Interpolation method.
   *
   * @param input The input matrix.
   * @param output The resulting interpolated output matrix.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass. Since the layer does not have any learn-able parameters,
   * we just have to down-sample the gradient to make its size compatible with
   * the input size.
   *
   * @param * (input) The input matrix.
   * @param gradient The computed backward gradient.
   * @param output The resulting down-sampled output.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /*input*/,
                const arma::Mat<eT>& gradient,
                arma::Mat<eT>& output);

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the row size of the input.
  size_t const& InRowSize() const { return inRowSize; }
  //! Modify the row size of the input.
  size_t& InRowSize() { return inRowSize; }

  //! Get the column size of the input.
  size_t const& InColSize() const { return inColSize; }
  //! Modify the column size of the input.
  size_t& InColSize() { return inColSize; }

  //! Get the row size of the output.
  size_t const& OutRowSize() const { return outRowSize; }
  //! Modify the row size of the output.
  size_t& OutRowSize() { return outRowSize; }

  //! Get the column size of the output.
  size_t const& OutColSize() const { return outColSize; }
  //! Modify the column size of the output.
  size_t& OutColSize() { return outColSize; }

  //! Get the depth of the input.
  size_t const& InDepth() const { return depth; }
  //! Modify the depth of the input.
  size_t& InDepth() { return depth; }

  //! Get the mode of interpolation.
  string const& Mode() const { return mode; }
  //! Modify the depth of interpolation.
  string& Mode() { return mode; }

  //! Get the shape of the input.
  size_t InputShape() const
  {
    return inRowSize;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally stored row size of the input.
  size_t inRowSize;
  //! Locally stored column size of the input.
  size_t inColSize;
  //! Locally stored row size of the output.
  size_t outRowSize;
  //! Locally stored column size of the input.
  size_t outColSize;
  //! Locally stored depth of the input.
  size_t depth;
  //! Locally stored mode of interpolation.
  string mode.
  //! Locally stored number of input points.
  size_t batchSize;
  // Locally-stored coeffs precomputation.
  OutputDataType coeffs_pre;
  // Locally-stored submat index precomputation.
  OutputDataType index_pre;
  //! Locally-stored delta object.
  OutputDataType delta;
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class Upsample

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "upsample_impl.hpp"

#endif
