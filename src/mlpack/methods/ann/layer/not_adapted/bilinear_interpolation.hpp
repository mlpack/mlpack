// Temporarily drop.
/**
 * @file methods/ann/layer/bilinear_interpolation.hpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
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

/**
 * Definition and Implementation of the Bilinear Interpolation Layer.
 *
 * Bilinear Interpolation is an mathematical technique, primarily used for
 * scaling purposes. It is an extension of linear interpolation, for
 * interpolating functions of two variables on a rectangular grid. The key
 * idea is to perform linear interpolation first in one direction (e.g., along
 * x-axis), and then again in the other direction (i.e., y-axis), on four
 * different known points in the grid. This way, we represent any arbitrary
 * point, present within the grid, as a function of those four points.
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
class BilinearInterpolationType : public Layer<InputType, OutputType>
{
 public:
  //! Create the BilinearInterpolationType object.
  BilinearInterpolationType();

  // TODO: use scaleFactors instead of outRowSize and outColSize

  /**
   * The constructor for the Bilinear Interpolation.  The input size will be set
   * by the given input when the layer is used.
   *
   * @param outRowSize Number of output rows.
   * @param outColSize Number of output columns.
   */
  BilinearInterpolationType(const size_t outRowSize,
                            const size_t outColSize);

  /**
   * Forward pass through the layer. The layer interpolates
   * the matrix using the given Bilinear Interpolation method.
   *
   * @param input The input matrix.
   * @param output The resulting interpolated output matrix.
   */
  void Forward(const InputType& input, OutputType& output);

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
  void Backward(const InputType& /*input*/,
                const OutputType& gradient,
                OutputType& output);

  const std::vector<size_t>& OutputDimensions() const
  {
    std::vector<size_t> result(this->inputDimensions.size(), 0);
    result[0] = outRowSize;
    result[1] = outColSize;
    if (result.size() > 2)
    {
      for (size_t i = 0; i < result.size(); ++i)
        result[i] = this->inputDimensions[i];
    }
    return result;
  }

  //! Get the row size of the output.
  size_t const& OutRowSize() const { return outRowSize; }
  //! Modify the row size of the output.
  size_t& OutRowSize() { return outRowSize; }

  //! Get the column size of the output.
  size_t const& OutColSize() const { return outColSize; }
  //! Modify the column size of the output.
  size_t& OutColSize() { return outColSize; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally stored row size of the output.
  size_t outRowSize;

  //! Locally stored column size of the input.
  size_t outColSize;
}; // class BilinearInterpolation

// Standard BilinearInterpolation layer.
using BilinearInterpolation = BilinearInterpolationType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "bilinear_interpolation_impl.hpp"

#endif
