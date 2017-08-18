/**
 * @file bilinear_function.hpp
 * @author Kris Singh
 *
 * Definition and implementation of the bilinear interpolation function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_IMAGE_FUNCTIONS_INTERPOLATION_FUNCTION_HPP
#define MLPACK_METHODS_ANN_IMAGE_FUNCTIONS_INTERPOLATION_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The bilinear interpolation function
 *
 */
class BiLinearFunction
{
 public:
  BiLinearFunction(
      const size_t inRowSize,
      const size_t inColSize,
      const size_t scale):
      inRowSize(inRowSize),
      inColSize(inColSize),
      outRowSize(2 * inRowSize),
      outColSize(2 * inColSize),
      scale(scale)
  {};
  /**
   * UpSample the given Image
   *
   * @param input Input image.
   * @param output The resulting interpolated output image.
   */
  template<typename eT>
  void UpSample(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    // Get dimensions.
    arma::Mat<eT> tempInput = input;
    if (input.n_rows != inRowSize || input.n_cols != inColSize)
      tempInput.reshape(inRowSize, inColSize);

    output.set_size(outRowSize, outColSize);

    if (output.n_rows > tempInput.n_rows)
      scaleRow = inRowSize / outRowSize;
    else
      scaleRow = (inRowSize - 1) / outRowSize;

    if (output.n_cols > tempInput.n_cols)
      scaleCol = inColSize / outColSize;
    else
      scaleCol = (inColSize - 1) / outColSize;

    for (size_t i = 0; i < outRowSize; i++)
      for (size_t j = 0; j < outColSize; j++)
      {
        r_origin = std::floor(i * scaleRow);
        c_origin = std::floor(j * scaleCol);
        GetCoff(i - r_origin, j - c_origin);
        output = tempInput(r_origin, c_origin) * coff1 +
                 tempInput(r_origin + 1, c_origin) * coff2 +
                 tempInput(r_origin, c_origin + 1) * coff3 +
                 tempInput(r_origin + 1, c_origin + 1) * coff4;
      }
  }

  template<typename eT>
  void DownSample(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    arma::Mat<eT> tempInput = input;
    // Get dimensions.
    if (input.n_rows != outRowSize || input.n_cols != outColSize)
      tempInput.reshape(outRowSize, outColSize);

    output.set_size(inRowSize, inColSize);

    if (output.n_rows > tempInput.n_rows)
      scaleRow = inRowSize / outRowSize;
    else
      scaleRow = (inRowSize - 1) / outRowSize;

    if (output.n_cols > tempInput.n_cols)
      scaleCol = inColSize / outColSize;
    else
      scaleCol = (inColSize - 1) / outColSize;

    for (size_t i = 0; i < outRowSize; i++)
      for (size_t j = 0; j < outColSize; j++)
      {
        r_origin = std::floor(i * scaleRow);
        c_origin = std::floor(j * scaleCol);
        GetCoff(i - r_origin, j - c_origin);
        output = tempInput(r_origin, c_origin) * coff1 +
                 tempInput(r_origin + 1, c_origin) * coff2 +
                 tempInput(r_origin, c_origin + 1) * coff3+
                 tempInput(r_origin + 1, c_origin + 1) * coff4;
      }
  }

  /**
   * Computes the first derivative of the logistic function.
   *
   * @param x Input data.
   * @return f'(x)
   */
  void GetCoff(const size_t row, const size_t col)
  {
    coff1 = (1 - row) * (1 - col);
    coff2 =  row * ( 1 - col);
    coff3 = (1 - row) * col;
    coff4 =   row * col;
  }

private:
  const size_t inRowSize;
  const size_t inColSize;
  const size_t outRowSize;
  const size_t outColSize;
  const size_t scale;
  double scaleRow;
  double scaleCol;
  size_t r_origin;
  size_t c_origin;

  size_t coff1;
  size_t coff2;
  size_t coff3;
  size_t coff4;
}; // class BiLinearFunction

} // namespace ann
} // namespace mlpack

#endif
