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
      const size_t outRowSize,
      const size_t outColSize):
      inRowSize(inRowSize),
      inColSize(inColSize),
      outRowSize(outRowSize),
      outColSize(outColSize)
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
    output.set_size(outRowSize, outColSize);
    if (input.n_rows != inRowSize || input.n_cols != inColSize)
      tempInput.reshape(inRowSize, inColSize);

    scaleRow = (double)(input.n_rows - 1) / output.n_rows;
    scaleCol = (double)(input.n_cols - 1) / output.n_cols;
    
    for (size_t i = 0; i < output.n_rows; i++)
      for (size_t j = 0; j < output.n_cols; j++)
      {
        r_origin = std::floor(i * scaleRow);
        c_origin = std::floor(j * scaleCol);

        if (r_origin > tempInput.n_rows - 2)
          r_origin = tempInput.n_rows - 2;
        if (c_origin > tempInput.n_cols - 2)
          c_origin = tempInput.n_cols - 2;

        double deltaR = i * scaleRow - r_origin;
        double deltaC = j * scaleCol - c_origin;
        coff1 = (1 - deltaC) * (1 - deltaC);
        coff2 =  deltaR * ( 1 - deltaC);
        coff3 = (1 - deltaR) * deltaC;
        coff4 =  deltaR * deltaC;
        output(i, j) = tempInput(r_origin, c_origin) * coff1 +
                 tempInput(r_origin + 1, c_origin) * coff2 +
                 tempInput(r_origin, c_origin + 1) * coff3+
                 tempInput(r_origin + 1, c_origin + 1) * coff4;
      }
  }

  template<typename eT>
  void DownSample(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    arma::Mat<eT> tempInput = input;
    output.set_size(inRowSize, inColSize);
    // Get dimensions.
    if (input.n_rows != outRowSize || input.n_cols != outColSize)
      tempInput.reshape(outRowSize, outColSize);
    
    scaleRow = (double)(input.n_rows - 1) / output.n_rows;
    scaleCol = (double)(input.n_cols - 1) / output.n_cols;

    for (size_t i = 0; i < output.n_rows; i++)
      for (size_t j = 0; j < output.n_cols; j++)
      {
        r_origin = std::floor(i * scaleRow);
        c_origin = std::floor(j * scaleCol);

        if (r_origin > tempInput.n_rows - 2)
          r_origin = tempInput.n_rows - 2;
        if (c_origin > tempInput.n_cols - 2)
          c_origin = tempInput.n_cols - 2;

        double deltaR = i * scaleRow - r_origin;
        double deltaC = j * scaleCol - c_origin;
        coff1 = (1 - deltaC) * (1 - deltaC);
        coff2 =  deltaR * ( 1 - deltaC);
        coff3 = (1 - deltaR) * deltaC;
        coff4 =  deltaR * deltaC;
        output(i, j) = tempInput(r_origin, c_origin) * coff1 +
                 tempInput(r_origin + 1, c_origin) * coff2 +
                 tempInput(r_origin, c_origin + 1) * coff3+
                 tempInput(r_origin + 1, c_origin + 1) * coff4;
      }
  }

private:
  const size_t inRowSize;
  const size_t inColSize;
  const size_t outRowSize;
  const size_t outColSize;

  double scaleRow;
  double scaleCol;
  size_t r_origin;
  size_t c_origin;

  double coff1;
  double coff2;
  double coff3;
  double coff4;
}; // class BiLinearFunction

} // namespace ann
} // namespace mlpack

#endif
