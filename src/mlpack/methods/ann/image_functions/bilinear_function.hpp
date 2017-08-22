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
 /*
  * The constructor for the bilinear interpolation.
  * 
  * @param inRowSize: Number of input rows
  * @param inColSize: Number of input columns.
  * @param outRowSize: Number of output rows.
  * @param outColSie: Number of output columns.
  */
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
   * UpSample the given input.
   *
   * @param input The input matrix
   * @param output The resulting interpolated output image.
   */
  template<typename eT>
  void UpSample(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    // Get dimensions.
    if (output.is_empty() && (output.n_cols != outColSize || output.n_rows != outRowSize))
      output = arma::resize(output, outRowSize, outColSize);

    scaleRow = (double)input.n_rows / (double) output.n_rows;
    scaleCol = (double)input.n_cols / (double) output.n_cols;

    for (size_t i = 0; i < output.n_rows; i++)
    {
      for (size_t j = 0; j < output.n_cols; j++)
      {
        rOrigin = std::floor(i * scaleRow);
        cOrigin = std::floor(j * scaleCol);

        if (rOrigin < 0)
          rOrigin = 0;
        if (cOrigin < 0)
          cOrigin = 0;

        /*
        if (rOrigin > input.n_rows - 2)
          rOrigin = input.n_rows - 2;
        if (cOrigin > input.n_cols - 2)
          cOrigin = input.n_cols - 2;
        */

        double deltaR = i * scaleRow - rOrigin;
        double deltaC = j * scaleCol - cOrigin;
        coeff1 = (1 - deltaR) * (1 - deltaC);
        coeff2 = deltaR * (1 - deltaC);
        coeff3 = (1 - deltaR) * deltaC;
        coeff4 = deltaR * deltaC;


        output(i, j) =  input(cOrigin * input.n_rows + rOrigin) * coeff1 +
                        input(cOrigin * input.n_rows + rOrigin + 1) * coeff2 +
                        input((cOrigin + 1) * input.n_rows + rOrigin) * coeff3 +
                        input((cOrigin + 1) * input.n_rows + rOrigin+1) * coeff4;
      }
    }
  }

  /**
   * DownSample the given input.
   *
   * @param input The input matrix
   * @param output The resulting down-sampled output image.
   */
  template<typename eT>
  void DownSample(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {

    if (output.is_empty() && (output.n_cols != inColSize || output.n_rows != inRowSize))
      output = arma::resize(output, inRowSize, inColSize);

    scaleRow = (double)(input.n_rows - 1) / output.n_rows;
    scaleCol = (double)(input.n_cols - 1) / output.n_cols;

    for (size_t i = 0; i < output.n_rows; i++)
      for (size_t j = 0; j < output.n_cols; j++)
      {
        rOrigin = std::floor(i * scaleRow);
        cOrigin = std::floor(j * scaleCol);

        /*
        if (rOrigin < 0)
          rOrigin = 0;
        if (cOrigin < 0)
          cOrigin = 0;
        */

        if (rOrigin > input.n_rows - 2)
          rOrigin = input.n_rows - 2;
        if (cOrigin > input.n_cols - 2)
          cOrigin = input.n_cols - 2;

        double deltaR = i * scaleRow - rOrigin;
        double deltaC = j * scaleCol - cOrigin;
        coeff1 = (1 - deltaR) * (1 - deltaC);
        coeff2 = deltaR * (1 - deltaC);
        coeff3 = (1 - deltaR) * deltaC;
        coeff4 = deltaR * deltaC;

        output(i, j) =  input(cOrigin * input.n_rows + rOrigin) * coeff1 +
                        input(cOrigin * input.n_rows + rOrigin + 1) * coeff2 +
                        input((cOrigin + 1) * input.n_rows + rOrigin) * coeff3 +
                        input((cOrigin + 1) * input.n_rows + rOrigin+1) * coeff4;
      }
  }

 private:
  //! Locally stored row size of the input.
  const size_t inRowSize;
  //! Locally stored column size of the input.
  const size_t inColSize;
  //! Locally stored row size of the output.
  const size_t outRowSize;
  //! Locally stored column size of the input.
  const size_t outColSize;

  //! Locally stored scaling factor along row.
  double scaleRow;
  //! Locally stored scaling factor along row.
  double scaleCol;
  //! Locally stored interger part of the current row idx in input.
  double rOrigin;
  //! Locally stored interger part of the current column idx in input.
  double cOrigin;

  //! Locally stored coefficient around given idx.
  double coeff1;
  //! Locally stored coefficient around given idx.
  double coeff2;
  //! Locally stored coefficient around given idx.
  double coeff3;
  //! Locally stored coefficient around given idx.
  double coeff4;
}; // class BiLinearFunction

} // namespace ann
} // namespace mlpack

#endif
