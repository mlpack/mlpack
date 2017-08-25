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
#ifndef MLPACK_METHODS_ANN_IMAGE_FUNCTIONS_BILINEAR_FUNCTION_HPP
#define MLPACK_METHODS_ANN_IMAGE_FUNCTIONS_BILINEAR_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <assert.h>

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
    if (output.is_empty())
      output.set_size(outRowSize * outColSize, 1);

    assert(output.n_rows == outRowSize * outColSize);
    assert(output.n_cols == 1);

    scaleRow = (double)inRowSize / (double) outRowSize;
    scaleCol = (double)inColSize / (double) outColSize;

    for (size_t i = 0; i < outRowSize; i++)
    {
      for (size_t j = 0; j < outColSize; j++)
      {
        rOrigin = std::floor(i * scaleRow);
        cOrigin = std::floor(j * scaleCol);

        if (rOrigin > inRowSize - 2)
          rOrigin = inRowSize - 2;
        if (cOrigin > inColSize - 2)
          cOrigin = inColSize - 2;

        double deltaR = i * scaleRow - rOrigin;
        double deltaC = j * scaleCol - cOrigin;
        coeff1 = (1 - deltaR) * (1 - deltaC);
        coeff2 = deltaR * (1 - deltaC);
        coeff3 = (1 - deltaR) * deltaC;
        coeff4 = deltaR * deltaC;

        output(j * outColSize + i) =
            input(cOrigin * inColSize + rOrigin) * coeff1 +
            input(cOrigin * inColSize + rOrigin + 1) * coeff2 +
            input((cOrigin + 1) * inColSize + rOrigin) * coeff3 +
            input((cOrigin + 1) * inColSize + rOrigin + 1) * coeff4;
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
    if (output.is_empty())
      output.set_size(inRowSize * inColSize, 1);

    assert(output.n_rows == inRowSize * inColSize);
    assert(output.n_cols == 1);

    if (input.n_elem == output.n_elem)
    {
      output = input;
    }
    else
    {
      scaleRow = (double)(outRowSize) / inRowSize;
      scaleCol = (double)(outColSize) / inColSize;

      for (size_t i = 0; i < inRowSize; i++)
        for (size_t j = 0; j < inColSize; j++)
        {
          rOrigin = std::floor(i * scaleRow);
          cOrigin = std::floor(j * scaleCol);

          if (rOrigin > outRowSize - 2)
            rOrigin = outRowSize - 2;
          if (cOrigin > outColSize - 2)
            cOrigin = outColSize - 2;

          double deltaR = i * scaleRow - rOrigin;
          double deltaC = j * scaleCol - cOrigin;
          coeff1 = (1 - deltaR) * (1 - deltaC);
          coeff2 = deltaR * (1 - deltaC);
          coeff3 = (1 - deltaR) * deltaC;
          coeff4 = deltaR * deltaC;

          output(j * inColSize + i) =
              input(cOrigin * outColSize + rOrigin) * coeff1 +
              input(cOrigin * outColSize + rOrigin + 1) * coeff2 +
              input((cOrigin + 1) * outColSize + rOrigin) * coeff3 +
              input((cOrigin + 1) * outColSize + rOrigin + 1) * coeff4;
      }
    }
  }
  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(inRowSize, "inRowSize");
    ar & data::CreateNVP(inColSize, "inColSize");
    ar & data::CreateNVP(outRowSize, "outRowSize");
    ar & data::CreateNVP(outColSize, "outColSize");
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
