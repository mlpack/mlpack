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
  * @param depth: Number of slices of input.
  */
 public:
  BiLinearFunction(
      const size_t inRowSize,
      const size_t inColSize,
      const size_t outRowSize,
      const size_t outColSize,
      const size_t depth = 1):
      inRowSize(inRowSize),
      inColSize(inColSize),
      outRowSize(outRowSize),
      outColSize(outColSize),
      depth(depth)
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
      output.set_size(outRowSize * outColSize * depth, 1);

    assert(output.n_rows == outRowSize * outColSize);
    assert(output.n_cols == 1);

    double scaleRow = (double)inRowSize / (double) outRowSize;
    double scaleCol = (double)inColSize / (double) outColSize;

    for (size_t k = 0; k < depth; k++)
    {
      for (size_t i = 0; i < outRowSize; i++)
      {
        for (size_t j = 0; j < outColSize; j++)
        {
          double rOrigin = std::floor(i * scaleRow);
          double cOrigin = std::floor(j * scaleCol);

          if (rOrigin > inRowSize - 2)
            rOrigin = inRowSize - 2;
          if (cOrigin > inColSize - 2)
            cOrigin = inColSize - 2;

          double deltaR = i * scaleRow - rOrigin;
          double deltaC = j * scaleCol - cOrigin;
          double coeff1 = (1 - deltaR) * (1 - deltaC);
          double coeff2 = deltaR * (1 - deltaC);
          double coeff3 = (1 - deltaR) * deltaC;
          double coeff4 = deltaR * deltaC;

          size_t ptr = k * inRowSize * inColSize + cOrigin * inColSize +
              rOrigin;

          output(k * outRowSize * outColSize + j * outColSize + i) =
              input(ptr) * coeff1 +
              input(ptr + 1) * coeff2 +
              input(ptr + inColSize) * coeff3 +
              input(ptr + inColSize + 1) * coeff4;
        }
      }
    }
  }

  /**
   * DownSample the given input.
   *
   * @param input The input matrix
   * @param gradient The computed backward gradient
   * @param output The resulting down-sampled output image.
   */
  template<typename eT>
  void DownSample(const arma::Mat<eT>& /*input*/, 
      const arma::Mat<eT>& gradient, arma::Mat<eT>& output)
  {
    if (output.is_empty())
      output.set_size(inRowSize * inColSize * depth, 1);

    assert(output.n_rows == inRowSize * inColSize);
    assert(output.n_cols == 1);

    if (gradient.n_elem == output.n_elem)
    {
      output = gradient;
    }
    else
    {
      double scaleRow = (double)(outRowSize) / inRowSize;
      double scaleCol = (double)(outColSize) / inColSize;

      for (size_t k = 0; k < depth; k++)
        for (size_t i = 0; i < inRowSize; i++)
          for (size_t j = 0; j < inColSize; j++)
          {
            double rOrigin = std::floor(i * scaleRow);
            double cOrigin = std::floor(j * scaleCol);

            if (rOrigin > outRowSize - 2)
              rOrigin = outRowSize - 2;
            if (cOrigin > outColSize - 2)
              cOrigin = outColSize - 2;

            double deltaR = i * scaleRow - rOrigin;
            double deltaC = j * scaleCol - cOrigin;
            double coeff1 = (1 - deltaR) * (1 - deltaC);
            double coeff2 = deltaR * (1 - deltaC);
            double coeff3 = (1 - deltaR) * deltaC;
            double coeff4 = deltaR * deltaC;

            size_t ptr =  k * outRowSize * outColSize + cOrigin * outColSize +
                rOrigin;

            output(k * inColSize * inRowSize + j * inColSize + i) =
                gradient(ptr) * coeff1 +
                gradient(ptr + 1) * coeff2 +
                gradient(ptr + outColSize) * coeff3 +
                gradient(ptr + outColSize + 1) * coeff4;
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
    ar & data::CreateNVP(depth, "depth");
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
  //! Locally stored depth of the input.
  const size_t depth;
}; // class BiLinearFunction

} // namespace ann
} // namespace mlpack

#endif
