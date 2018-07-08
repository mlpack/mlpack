/**
 * @file bilinear_interpolation_impl.hpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
 *
 * Implementation of the bilinear interpolation function as an individual layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_BILINEAR_INTERPOLATION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_BILINEAR_INTERPOLATION_IMPL_HPP

// In case it hasn't yet been included.
#include "bilinear_interpolation.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


template<typename InputDataType, typename OutputDataType>
BilinearInterpolation<InputDataType, OutputDataType>::
BilinearInterpolation():
    inRowSize(0),
    inColSize(0),
    outRowSize(0),
    outColSize(0),
    depth(0),
    batchSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
BilinearInterpolation<InputDataType, OutputDataType>::
BilinearInterpolation(
    const size_t inRowSize,
    const size_t inColSize,
    const size_t outRowSize,
    const size_t outColSize,
    const size_t depth):
    inRowSize(inRowSize),
    inColSize(inColSize),
    outRowSize(outRowSize),
    outColSize(outColSize),
    depth(depth),
    batchSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BilinearInterpolation<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  batchSize = input.n_cols;
  if (output.is_empty())
    output.set_size(outRowSize * outColSize * depth, batchSize);
  else
  {
    assert(output.n_rows == outRowSize * outColSize * depth);
    assert(output.n_cols == batchSize);
  }

  assert(inRowSize >= 2);
  assert(inColSize >= 2);

  arma::cube inputAsCube(const_cast<arma::Mat<eT>&&>(input).memptr(),
      inRowSize, inColSize, depth * batchSize, false, false);
  arma::cube outputAsCube(output.memptr(), outRowSize, outColSize,
                          depth * batchSize, false, true);

  double scaleRow = (double) inRowSize / (double) outRowSize;
  double scaleCol = (double) inColSize / (double) outColSize;

  arma::mat22 coeffs;
  for (size_t i = 0; i < outRowSize; i++)
  {
    size_t rOrigin = (size_t) std::floor(i * scaleRow);
    if (rOrigin > inRowSize - 2)
      rOrigin = inRowSize - 2;

    // Scaled distance of the interpolated point from the topmost row.
    double deltaR = i * scaleRow - rOrigin;
    if (deltaR > 1)
      deltaR = 1.0;
    for (size_t j = 0; j < outColSize; j++)
    {
      // Scaled distance of the interpolated point from the leftmost column.
      size_t cOrigin = (size_t) std::floor(j * scaleCol);
      if (cOrigin > inColSize - 2)
        cOrigin = inColSize - 2;

      double deltaC = j * scaleCol - cOrigin;
      if (deltaC > 1)
        deltaC = 1.0;
      coeffs[0] = (1 - deltaR) * (1 - deltaC);
      coeffs[1] = deltaR * (1 - deltaC);
      coeffs[2] = (1 - deltaR) * deltaC;
      coeffs[3] = deltaR * deltaC;

      for (size_t k = 0; k < depth * batchSize; k++)
      {
        outputAsCube(i, j, k) = arma::accu(inputAsCube.slice(k).submat(
            rOrigin, cOrigin, rOrigin + 1, cOrigin + 1) % coeffs);
      }
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BilinearInterpolation<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /*input*/,
    arma::Mat<eT>&& gradient,
    arma::Mat<eT>&& output)
{
  if (output.is_empty())
    output.set_size(inRowSize * inColSize * depth, batchSize);
  else
  {
    assert(output.n_rows == inRowSize * inColSize * depth);
    assert(output.n_cols == batchSize);
  }

  assert(outRowSize >= 2);
  assert(outColSize >= 2);

  arma::cube gradientAsCube(gradient.memptr(), outRowSize, outColSize,
                            depth * batchSize, false, false);
  arma::cube outputAsCube(output.memptr(), inRowSize, inColSize,
                          depth * batchSize, false, true);

  if (gradient.n_elem == output.n_elem)
  {
    outputAsCube = gradientAsCube;
  }
  else
  {
    double scaleRow = (double)(outRowSize) / inRowSize;
    double scaleCol = (double)(outColSize) / inColSize;

    arma::mat22 coeffs;
    for (size_t i = 0; i < inRowSize; i++)
    {
      size_t rOrigin = (size_t) std::floor(i * scaleRow);
      if (rOrigin > outRowSize - 2)
        rOrigin = outRowSize - 2;
      double deltaR = i * scaleRow - rOrigin;
      for (size_t j = 0; j < inColSize; j++)
      {
        size_t cOrigin = (size_t) std::floor(j * scaleCol);

        if (cOrigin > outColSize - 2)
          cOrigin = outColSize - 2;

        double deltaC = j * scaleCol - cOrigin;
        coeffs[0] = (1 - deltaR) * (1 - deltaC);
        coeffs[1] = deltaR * (1 - deltaC);
        coeffs[2] = (1 - deltaR) * deltaC;
        coeffs[3] = deltaR * deltaC;

        for (size_t k = 0; k < depth * batchSize; k++)
        {
          outputAsCube(i, j, k) = arma::accu(gradientAsCube.slice(k).submat(
              rOrigin, cOrigin, rOrigin + 1, cOrigin + 1) % coeffs);
        }
      }
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void BilinearInterpolation<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(inRowSize);
  ar & BOOST_SERIALIZATION_NVP(inColSize);
  ar & BOOST_SERIALIZATION_NVP(outRowSize);
  ar & BOOST_SERIALIZATION_NVP(outColSize);
  ar & BOOST_SERIALIZATION_NVP(depth);
}

} // namespace ann
} // namespace mlpack

#endif
