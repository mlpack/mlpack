/**
 * @file methods/ann/layer/bicubic_interpolation_impl.hpp
 * @author Abhinav Anand
 *
 * Implementation of the Bicubic interpolation function as an individual layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_BICUBIC_INTERPOLATION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_BICUBIC_INTERPOLATION_IMPL_HPP

// In case it hasn't yet been included.
#include "bicubic_interpolation.hpp"

namespace mlpack {


template<typename InputDataType, typename OutputDataType>
BicubicInterpolation<InputDataType, OutputDataType>::
BicubicInterpolation():
  inRowSize(0),
  inColSize(0),
  outRowSize(0),
  outColSize(0),
  depth(0),
  alpha(-0.75),
  batchSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
BicubicInterpolation<InputDataType, OutputDataType>::
BicubicInterpolation(const size_t inRowSize,
                     const size_t inColSize,
                     const size_t outRowSize,
                     const size_t outColSize,
                     const size_t depth,
                     const double alpha):
  inRowSize(inRowSize),
  inColSize(inColSize),
  outRowSize(outRowSize),
  outColSize(outColSize),
  depth(depth),
  alpha(alpha),
  batchSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
void BicubicInterpolation<InputDataType, OutputDataType>::GetKernalWeight(
    double delta,
    arma::mat& coeffs)
{
  coeffs(0) = ((alpha * (delta + 1) - 5 * alpha) * (delta + 1) + 8 * alpha) *
      (delta + 1) - 4 * alpha;
  coeffs(1) = ((alpha + 2) * delta - (alpha + 3)) * delta * delta + 1;
  coeffs(2) = ((alpha + 2) * (1 - delta) - (alpha + 3)) * (1 - delta) *
      (1 - delta) + 1;
  coeffs(3) = 1 - coeffs[0] - coeffs[1] - coeffs[2];
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BicubicInterpolation<InputDataType, OutputDataType>::Forward(
  const arma::Mat<eT>& input, arma::Mat<eT>& output)
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

  const double scaleRow = (double) inRowSize / (double) outRowSize;
  const double scaleCol = (double) inColSize / (double) outColSize;

  arma::cube inputAsCube(const_cast<arma::Mat<eT>&>(input).memptr(),
      inRowSize, inColSize, depth * batchSize, false, false);
  arma::cube outputAsCube(output.memptr(), outRowSize, outColSize,
      depth * batchSize, false, true);

  for (size_t k = 0; k < depth * batchSize; ++k)
  {
    // The input is padded on all sides using replication of the input boundary.
    arma::mat grid = zeros(inRowSize + 4, inColSize + 4);
    grid(arma::span(2, inRowSize + 1), arma::span(2, inColSize + 1)) =
        inputAsCube.slice(k);
    grid(arma::span(2, inRowSize + 1), 0) =
        grid(arma::span(2, inRowSize + 1), 2);
    grid(arma::span(2, inRowSize + 1), 1) =
        grid(arma::span(2, inRowSize + 1), 2);
    grid(arma::span(2, inRowSize + 1), inColSize + 2) =
        grid(arma::span(2, inRowSize + 1), inColSize + 1);
    grid(arma::span(2, inRowSize + 1), inColSize + 3) =
        grid(arma::span(2, inRowSize + 1), inColSize + 1);
    grid(0, arma::span(2, inColSize+ 1)) =
        grid(2, arma::span(2, inColSize + 1));
    grid(1, arma::span(2, inColSize + 1)) =
        grid(2, arma::span(2, inColSize + 1));
    grid(inRowSize + 2, arma::span(2, inColSize + 1)) =
        grid(inRowSize + 1, arma::span(2, inColSize + 1));
    grid(inRowSize + 3, arma::span(2, inColSize + 1)) =
        grid(inRowSize + 1, arma::span(2, inColSize + 1));
    grid(arma::span(0, 1), arma::span(0, 1)) += grid(2, 2);
    grid(arma::span(inRowSize + 2, inRowSize + 3),
         arma::span(inColSize + 2, inColSize + 3)) +=
        grid(inRowSize + 1, inColSize + 1);
    grid(arma::span(inRowSize + 2, inRowSize + 3), arma::span(0, 1)) +=
        grid(inRowSize + 1, 2);
    grid(arma::span(0, 1), arma::span(inColSize + 2, inColSize + 3)) +=
        grid(2, inColSize + 1);

    for (size_t i = 0; i < outRowSize; ++i)
    {
      double rOrigin = (i + 0.5) * scaleRow;
      for (size_t j = 0; j < outColSize; ++j)
      {
        arma::mat kernel = arma::mat(4, 4);
        double cOrigin = (j + 0.5) * scaleCol;
        // Bottom right corner of the kernel.
        const size_t cEnd = (size_t) std::floor(cOrigin + 1.5) + 2;
        const size_t rEnd = (size_t) std::floor(rOrigin + 1.5) + 2;
        kernel = grid(arma::span(rEnd - 3, rEnd),
            arma::span(cEnd - 3, cEnd));

        double fc = cOrigin - 0.5;
        fc = fc - std::floor(fc);
        double fr = rOrigin - 0.5;
        fr = fr - std::floor(fr);

        arma::mat weightR = arma::mat(4, 1);
        arma::mat weightC = arma::mat(1, 4);
        GetKernalWeight(fr, weightR);
        GetKernalWeight(fc, weightC);

        arma::mat weightMatrix = weightR * weightC;
        outputAsCube(i, j, k) = accu(weightMatrix % kernel);
      }
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BicubicInterpolation<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>& /*input*/,
  const arma::Mat<eT>& gradient,
  arma::Mat<eT>& output)
{
  if (output.is_empty())
    output.zeros(inRowSize * inColSize * depth, batchSize);
  else
  {
    assert(output.n_rows == inRowSize * inColSize * depth);
    assert(output.n_cols == batchSize);
  }

  assert(outRowSize >= 2);
  assert(outColSize >= 2);

  arma::cube gradientAsCube(((arma::Mat<eT>&) gradient).memptr(), outRowSize,
      outColSize, depth * batchSize, false, false);
  arma::cube outputAsCube(output.memptr(), inRowSize, inColSize,
      depth * batchSize, false, true);

  const double scaleRow = (double) inRowSize / (double) outRowSize;
  const double scaleCol = (double) inColSize / (double) outColSize;

  if (gradient.n_elem == output.n_elem)
  {
    outputAsCube = gradientAsCube;
  }
  else
  {
    for (size_t k = 0; k < depth * batchSize; ++k)
    {
      arma::mat temp = zeros(inRowSize + 4, inColSize + 4);
      for (size_t i = 0; i < outRowSize; ++i)
      {
        double rOrigin = (i + 0.5) * scaleRow;
        for (size_t j = 0; j < outColSize; ++j)
        {
          double cOrigin = (j + 0.5) * scaleCol;
          // Bottom right corner of the kernel.
          const size_t cEnd = (size_t) std::floor(cOrigin + 1.5) + 2;
          const size_t rEnd = (size_t) std::floor(rOrigin + 1.5) + 2;

          double fc = cOrigin - 0.5;
          fc = fc - std::floor(fc);
          double fr = rOrigin - 0.5;
          fr = fr - std::floor(fr);

          arma::mat weightR = arma::mat(4, 1);
          arma::mat weightC = arma::mat(1, 4);
          GetKernalWeight(fr, weightR);
          GetKernalWeight(fc, weightC);

          temp(arma::span(rEnd - 3, rEnd), arma::span(cEnd - 3, cEnd)) +=
              gradientAsCube(i, j, k) * weightR * weightC;
        }
      }
      // Adding the contribution of the corner points to the output matrix.
      temp.row(2) += temp.row(0);
      temp.row(2) += temp.row(1);
      temp.row(inRowSize + 1) += temp.row(inRowSize + 2);
      temp.row(inRowSize + 1) += temp.row(inRowSize + 3);
      temp.col(2) += temp.col(0);
      temp.col(2) += temp.col(1);
      temp.col(inColSize + 1) += temp.col(inColSize + 2);
      temp.col(inColSize + 1) += temp.col(inColSize + 3);

      outputAsCube.slice(k) += temp(arma::span(2, inRowSize + 1),
                                    arma::span(2, inColSize + 1));
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void BicubicInterpolation<InputDataType, OutputDataType>::serialize(
  Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(inRowSize));
  ar(CEREAL_NVP(inColSize));
  ar(CEREAL_NVP(outRowSize));
  ar(CEREAL_NVP(outColSize));
  ar(CEREAL_NVP(depth));
  ar(CEREAL_NVP(alpha));
}

} // namespace mlpack

#endif
