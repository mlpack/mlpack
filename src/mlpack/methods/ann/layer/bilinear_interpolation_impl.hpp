/**
 * @file methods/ann/layer/bilinear_interpolation_impl.hpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
 * @author Abhinav Anand
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
    // Let Po denote any output matrix point. This value at this
    // point will be determined by four points in the padded input
    // matrix. Let these be denoted by [P1, P2, P3, P4]. This will
    // form a unit square. So if we know the bottom right corner then
    // we can determine the other points.
    // Po = P1*d1 + P2*d2 + P3*d3 + P4*d4 where di's is a constant.
    // The di's and Pi's depend on the dimensions of input matrix
    // and output matrix. So, we can precompute these points in the
    // constructor and use them directly in the forward and backward
    // functions. In bilinear interpolation we will pad the input
    // matrix to make the calculations more simple.
    weights.set_size(WeightSize(), 1);
    double scaleRow = (double) inRowSize / (double) outRowSize;
    double scaleCol = (double) inColSize / (double) outColSize;
    coeffsPre = arma::cube(weights.memptr(), 2, 2,
      outRowSize * outColSize, false, false);
    indexPre = arma::cube(weights.memptr() + 4 * outRowSize * outColSize,
      1, 2, outRowSize * outColSize, false, false);
    temp = arma::mat(weights.memptr() + 6 * outRowSize * outColSize,
      inRowSize + 2, inColSize + 2, false, false);

    for (size_t i = 0; i < outRowSize; ++i)
    {
      double rOrigin = (i + 0.5) * scaleRow;

      for(size_t j = 0; j < outColSize; ++j)
      {

        double cOrigin = (j + 0.5) * scaleCol;
        size_t cEnd = (size_t) std::floor(cOrigin + 0.5);
        size_t rEnd = (size_t) std::floor(rOrigin + 0.5);

        double deltaC = 0.5 + cOrigin - cEnd;
        double deltaR = 0.5 + rOrigin - rEnd; 
        if (deltaR >= 1)
          deltaR -= 1;

        if (deltaC >= 1)
          deltaC -= 1;

        size_t currentIdx = j * outRowSize + i;
        // Value of di's for each point in Output matrix.
        coeffsPre.slice(currentIdx)(0, 0) = (1 - deltaC) * (1 - deltaR);
        coeffsPre.slice(currentIdx)(0, 1) = (1 - deltaR) * deltaC;
        coeffsPre.slice(currentIdx)(1, 0) = deltaR * (1 - deltaC);
        coeffsPre.slice(currentIdx)(1, 1) = deltaR * deltaC;
        // Bottom Right corner of the Pi's for each point in Output matrix.
        indexPre.slice(currentIdx)(0, 0) = cEnd + 1;
        indexPre.slice(currentIdx)(0, 1) = rEnd + 1;
      }
    }
  }

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BilinearInterpolation<InputDataType, OutputDataType>::Forward(
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

    arma::cube inputAsCube(const_cast<arma::Mat<eT>&>(input).memptr(),
      inRowSize, inColSize, depth * batchSize, false, false);
    arma::cube outputAsCube(output.memptr(), outRowSize, outColSize,
      depth * batchSize, false, true);

    for (size_t k = 0; k < depth * batchSize; ++k)
    {
      // The input is padded on all sides using replication of the input boundary.
      arma::mat grid = arma::mat(inRowSize + 2, inColSize + 2);
      grid(arma::span(1, inRowSize), arma::span(1, inColSize)) = inputAsCube.slice(k);
      grid(arma::span(1, inRowSize), 0) = grid(arma::span(1, inRowSize), 1);
      grid(arma::span(1, inRowSize), inColSize + 1) = grid(arma::span(1, inRowSize), inColSize);
      grid(0, arma::span(1, inColSize)) = grid(1, arma::span(1, inColSize));
      grid(inRowSize + 1, arma::span(1, inColSize)) = grid(inRowSize, arma::span(1, inColSize));
      grid(0, 0) = grid(1, 1);
      grid(inRowSize + 1, inColSize + 1) = grid(inRowSize, inColSize);
      grid(0, inColSize + 1) = grid(1, inColSize);
      grid(inRowSize + 1, 0) = grid(inRowSize, 1);

      for (size_t i = 0; i < outRowSize; ++i)
      {
        for (size_t j = 0; j < outColSize; ++j)
        {
          size_t currentIdx = j * outRowSize + i;
          size_t cEnd = indexPre.slice(currentIdx)(0, 0);
          size_t rEnd = indexPre.slice(currentIdx)(0, 1);

          outputAsCube(i, j, k) = arma::accu(grid(arma::span(rEnd - 1, rEnd),
            arma::span(cEnd - 1, cEnd)) % coeffsPre.slice(currentIdx));
        }
      }
    }
  }

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BilinearInterpolation<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>& /*input*/,
  const arma::Mat<eT>& gradient,
  arma::Mat<eT>& output)
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

    arma::cube gradientAsCube(((arma::Mat<eT>&) gradient).memptr(), outRowSize,
      outColSize, depth * batchSize, false, false);
    arma::cube outputAsCube(output.memptr(), inRowSize, inColSize,
      depth * batchSize, false, true);

    if (gradient.n_elem == output.n_elem)
    {
      outputAsCube = gradientAsCube;
    }
    else
    {
      for (size_t k = 0; k < depth * batchSize; ++k)
      {
        temp.zeros();
        for (size_t i = 0; i < outRowSize; ++i)
        {
          for (size_t j = 0; j < outColSize; ++j)
          {
            size_t currentIdx = j * outRowSize + i;
            // Bottom right corner of the mapped unit matrix.
            size_t cEnd = indexPre.slice(currentIdx)(0, 0);
            size_t rEnd = indexPre.slice(currentIdx)(0, 1);

            temp(arma::span(rEnd - 1, rEnd), arma::span(cEnd - 1, cEnd)) +=
            coeffsPre.slice(currentIdx) * gradientAsCube(i, j, k) ;
          }
        }
        // Adding the contribution of the corner points to the output matrix.
        temp.row(1) += temp.row(0);
        temp.row(inRowSize) += temp.row(inRowSize + 1);
        temp.col(1) += temp.col(0);
        temp.col(inColSize) += temp.col(inColSize + 1);
        outputAsCube.slice(k) += temp(arma::span(1, inRowSize), arma::span(1, inColSize));
      }
    }
  }

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void BilinearInterpolation<InputDataType, OutputDataType>::serialize(
  Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(inRowSize));
    ar(CEREAL_NVP(inColSize));
    ar(CEREAL_NVP(outRowSize));
    ar(CEREAL_NVP(outColSize));
    ar(CEREAL_NVP(depth));
  }

} // namespace ann
} // namespace mlpack

#endif
