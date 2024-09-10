/**
 * @file methods/ann/layer/bilinear_interpolation_impl.hpp
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

template<typename InputType, typename OutputType>
BilinearInterpolationType<InputType, OutputType>::
BilinearInterpolationType():
    outRowSize(0),
    outColSize(0)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
BilinearInterpolationType<InputType, OutputType>::
BilinearInterpolationType(const size_t outRowSize,
                          const size_t outColSize) :
    outRowSize(outRowSize),
    outColSize(outColSize)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
void BilinearInterpolationType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  const size_t batchSize = input.n_cols;
  const size_t depth = this->inputDimensions.size() <= 2 ? 1 :
      std::accumulate(this->inputDimensions.begin() + 2,
                      this->inputDimensions.end(), 0);

  assert(output.n_rows == outRowSize * outColSize * depth);
  assert(output.n_cols == batchSize);

  assert(this->inputDimensions[0] >= 2);
  assert(this->inputDimensions[1] >= 2);

  arma::Cube<typename InputType::elem_type> inputAsCube(
      const_cast<InputType&>(input).memptr(), this->inputDimensions[0],
      this->inputDimensions[1], depth * batchSize, false, false);
  arma::Cube<typename OutputType::elem_type> outputAsCube(
      output.memptr(), outRowSize, outColSize, depth * batchSize, false, true);

  double scaleRow = (double) this->inputDimensions[0] / (double) outRowSize;
  double scaleCol = (double) this->inputDimensions[1] / (double) outColSize;

  arma::mat22 coeffs;
  for (size_t i = 0; i < outRowSize; ++i)
  {
    size_t rOrigin = (size_t) std::floor(i * scaleRow);
    if (rOrigin > this->inputDimensions[0] - 2)
      rOrigin = this->inputDimensions[0] - 2;

    // Scaled distance of the interpolated point from the topmost row.
    double deltaR = i * scaleRow - rOrigin;
    if (deltaR > 1)
      deltaR = 1.0;
    for (size_t j = 0; j < outColSize; ++j)
    {
      // Scaled distance of the interpolated point from the leftmost column.
      size_t cOrigin = (size_t) std::floor(j * scaleCol);
      if (cOrigin > this->inputDimensions[1] - 2)
        cOrigin = this->inputDimensions[1] - 2;

      double deltaC = j * scaleCol - cOrigin;
      if (deltaC > 1)
        deltaC = 1.0;
      coeffs[0] = (1 - deltaR) * (1 - deltaC);
      coeffs[1] = deltaR * (1 - deltaC);
      coeffs[2] = (1 - deltaR) * deltaC;
      coeffs[3] = deltaR * deltaC;

      for (size_t k = 0; k < depth * batchSize; ++k)
      {
        outputAsCube(i, j, k) = accu(inputAsCube.slice(k).submat(
            rOrigin, cOrigin, rOrigin + 1, cOrigin + 1) % coeffs);
      }
    }
  }
}

template<typename InputType, typename OutputType>
void BilinearInterpolationType<InputType, OutputType>::Backward(
    const InputType& /*input*/,
    const OutputType& gradient,
    OutputType& output)
{
  const size_t batchSize = output.n_cols;
  const size_t depth = this->inputDimensions.size() <= 2 ? 1 :
      std::accumulate(this->inputDimensions.begin() + 2,
                      this->inputDimensions.end(), 0);

  assert(output.n_rows == this->inputDimensions[0] *
      this->inputDimensions[1] * depth);

  assert(outRowSize >= 2);
  assert(outColSize >= 2);

  arma::Cube<typename OutputType::elem_type> gradientAsCube(
      ((OutputType&) gradient).memptr(), outRowSize, outColSize, depth *
      batchSize, false, false);
  arma::Cube<typename OutputType::elem_type> outputAsCube(
      output.memptr(), this->inputDimensions[0], this->inputDimensions[1],
      depth * batchSize, false, true);

  if (gradient.n_elem == output.n_elem)
  {
    outputAsCube = gradientAsCube;
  }
  else
  {
    double scaleRow = (double)(outRowSize) / this->inputDimensions[0];
    double scaleCol = (double)(outColSize) / this->inputDimensions[1];

    arma::mat22 coeffs;
    for (size_t i = 0; i < this->inputDimensions[0]; ++i)
    {
      size_t rOrigin = (size_t) std::floor(i * scaleRow);
      if (rOrigin > outRowSize - 2)
        rOrigin = outRowSize - 2;
      double deltaR = i * scaleRow - rOrigin;
      for (size_t j = 0; j < this->inputDimensions[1]; ++j)
      {
        size_t cOrigin = (size_t) std::floor(j * scaleCol);

        if (cOrigin > outColSize - 2)
          cOrigin = outColSize - 2;

        double deltaC = j * scaleCol - cOrigin;
        coeffs[0] = (1 - deltaR) * (1 - deltaC);
        coeffs[1] = deltaR * (1 - deltaC);
        coeffs[2] = (1 - deltaR) * deltaC;
        coeffs[3] = deltaR * deltaC;

        for (size_t k = 0; k < depth * batchSize; ++k)
        {
          outputAsCube(i, j, k) = accu(gradientAsCube.slice(k).submat(
              rOrigin, cOrigin, rOrigin + 1, cOrigin + 1) % coeffs);
        }
      }
    }
  }
}

template<typename InputType, typename OutputType>
template<typename Archive>
void BilinearInterpolationType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(outRowSize));
  ar(CEREAL_NVP(outColSize));
}

} // namespace mlpack

#endif
