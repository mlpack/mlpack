/**
 * @file methods/ann/layer/upsample_impl.hpp
 * @author Abhinav Anand
 *
 * Implementation of the Upsample layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_UPSAMPLE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_UPSAMPLE_IMPL_HPP

// In case it hasn't yet been included.
#include "upsample.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


template<typename InputDataType, typename OutputDataType>
Upsample<InputDataType, OutputDataType>::
Upsample():
    inRowSize(0),
    inColSize(0),
    outRowSize(0),
    outColSize(0),
    depth(0),
    mode("nearest"),
    batchSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
Upsample<InputDataType, OutputDataType>::
Upsample(
    const size_t inRowSize,
    const size_t inColSize,
    const size_t outRowSize,
    const size_t outColSize,
    const size_t depth,
    const string mode):
    inRowSize(inRowSize),
    inColSize(inColSize),
    outRowSize(outRowSize),
    outColSize(outColSize),
    depth(depth),
    mode(mode),
    batchSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
void Upsample<InputDataType, OutputDataType>::Reset()
{
  if(mode == "bilinear")
  {
    double scaleRow = (double) inRowSize / (double) outRowSize;
    double scaleCol = (double) inColSize / (double) outColSize;
    coeffs_pre = arma::cube(2, 2, outRowSize * outColSize);
    index_pre = arma::cube(2, 2, outRowSize * outColSize);

    for (size_t i = 0; i < outRowSize; i++)
    {
      double yOrigin = (i + 0.5) * scaleRow;

      for(size_t j = 0; j < outColSize; j++)
      {

        double xOrigin = (j + 0.5) * scaleCol;
        size_t x_right = (size_t) std::floor(xOrigin + 0.5);
        size_t y_down = (size_t) std::floor(yOrigin + 0.5);
        size_t x_left = x_right;
        size_t y_up = y_down; 
        double deltaR = 0.5 + xOrigin - x_right;
        double deltaC = 0.5 + yOrigin - y_down; 
        if(deltaR >= 1)
          deltaR -= 1;
        
        if(deltaC >= 1)
        deltaC -= 1;

        if(x_right >= inColSize)
          x_right = inColSize - 1;
        if(y_down >= inRowSize)
          y_down = inRowSize - 1;
        
        if(x_left > 0)
          x_left -= 1;
        if(y_up > 0)
          y_up -= 1;

        size_t current_idx = j * outRowSize + i;
        coeffs_pre.slice(current_idx)(0, 0) = (1 - deltaR) * (1 - deltaC);
        coeffs_pre.slice(current_idx)(1, 0) = (1 - deltaR) * deltaC;
        coeffs_pre.slice(current_idx)(0, 1) = deltaR * (1 - deltaC);
        coeffs_pre.slice(current_idx)(1, 1) = deltaR * deltaC;
        index_pre.slice(current_idx)(0, 0) = x_left * inRowSize + y_up;
        index_pre.slice(current_idx)(1, 0) = x_left * inRowSize + y_down;
        index_pre.slice(current_idx)(0, 1) = x_right * inRowSize + y_up;
        index_pre.slice(current_idx)(1, 1) = x_right * inRowSize + y_down;
      }
    }
  }
}


template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Upsample<InputDataType, OutputDataType>::Forward(
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

  double scaleRow = (double) inRowSize / (double) outRowSize;
  double scaleCol = (double) inColSize / (double) outColSize;

  arma::mat22 vals;
  for (size_t i = 0; i < outRowSize; ++i)
  {
    double yOrigin = (i + 0.5) * scaleRow;

    for (size_t j = 0; j < outColSize; ++j)
    {
      double xOrigin = (j + 0.5) * scaleCol;

      if(mode == "nearest"){

        for (size_t k = 0; k < depth * batchSize; ++k)
        {
          outputAsCube(i, j, k) = inputAsCube.slice(k)(
            std::floor(xOrigin), std::floor(yOrigin));
        }
      }

      else if(mode == "bilinear"){
        size_t current_idx = i * outRowSize + j;

        for (size_t k = 0; k < depth * batchSize; ++k)
        { 
          arma::idx = index_pre.slice(current_idx);
          vals[0] = inputAsCube.slice(k)(idx(0));
          vals[1] = inputAsCube.slice(k)(idx(1));
          vals[2] = inputAsCube.slice(k)(idx(2));
          vals[3] = inputAsCube.slice(k)(idx(3));

          outputAsCube(i, j, k) = arma::accu(vals % coeffs_pre.slice(current_idx));
        }
      }
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Upsample<InputDataType, OutputDataType>::Backward(
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
    double scaleRow = (double)(outRowSize) / inRowSize;
    double scaleCol = (double)(outColSize) / inColSize;

    arma::mat22 coeffs;
    for (size_t i = 0; i < inRowSize; ++i)
    {
      double yOrigin = (i + 0.5) * scaleRow;

      for (size_t j = 0; j < inColSize; ++j)
      {
        double xOrigin = (j + 0.5) * scaleCol;
        if(mode == "nearest")
        {

          for (size_t k = 0; k < depth * batchSize; ++k)
          {
            outputAsCube(i, j, k) = gradientAsCube.slice(k)(
              std::floor(xOrigin), std::floor(yOrigin));
          }
        }
        else if(mode == "bilinear")
        {
          size_t current_idx = j * outRowSize + i;
          for (size_t k = 0; k < depth * batchSize; ++k)
          { 
            arma::mat idx = index_pre.slice(current_idx);
            vals[0] = gradientAsCube.slice(k)(idx(0));
            vals[1] = gradientAsCube.slice(k)(idx(1));
            vals[2] = gradientAsCube.slice(k)(idx(2));
            vals[3] = gradientAsCube.slice(k)(idx(3)); 
        
           outputAsCube(i, j, k) = arma::accu(vals % coeffs_pre.slice(current_idx));

          }
        }
      }
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Upsample<InputDataType, OutputDataType>::serialize(
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
