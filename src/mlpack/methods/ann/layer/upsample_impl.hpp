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
      index_pre = arma::cube(1, 2, outRowSize * outColSize);

      for (size_t i = 0; i < outRowSize; i++)
      {
        double rOrigin = (i + 0.5) * scaleRow;

        for(size_t j = 0; j < outColSize; j++)
        {

          double cOrigin = (j + 0.5) * scaleCol;
          size_t c_end = (size_t) std::floor(cOrigin + 0.5);
          size_t r_end = (size_t) std::floor(rOrigin + 0.5);

          double deltaC = 0.5 + cOrigin - c_end;
          double deltaR = 0.5 + rOrigin - r_end; 
          if(deltaR >= 1)
            deltaR -= 1;

          if(deltaC >= 1)
            deltaC -= 1;


          size_t current_idx = j * outRowSize + i;
          coeffs_pre.slice(current_idx)(0, 0) = (1 - deltaC) * (1 - deltaR);
          coeffs_pre.slice(current_idx)(0, 1) = (1 - deltaR) * deltaC;
          coeffs_pre.slice(current_idx)(1, 0) = deltaR * (1 - deltaC);
          coeffs_pre.slice(current_idx)(1, 1) = deltaR * deltaC;
          index_pre.slice(current_idx)(0, 0) = c_end + 1;
          index_pre.slice(current_idx)(0, 1) = r_end + 1;
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

    if(mode == "nearest")
    {

      for (size_t i = 0; i < outRowSize; ++i)
      {
        double rOrigin = (i + 0.5) * scaleRow;

        for (size_t j = 0; j < outColSize; ++j)
        {
          double cOrigin = (j + 0.5) * scaleCol;

          for (size_t k = 0; k < depth * batchSize; ++k)
          {
            outputAsCube(i, j, k) = inputAsCube.slice(k)(
              std::floor(cOrigin), std::floor(rOrigin));
          }
        }
      }
    }
    else if(mode == "bilinear")
    {
      for (size_t k = 0; k < depth * batchSize; ++k)
      {
        arma::mat grid = arma::mat(inRowSize + 2, inColSize + 2);
        grid(span(1, inRowSize), span(1, inColSize)) = inputAsCube.slice(k);
        grid(span(1, inRowSize), 0) = grid(span(1, inRowSize), 1);
        grid(span(1, inRowSize), inColSize + 1) = grid(span(1, inRowSize), inColSize);
        grid(0, span(1, inColSize)) = grid(1, span(1, inColSize));
        grid(inRowSize + 1, span(1, inColSize)) = grid(inRowSize, span(1, inColSize));
        grid(0, 0) = grid(1, 1);
        grid(inRowSize + 1, inColSize + 1) = grid(inRowSize, inColSize);
        grid(0, inColSize + 1) = grid(1, inColSize);
        grid(inRowSize + 1, 0) = grid(inRowSize, 1);

        for (size_t i = 0; i < outRowSize; ++i)
        {
          for (size_t j = 0; j < outColSize; ++j)
          {
            size_t current_idx = j * outRowSize + i;
            size_t c_right = index_pre.slice(current_idx)(0, 0);
            size_t r_down = index_pre.slice(current_idx)(0, 1);

            outputAsCube(i, j, k) = arma::accu(grid(span(r_down - 1, r_down),
              span(c_right - 1, c_right)) % coeffs_pre.slice(current_idx));
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
    {
      if (mode == "nearest")
        output.set_size(inRowSize * inColSize * depth, batchSize);
      else if (mode == "bilinear")
        output.set_size((inRowSize + 2) * (inColSize + 2) * depth, batchSize);
    }
    else
    {
      assert(output.n_rows == inRowSize * inColSize * depth);
      assert(output.n_cols == batchSize);
    }

    assert(outRowSize >= 2);
    assert(outColSize >= 2);
    arma::cube outputAsCube;
    arma::cube gradientAsCube(((arma::Mat<eT>&) gradient).memptr(), outRowSize,
      outColSize, depth * batchSize, false, false);
    if (mode == "nearest")
      outputAsCube(output.memptr(), inRowSize, inColSize,
        depth * batchSize, false, true);
    else if (mode == "bilinear")
      outputAsCube(output.memptr(), inRowSize + 2, inColSize + 2,
        depth * batchSize, false, true);

    double scaleRow = (double)(inRowSize) / outRowSize;
    double scaleCol = (double)(inColSize) / outColSize;

    if (gradient.n_elem == output.n_elem)
    {
      outputAsCube = gradientAsCube;
    }
    else
    {
      if(mode == "nearest")
      {      
        for (size_t i = 0; i < outRowSize; ++i)
        {
          double rOrigin = (i + 0.5) * scaleRow;

          for (size_t j = 0; j < outColSize; ++j)
          {
            double cOrigin = (j + 0.5) * scaleCol;

            for (size_t k = 0; k < depth * batchSize; ++k)
            {
              outputAsCube(std::floor(rOrigin), std::floor(cOrigin), k) += gradientAsCube(i, j, k);
            }
          }
        }
      }
      else if(mode == "bilinear")
      {
        output.zeros(); 
        for (size_t i = 0; i < outRowSize; ++i)
        {
          for (size_t j = 0; j < outColSize; ++j)
          {
            size_t current_idx = i * outRowSize + j;

            for (size_t k = 0; k < depth * batchSize; ++k)
            { 
             size_t current_idx = j * outRowSize + i;
             size_t c_right = index_pre.slice(current_idx)(0, 0);
             size_t r_down = index_pre.slice(current_idx)(0, 1);

             outputAsCube1.slice(k)(span(r_down - 1, r_down), span(c_right - 1, c_right)) +=
              coeffs_pre.slice(current_idx) * gradientAsCube.slice(k)(i, j) ;  }
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
