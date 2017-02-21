/**
 * @file glimpse_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the GlimpseLayer class, which takes an input image and a
 * location to extract a retina-like representation of the input image at
 * different increasing scales.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_GLIMPSE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_GLIMPSE_IMPL_HPP

// In case it hasn't yet been included.
#include "glimpse.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
Glimpse<InputDataType, OutputDataType>::Glimpse(
    const size_t inSize,
    const size_t size,
    const size_t depth,
    const size_t scale,
    const size_t inputWidth,
    const size_t inputHeight) :
    inSize(inSize),
    size(size),
    depth(depth),
    scale(scale),
    inputWidth(inputWidth),
    inputHeight(inputHeight)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Glimpse<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  inputTemp = arma::cube(input.colptr(0), inputWidth, inputHeight, inSize);
  outputTemp = arma::Cube<eT>(size, size, depth * inputTemp.n_slices);

  location = input.submat(0, 1, 1, 1);

  if (!deterministic)
  {
    locationParameter.push_back(location);
  }

  inputDepth = inputTemp.n_slices / inSize;

  for (size_t inputIdx = 0; inputIdx < inSize; inputIdx++)
  {
    for (size_t depthIdx = 0, glimpseSize = size;
        depthIdx < depth; depthIdx++, glimpseSize *= scale)
    {
      size_t padSize = std::floor((glimpseSize - 1) / 2);

      arma::Cube<eT> inputPadded = arma::zeros<arma::Cube<eT> >(
          inputTemp.n_rows + padSize * 2, inputTemp.n_cols + padSize * 2,
          inputTemp.n_slices / inSize);

      inputPadded.tube(padSize, padSize, padSize + inputTemp.n_rows - 1,
          padSize + inputTemp.n_cols - 1) = inputTemp.subcube(0, 0,
          inputIdx * inputDepth, inputTemp.n_rows - 1, inputTemp.n_cols - 1,
          (inputIdx + 1) * inputDepth - 1);

      size_t h = inputPadded.n_rows - glimpseSize;
      size_t w = inputPadded.n_cols - glimpseSize;

      size_t x = std::min(h, (size_t) std::max(0.0,
          (location(0, inputIdx) + 1) / 2.0 * h));
      size_t y = std::min(w, (size_t) std::max(0.0,
          (location(1, inputIdx) + 1) / 2.0 * w));

      if (depthIdx == 0)
      {
        for (size_t j = (inputIdx + depthIdx), paddedSlice = 0;
            j < outputTemp.n_slices; j += (inSize * depth), paddedSlice++)
        {
          outputTemp.slice(j) = inputPadded.subcube(x, y,
              paddedSlice, x + glimpseSize - 1, y + glimpseSize - 1,
              paddedSlice);
        }
      }
      else
      {
        for (size_t j = (inputIdx + depthIdx * (depth - 1)), paddedSlice = 0;
            j < outputTemp.n_slices; j += (inSize * depth), paddedSlice++)
        {
          arma::Mat<eT> poolingInput = inputPadded.subcube(x, y,
              paddedSlice, x + glimpseSize - 1, y + glimpseSize - 1,
              paddedSlice);

          if (scale == 2)
          {
            Pooling(glimpseSize / size, poolingInput, outputTemp.slice(j));
          }
          else
          {
            ReSampling(poolingInput, outputTemp.slice(j));
          }
        }
      }
    }
  }

  for (size_t i = 0; i < outputTemp.n_slices; ++i)
  {
    outputTemp.slice(i) = arma::trans(outputTemp.slice(i));
  }

  output = arma::Mat<eT>(outputTemp.memptr(), outputTemp.n_elem, 1);

  outputWidth = outputTemp.n_rows;
  outputHeight = outputTemp.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Glimpse<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  // Generate a cube using the backpropagated error matrix.
  arma::Cube<eT> mappedError = arma::zeros<arma::cube>(outputWidth,
      outputHeight, 1);

  location = locationParameter.back();
  locationParameter.pop_back();

  for (size_t s = 0, j = 0; s < mappedError.n_slices; s+= gy.n_cols, j++)
  {
    for (size_t i = 0; i < gy.n_cols; i++)
    {
      mappedError.slice(s + i) = arma::Mat<eT>(gy.memptr(),
          outputWidth, outputHeight);
    }
  }

  gTemp = arma::zeros<arma::cube>(inputTemp.n_rows, inputTemp.n_cols,
      inputTemp.n_slices);

  for (size_t inputIdx = 0; inputIdx < inSize; inputIdx++)
  {
    for (size_t depthIdx = 0, glimpseSize = size;
        depthIdx < depth; depthIdx++, glimpseSize *= scale)
    {
      size_t padSize = std::floor((glimpseSize - 1) / 2);

      arma::Cube<eT> inputPadded = arma::zeros<arma::Cube<eT> >(
          inputTemp.n_rows + padSize * 2, inputTemp.n_cols +
          padSize * 2, inputTemp.n_slices / inSize);

      size_t h = inputPadded.n_rows - glimpseSize;
      size_t w = inputPadded.n_cols - glimpseSize;

      size_t x = std::min(h, (size_t) std::max(0.0,
          (location(0, inputIdx) + 1) / 2.0 * h));
      size_t y = std::min(w, (size_t) std::max(0.0,
          (location(1, inputIdx) + 1) / 2.0 * w));

      if (depthIdx == 0)
      {
        for (size_t j = (inputIdx + depthIdx), paddedSlice = 0;
            j < mappedError.n_slices; j += (inSize * depth), paddedSlice++)
        {
          inputPadded.subcube(x, y,
              paddedSlice, x + glimpseSize - 1, y + glimpseSize - 1,
              paddedSlice) = mappedError.slice(j);
        }
      }
      else
      {
        for (size_t j = (inputIdx + depthIdx * (depth - 1)), paddedSlice = 0;
            j < mappedError.n_slices; j += (inSize * depth), paddedSlice++)
        {
          arma::Mat<eT> poolingOutput = inputPadded.subcube(x, y,
               paddedSlice, x + glimpseSize - 1, y + glimpseSize - 1,
               paddedSlice);

          if (scale == 2)
          {
            Unpooling(inputTemp.slice(paddedSlice), mappedError.slice(j),
                poolingOutput);
          }
          else
          {
            DownwardReSampling(inputTemp.slice(paddedSlice),
                mappedError.slice(j), poolingOutput);
          }

          inputPadded.subcube(x, y,
              paddedSlice, x + glimpseSize - 1, y + glimpseSize - 1,
              paddedSlice) = poolingOutput;
        }
      }

      gTemp += inputPadded.tube(padSize, padSize, padSize +
          inputTemp.n_rows - 1, padSize + inputTemp.n_cols - 1);
    }
  }

  Transform(gTemp);
  g = arma::mat(gTemp.memptr(), gTemp.n_elem, 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Glimpse<InputDataType, OutputDataType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(inSize, "inSize");
  ar & data::CreateNVP(size, "size");
  ar & data::CreateNVP(depth, "depth");
  ar & data::CreateNVP(scale, "scale");
  ar & data::CreateNVP(inputWidth, "inputWidth");
  ar & data::CreateNVP(location, "location");
}

} // namespace ann
} // namespace mlpack

#endif
