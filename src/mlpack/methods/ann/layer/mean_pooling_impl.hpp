/**
 * @file methods/ann/layer/mean_pooling_impl.hpp
 * @author Marcus Edel
 * @author Nilay Jain
 * @author Shubham Agrawal
 *
 * Implementation of the MeanPooling layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MEAN_POOLING_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MEAN_POOLING_IMPL_HPP

// In case it hasn't yet been included.
#include "mean_pooling.hpp"

namespace mlpack {

template<typename MatType>
MeanPoolingType<MatType>::MeanPoolingType() :
    Layer<MatType>()
{
  // Nothing to do here.
}

template<typename MatType>
MeanPoolingType<MatType>::MeanPoolingType(
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const bool floor) :
    Layer<MatType>(),
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    floor(floor),
    channels(0)
{
  // Nothing to do here.
}

template<typename MatType>
MeanPoolingType<MatType>::MeanPoolingType(
    const MeanPoolingType& other) :
    Layer<MatType>(other),
    kernelWidth(other.kernelWidth),
    kernelHeight(other.kernelHeight),
    strideWidth(other.strideWidth),
    strideHeight(other.strideHeight),
    floor(other.floor),
    channels(other.channels)
{
  // Nothing to do here.
}

template<typename MatType>
MeanPoolingType<MatType>::MeanPoolingType(
    MeanPoolingType&& other) :
    Layer<MatType>(std::move(other)),
    kernelWidth(std::move(other.kernelWidth)),
    kernelHeight(std::move(other.kernelHeight)),
    strideWidth(std::move(other.strideWidth)),
    strideHeight(std::move(other.strideHeight)),
    floor(std::move(other.floor)),
    channels(std::move(other.channels))
{
  // Nothing to do here.
}

template<typename MatType>
MeanPoolingType<MatType>&
MeanPoolingType<MatType>::operator=(const MeanPoolingType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    kernelWidth = other.kernelWidth;
    kernelHeight = other.kernelHeight;
    strideWidth = other.strideWidth;
    strideHeight = other.strideHeight;
    floor = other.floor;
    channels = other.channels;
  }

  return *this;
}

template<typename MatType>
MeanPoolingType<MatType>&
MeanPoolingType<MatType>::operator=(MeanPoolingType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    kernelWidth = std::move(other.kernelWidth);
    kernelHeight = std::move(other.kernelHeight);
    strideWidth = std::move(other.strideWidth);
    strideHeight = std::move(other.strideHeight);
    floor = std::move(other.floor);
    channels = std::move(other.channels);
  }

  return *this;
}

template<typename MatType>
void MeanPoolingType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  // Create Alias of input as 2D image as input is 1D vector.
  arma::Cube<typename MatType::elem_type> inputTemp(
      const_cast<MatType&>(input).memptr(), this->inputDimensions[0],
      this->inputDimensions[1], input.n_cols * channels, false, false);

  // Create Alias of output as 2D image as output is 1D vector.
  arma::Cube<typename MatType::elem_type> outputTemp(output.memptr(),
      this->outputDimensions[0], this->outputDimensions[1],
      input.n_cols * channels, false, true);

  // Apply Pooling to the input.
  PoolingOperation(inputTemp, outputTemp);
}

template<typename MatType>
void MeanPoolingType<MatType>::Backward(
  const MatType& input,
  const MatType& /* output */,
  const MatType& gy,
  MatType& g)
{
  // Create Alias of gy as 2D matrix as gy is 1D vector.
  arma::Cube<typename MatType::elem_type> mappedError =
      arma::Cube<typename MatType::elem_type>(((MatType&) gy).memptr(),
      this->outputDimensions[0], this->outputDimensions[1],
      channels * input.n_cols, false, false);

  // Create Alias of g as 2D matrix as g is 1D vector.
  arma::Cube<typename MatType::elem_type> gTemp(g.memptr(),
      this->inputDimensions[0], this->inputDimensions[1],
      channels * input.n_cols, false, true);

  // Initialize the gradient with zero.
  gTemp.zeros();
  #pragma omp parallel for
  for (size_t s = 0; s < (size_t) mappedError.n_slices; s++)
  {
    // Computing gradient of each slice.
    Unpooling(mappedError.slice(s), gTemp.slice(s));
  }
}

template<typename MatType>
void MeanPoolingType<MatType>::ComputeOutputDimensions()
{
  this->outputDimensions = this->inputDimensions;

  // Compute the size of the output.
  if (floor)
  {
    this->outputDimensions[0] = std::floor((this->inputDimensions[0] -
        (double) kernelWidth) / (double) strideWidth + 1);
    this->outputDimensions[1] = std::floor((this->inputDimensions[1] -
        (double) kernelHeight) / (double) strideHeight + 1);
  }
  else
  {
    this->outputDimensions[0] = std::ceil((this->inputDimensions[0] -
        (double) kernelWidth) / (double) strideWidth + 1);
    this->outputDimensions[1] = std::ceil((this->inputDimensions[1] -
        (double) kernelHeight) / (double) strideHeight + 1);
  }

  // Higher dimensions are not modified.

  // Cache higher dimension points.
  channels = 1;
  for (size_t i = 2; i < this->inputDimensions.size(); ++i)
    channels *= this->inputDimensions[i];
}

template<typename MatType>
template<typename Archive>
void MeanPoolingType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(kernelWidth));
  ar(CEREAL_NVP(kernelHeight));
  ar(CEREAL_NVP(strideWidth));
  ar(CEREAL_NVP(strideHeight));
  ar(CEREAL_NVP(channels));
  ar(CEREAL_NVP(floor));
}

template<typename MatType>
void MeanPoolingType<MatType>::PoolingOperation(
    const arma::Cube<typename MatType::elem_type>& input,
    arma::Cube<typename MatType::elem_type>& output)
{
  // Iterate over all slices individually.
  #pragma omp parallel for
  for (size_t s = 0; s < (size_t) input.n_slices; ++s)
  {
    for (size_t j = 0, colidx = 0; j < output.n_cols;
        ++j, colidx += strideHeight)
    {
      size_t colEnd = colidx + kernelHeight - 1;
      // Check if the kernel along column is out of bounds.
      if (colEnd > input.n_cols - 1)
      {
        // If so, we need to reduce the kernel height or terminate.
        if (floor)
          continue;
        colEnd = input.n_cols - 1;
      }
      for (size_t i = 0, rowidx = 0; i < output.n_rows;
          ++i, rowidx += strideWidth)
      {
        size_t rowEnd = rowidx + kernelWidth - 1;
        // Check if the kernel along row is out of bounds.
        if (rowEnd > input.n_rows - 1)
        {
          // If so, we need to reduce the kernel width or terminate.
          if (floor)
            continue;
          rowEnd = input.n_rows - 1;
        }
        output(i, j, s) = Pooling(input.slice(s).submat(
            rowidx,
            colidx,
            rowEnd,
            colEnd));
      }
    }
  }
}

template<typename MatType>
void MeanPoolingType<MatType>::Unpooling(
    const MatType& error,
    MatType& output)
{
  // This condition comes by comparing the number of operations involved in the
  // brute force method and the prefix method. Let the area of error be
  // errorArea and area of kernal be kernelArea. Total number of operations in
  // brute force method will be `errorArea * kernelArea` and for each element in
  // error we are doing `kernelArea` number of operations. Whereas in the prefix
  // method the total number of operations will be `4 * errorArea + 2 *
  // outputArea`. The term `2 * outputArea` comes from prefix sums performed
  // (col-wise and row-wise).  We can use this to determine which method to use.
  const bool condition = (error.n_elem * kernelHeight * kernelWidth) >
      (4 * error.n_elem + 2 * output.n_elem);

  if (condition)
  {
    // If this condition is true then theoretically the prefix sum method of
    // unpooling is faster. The aim of unpooling is to add `error(i, j) /
    // kernelArea` to `outputArea(kernal)`. This requires `outputArea.n_elem`
    // additions. So, total operations required will be `error.n_elem *
    // outputArea.n_elem` operations.  To improve this method we will use an
    // idea of prefix sums. Let's see this method in 1-D matrix then we will
    // extend it to 2-D matrix.  Let the input be a 1-D matrix input = `[0, 0,
    // 0, 0, 0, 0, 0, 0, 0, 0]` of size 10 and we want to add `10` to idx = 1 to
    // idx = 5. In brute force method we can run a loop from idx = 1 to idx = 5
    // and add `10` to each element. In prefix method We will add `+10` to idx =
    // 1 and `-10` to idx = (5 + 1). Now the input will look like `[0, +10, 0,
    // 0, 0, 0, -10, 0, 0, 0]`. After that we can just do prefix sum `input[i]
    // += input[i - 1]`. Then the input becomes `[0, +10, +10, +10, +10, +10, 0,
    // 0, 0, 0]`. So the total computation require by this method is (2
    // additions + Prefix operations).  Note that if there are `k` such
    // operation of adding a number of some continuous subarray. Then the brute
    // force method will require `k * size(subarray)` operations. But the prefix
    // method will require `2 * k + Prefix` operations, because the Prefix can
    // be performed once at the end.  Now for 2-D matrix. Lets say we want to
    // add `e` to all elements from input(x1 : x2, y1 : y2). So the inputArea =
    // (x2 - x1 + 1) * (y2 - y1 + 1).

    // In prefix method the following operations will be performed:
    //    1. Add `+e` to input(x1, y1).
    //    2. Add `-e` to input(x2 + 1, y1).
    //    3. Add `-e` to input(x1, y2 + 1).
    //    4. Add `+e` to input(x2 + 1, y2 + 1).
    //    5. Perform Prefix sum over columns i.e input(i, j) += input(i, j - 1)
    //    6. Perform Prefix sum over rows i.e input(i, j) += input(i - 1, j)
    // So lets say if we had `k` number of such operations. The brute force
    // method will require `kernelArea * k` operations.
    // The prefix method will require `4 * k + Prefix operation`.

    for (size_t j = 0, colidx = 0; j < output.n_cols; j += strideHeight,
         ++colidx)
    {
      size_t colEnd = j + kernelHeight - 1;
      // Check if the kernel along column is out of bounds.
      if (colEnd > output.n_cols - 1)
      {
        // If so, we need to reduce the kernel height or terminate.
        if (floor)
          continue;
        colEnd = output.n_cols - 1;
      }

      for (size_t i = 0, rowidx = 0; i < output.n_rows;
           i += strideWidth, ++rowidx)
      {
        // We have to add error(i, j) to output(span(rowidx, rowEnd),
        // span(colidx, colEnd)).
        // The steps of prefix sum method:
        //
        // 1. For each (rowidx, colidx) perform:
        //    1.1 Add +error(rowidx, colidx) to output(i, j)
        //    1.2 Add -error(rowidx, colidx) to output(i, colend + 1)
        //    1.3 Add -error(rowidx, colidx) to output(rowend + 1, j)
        //    1.4 Add +error(rowidx, colidx) to output(rowend + 1, colend + 1)
        //
        // 2. Do prefix sum column wise i.e output(i, j) += output(i, j - 1)
        // 2. Do prefix sum row wise i.e output(i, j) += output(i - 1, j)

        size_t rowEnd = i + kernelWidth - 1;
        // Check if the kernel along row is out of bounds.
        if (rowEnd > output.n_rows - 1)
        {
          // If so, we need to reduce the kernel width or terminate.
          if (floor)
            continue;
          rowEnd = output.n_rows - 1;
        }

        size_t kernelArea = (rowEnd - i + 1) * (colEnd - j + 1);
        output(i, j) += error(rowidx, colidx) / kernelArea;

        if (rowEnd + 1 < output.n_rows)
        {
          output(rowEnd + 1, j) -= error(rowidx, colidx) / kernelArea;

          if (colEnd + 1 < output.n_cols)
          {
            output(rowEnd + 1, colEnd + 1) += error(rowidx, colidx) /
                kernelArea;
          }
        }

        if (colEnd + 1 < output.n_cols)
          output(i, colEnd + 1) -= error(rowidx, colidx) / kernelArea;
      }
    }

    for (size_t i = 1; i < output.n_rows; ++i)
      output.row(i) += output.row(i - 1);

    for (size_t j = 1; j < output.n_cols; ++j)
      output.col(j) += output.col(j - 1);
  }
  else
  {
    arma::Mat<typename MatType::elem_type> unpooledError;
    for (size_t j = 0, colidx = 0; j < output.n_cols; j += strideHeight,
         ++colidx)
    {
      size_t colEnd = j + kernelHeight - 1;
      // Check if the kernel along column is out of bounds.
      if (colEnd > output.n_cols - 1)
      {
        // If so, we need to reduce the kernel height or terminate.
        if (floor)
          continue;
        colEnd = output.n_cols - 1;
      }

      for (size_t i = 0, rowidx = 0; i < output.n_rows; i += strideWidth,
           ++rowidx)
      {
        size_t rowEnd = i + kernelWidth - 1;
        // Check if the kernel along row is out of bounds.
        if (rowEnd > output.n_rows - 1)
        {
          // If so, we need to reduce the kernel width or terminate.
          if (floor)
            continue;
          rowEnd = output.n_rows - 1;
        }

        MatType outputArea = output(arma::span(i, rowEnd),
                                    arma::span(j, colEnd));

        unpooledError = arma::Mat<typename MatType::elem_type>(
            outputArea.n_rows, outputArea.n_cols);
        unpooledError.fill(error(rowidx, colidx) / outputArea.n_elem);

        output(arma::span(i, i + outputArea.n_rows - 1),
               arma::span(j, j + outputArea.n_cols - 1)) += unpooledError;
      }
    }
  }
}

} // namespace mlpack

#endif
