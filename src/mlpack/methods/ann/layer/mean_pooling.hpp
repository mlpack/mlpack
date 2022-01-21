/**
 * @file methods/ann/layer/mean_pooling.hpp
 * @author Marcus Edel
 * @author Nilay Jain
 *
 * Definition of the MeanPooling layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MEAN_POOLING_HPP
#define MLPACK_METHODS_ANN_LAYER_MEAN_POOLING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the MeanPooling.
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class MeanPoolingType : public Layer<InputType, OutputType>
{
 public:
  //! Create the MeanPoolingType object.
  MeanPoolingType();

  /**
   * Create the MeanPooling object using the specified number of units.
   *
   * @param kernelWidth Width of the pooling window.
   * @param kernelHeight Height of the pooling window.
   * @param strideWidth Width of the stride operation.
   * @param strideHeight Width of the stride operation.
   * @param floor Set to true to use floor method.
   */
  MeanPoolingType(const size_t kernelWidth,
                  const size_t kernelHeight,
                  const size_t strideWidth = 1,
                  const size_t strideHeight = 1,
                  const bool floor = true);

  // TODO: copy constructor / move constructor
  MeanPoolingType* Clone() const { return new MeanPoolingType(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  //! Get the kernel width.
  size_t const& KernelWidth() const { return kernelWidth; }
  //! Modify the kernel width.
  size_t& KernelWidth() { return kernelWidth; }

  //! Get the kernel height.
  size_t const& KernelHeight() const { return kernelHeight; }
  //! Modify the kernel height.
  size_t& KernelHeight() { return kernelHeight; }

  //! Get the stride width.
  size_t const& StrideWidth() const { return strideWidth; }
  //! Modify the stride width.
  size_t& StrideWidth() { return strideWidth; }

  //! Get the stride height.
  size_t const& StrideHeight() const { return strideHeight; }
  //! Modify the stride height.
  size_t& StrideHeight() { return strideHeight; }

  //! Get the value of the rounding operation
  bool const& Floor() const { return floor; }
  //! Modify the value of the rounding operation
  bool& Floor() { return floor; }

  //! Get the size of the output.
  const std::vector<size_t> OutputDimensions() const
  {
    outputDimensions = this->inputDimensions;

    // Compute the size of the output.
    if (floor)
    {
      outputDimensions[0] = std::floor((this->inputDimensions[0] -
          (double) kernelWidth) / (double) strideWidth + 1);
      outputDimensions[1] = std::floor((this->inputDimensions[1] -
          (double) kernelHeight) / (double) strideHeight + 1);
      offset = 0;
    }
    else
    {
      outputDimensions[0] = std::ceil((this->inputDimensions[0] -
          (double) kernelWidth) / (double) strideWidth + 1);
      outputDimensions[1] = std::ceil((this->inputDimensions[1] -
          (double) kernelHeight) / (double) strideHeight + 1);
      offset = 1;
    }

    // Higher dimensions are not modified.
    for (size_t i = 2; i < this->inputDimensions.size(); ++i)
      outputDimensions[i] = this->inputDimensions[i];

    // Cache input size and output size.
    channels = 1;
    for (size_t i = 2; i < this->inputDimensions.size(); ++i)
      channels *= this->inputDimensions[i];

    return outputDimensions;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * Apply pooling to the input and store the results.
   *
   * @param input The input to be apply the pooling rule.
   * @param output The pooled result.
   */
  void Pooling(const InputType& input, OutputType& output)
  {
    arma::Mat<eT> inputPre = input;

    for (size_t i = 1; i < input.n_cols; ++i)
      inputPre.col(i) += inputPre.col(i - 1);

    for (size_t i = 1; i < input.n_rows; ++i)
      inputPre.row(i) += inputPre.row(i - 1);

    for (size_t j = 0, colidx = 0; j < output.n_cols;
         ++j, colidx += strideHeight)
    {
      for (size_t i = 0, rowidx = 0; i < output.n_rows;
           ++i, rowidx += strideWidth)
      {
        InputType subInput = input(
            arma::span(rowidx, rowidx + kernelWidth - 1 - offset),
            arma::span(colidx, colidx + kernelHeight - 1 - offset));

        if (rowEnd > input.n_rows - 1)
          rowEnd = input.n_rows - 1;
        if (colEnd > input.n_cols - 1)
          colEnd = input.n_cols - 1;

        const size_t kernalArea = (rowEnd - rowidx + 1) * (colEnd - colidx + 1);
        val += inputPre(rowEnd, colEnd);
        if (rowidx >= 1)
        {
          if (colidx >= 1)
            val += inputPre(rowidx - 1, colidx - 1);
          val -= inputPre(rowidx - 1, colEnd);
        }
        if (colidx >= 1)
          val -= inputPre(rowEnd, colidx - 1);

        output(i, j) = val / kernalArea;
      }
    }
  }

  /**
   * Apply unpooling to the input and store the results.
   *
   * @param input The input to be apply the unpooling rule.
   * @param output The pooled result.
   */
  void Unpooling(const InputType& input,
                 const OutputType& error,
                 OutputType& output)
  {
    // This condition comes by comparing the number of operations involved in the brute
    // force method and the prefix method. Let the area of error be errorArea and area
    // of kernal be kernalArea. Total number of operations in brute force method will be
    // `errorArea * kernalArea` and for each element in error we are doing `kernalArea`
    // number of operations. Whereas in the prefix method the total number of operations
    // will be `4 * errorArea + 2 * inputArea`. The term `2 * inputArea` comes from
    // prefix sums performed (col-wise and row-wise).
    // We can use this to determine which method to use.
    const bool condition = (error.n_elem * kernelHeight * kernelWidth) >
        (4 * error.n_elem + 2 * input.n_elem);

    OutputType unpooledError;
    for (size_t j = 0; j < input.n_cols - cStep; j += cStep)
    {
      // If this condition is true then theoritically the prefix sum method of
      // unpooling is faster. The aim of unpooling is to add
      // `error(i, j) / kernalArea` to `inputArea(kernal)`. This requires
      // `inputArea.n_elem` additions. So, total operations required will be
      // `error.n_elem * inputArea.n_elem` operations.
      // To improve this method we will use an idea of prefix sums. Let's see
      // this method in 1-D matrix then we will extend it to 2-D matrix.
      // Let the input be a 1-D matrix input = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` of size 10
      // and we want to add `10` to idx = 1 to idx = 5. In brute force method we can run
      // a loop from idx = 1 to idx = 5 and add `10` to each element. In prefix method
      // We will add `+10` to idx = 1 and `-10` to idx = (5 + 1). Now the input will look
      // like `[0, +10, 0, 0, 0, 0, -10, 0, 0, 0]`. After that we can just do prefix
      // sum `input[i] += input[i - 1]`. Then the input becomes
      // `[0, +10, +10, +10, +10, +10, 0, 0, 0, 0]`. So the total computation require
      // by this method is (2 additions + Prefix operations).
      // Note that if there are `k` such operation of adding a number of some
      // continuous subarray. Then the brute force method will require
      // `k * size(subarray)` operations. But the prefix method will require
      // `2 * k + Prefix` operations, because the Prefix can be performed once at
      // the end.
      // Now for 2-D matrix. Lets say we want to add `e` to all elements from
      // input(x1 : x2, y1 : y2). So the inputArea = (x2 - x1 + 1) * (y2 - y1 + 1).
      // In prefix method the following operations will be performed:
      //    1. Add `+e` to input(x1, y1).
      //    2. Add `-e` to input(x1 + 1, y1).
      //    3. Add `-e` to input(x1, y1 + 1).
      //    4. Add `+e` to input(x1 + 1, y1 + 1).
      //    5. Perform Prefix sum over columns i.e input(i, j) += input(i, j - 1)
      //    6. Perform Prefix sum over rows i.e input(i, j) += input(i - 1, j)
      // So lets say if we had `k` number of such operations. The brute force
      // method will require `kernalArea * k` operations.
      // The prefix method will require `4 * k + Prefix operation`.

      for (size_t j = 0, colidx = 0; j < input.n_cols; j += strideHeight, ++colidx)
      {
        for (size_t i = 0, rowidx = 0; i < input.n_rows; i += strideWidth, ++rowidx)
        {
          // We have to add error(i, j) to output(span(rowidx, rowEnd),
          // span(colidx, colEnd)).
          // The steps of prefix sum method:
          //
          // 1. For each (i, j) perform:
          //    1.1 Add +error(i, j) to output(rowidx, colidx)
          //    1.2 Add -error(i, j) to output(rowidx, colidx + 1)
          //    1.3 Add -error(i, j) to output(rowidx + 1, colidx)
          //    1.4 Add +error(i, j) to output(rowidx + 1, colidx + 1)
          //
          // 2. Do prefix sum column wise i.e output(i, j) += output(i, j - 1)
          // 2. Do prefix sum row wise i.e output(i, j) += output(i - 1, j)

          size_t rowEnd = i + kernelWidth - 1;
          size_t colEnd = j + kernelHeight - 1;

          if (rowEnd > input.n_rows - 1)
          {
            if (floor)
              continue;
            rowEnd = input.n_rows - 1;
          }

          if (colEnd > input.n_cols - 1)
          {
            if (floor)
              continue;
            colEnd = input.n_cols - 1;
          }

          size_t kernalArea = (rowEnd - i + 1) * (colEnd - j + 1);
          output(i, j) += error(rowidx, colidx) / kernalArea;

          if (rowEnd + 1 < input.n_rows)
          {
            output(rowEnd + 1, j) -= error(rowidx, colidx) / kernalArea;

            if (colEnd + 1 < input.n_cols)
              output(rowEnd + 1, colEnd + 1) += error(rowidx, colidx) / kernalArea;
          }

          if (colEnd + 1 < input.n_cols)
            output(i, colEnd + 1) -= error(rowidx, colidx) / kernalArea;
        }
      }

      for (size_t i = 1; i < input.n_rows; ++i)
        output.row(i) += output.row(i - 1);

      for (size_t j = 1; j < input.n_cols; ++j)
        output.col(j) += output.col(j - 1);
    }
    else
    {
      arma::Mat<eT> unpooledError;
      for (size_t j = 0, colidx = 0; j < input.n_cols; j += strideHeight, ++colidx)
      {
        for (size_t i = 0, rowidx = 0; i < input.n_rows; i += strideWidth, ++rowidx)
        {
          size_t rowEnd = i + kernelWidth - 1;
          size_t colEnd = j + kernelHeight - 1;

          if (rowEnd > input.n_rows - 1)
          {
            if (floor)
              continue;
            rowEnd = input.n_rows - 1;
          }

          if (colEnd > input.n_cols - 1)
          {
            if (floor)
              continue;
            colEnd = input.n_cols - 1;
          }

          arma::mat InputArea = input(arma::span(i, rowEnd), arma::span(j, colEnd));

          unpooledError = arma::Mat<eT>(InputArea.n_rows, InputArea.n_cols);
          unpooledError.fill(error(rowidx, colidx) / InputArea.n_elem);

          output(arma::span(i, i + InputArea.n_rows - 1),
              arma::span(j, j + InputArea.n_cols - 1)) += unpooledError;
        }
      }
    }
  }

  //! Locally-stored width of the pooling window.
  size_t kernelWidth;

  //! Locally-stored height of the pooling window.
  size_t kernelHeight;

  //! Locally-stored width of the stride operation.
  size_t strideWidth;

  //! Locally-stored height of the stride operation.
  size_t strideHeight;

  //! Rounding operation used.
  bool floor;

  //! Locally-stored number channels.
  size_t channels;

  //! Locally-stored cached output dimensions.
  std::vector<size_t> outputDimensions;

  //! Locally-stored number of input units.
  size_t batchSize;

  //! Cached last-seen input.
  arma::Cube<typename InputType::elem_type> inputTemp;
}; // class MeanPoolingType

// Standard MeanPooling layer.
typedef MeanPoolingType<arma::mat, arma::mat> MeanPooling;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "mean_pooling_impl.hpp"

#endif
