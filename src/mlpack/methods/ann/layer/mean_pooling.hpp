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
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class MeanPooling
{
 public:
  //! Create the MeanPooling object.
  MeanPooling();

  /**
   * Create the MeanPooling object using the specified number of units.
   *
   * @param kernelWidth Width of the pooling window.
   * @param kernelHeight Height of the pooling window.
   * @param strideWidth Width of the stride operation.
   * @param strideHeight Width of the stride operation.
   * @param floor Set to true to use floor method.
   */
  MeanPooling(const size_t kernelWidth,
              const size_t kernelHeight,
              const size_t strideWidth = 1,
              const size_t strideHeight = 1,
              const bool floor = true);

   //! Copy Constructor.
  MeanPooling(const MeanPooling& layer);
    
  //! Move Constructor.
  MeanPooling(MeanPooling&& layer);
    
  //! Copy assignment operator.
  MeanPooling& operator=(const MeanPooling& layer);
    
  //! Move assignment operator.
  MeanPooling& operator=(MeanPooling&& layer);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the intput width.
  size_t const& InputWidth() const { return inputWidth; }
  //! Modify the input width.
  size_t& InputWidth() { return inputWidth; }

  //! Get the input height.
  size_t const& InputHeight() const { return inputHeight; }
  //! Modify the input height.
  size_t& InputHeight() { return inputHeight; }

  //! Get the output width.
  size_t const& OutputWidth() const { return outputWidth; }
  //! Modify the output width.
  size_t& OutputWidth() { return outputWidth; }

  //! Get the output height.
  size_t const& OutputHeight() const { return outputHeight; }
  //! Modify the output height.
  size_t& OutputHeight() { return outputHeight; }

  //! Get the input size.
  size_t InputSize() const { return inSize; }

  //! Get the output size.
  size_t OutputSize() const { return outSize; }

  //! Get the kernel width.
  size_t KernelWidth() const { return kernelWidth; }
  //! Modify the kernel width.
  size_t& KernelWidth() { return kernelWidth; }

  //! Get the kernel height.
  size_t KernelHeight() const { return kernelHeight; }
  //! Modify the kernel height.
  size_t& KernelHeight() { return kernelHeight; }

  //! Get the stride width.
  size_t StrideWidth() const { return strideWidth; }
  //! Modify the stride width.
  size_t& StrideWidth() { return strideWidth; }

  //! Get the stride height.
  size_t StrideHeight() const { return strideHeight; }
  //! Modify the stride height.
  size_t& StrideHeight() { return strideHeight; }

  //! Get the value of the rounding operation
  bool const& Floor() const { return floor; }
  //! Modify the value of the rounding operation
  bool& Floor() { return floor; }

  //! Get the value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

  //! Get the size of the weights.
  size_t WeightSize() const { return 0; }

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
  template<typename eT>
  void Pooling(const arma::Mat<eT>& input, arma::Mat<eT>& output)
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
        double val = 0.0;
        size_t rowEnd = rowidx + kernelWidth - 1;
        size_t colEnd = colidx + kernelHeight - 1;

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
  template<typename eT>
  void Unpooling(const arma::Mat<eT>& input,
                 const arma::Mat<eT>& error,
                 arma::Mat<eT>& output)
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

    if (condition)
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
          // We have to add error(i, j) to output(span(rowidx, rowEnd), span(colidx, colEnd)).
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

  //! Locally-stored number of input channels.
  size_t inSize;

  //! Locally-stored number of output channels.
  size_t outSize;

  //! Locally-stored input width.
  size_t inputWidth;

  //! Locally-stored input height.
  size_t inputHeight;

  //! Locally-stored output width.
  size_t outputWidth;

  //! Locally-stored output height.
  size_t outputHeight;

  //! Locally-stored reset parameter used to initialize the module once.
  bool reset;

  //! If true use maximum a posteriori during the forward pass.
  bool deterministic;

  //! Locally-stored number of input units.
  size_t batchSize;

  //! Locally-stored output parameter.
  arma::cube outputTemp;

  //! Locally-stored transformed input parameter.
  arma::cube inputTemp;

  //! Locally-stored transformed output parameter.
  arma::cube gTemp;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class MeanPooling


} // namespace ann
} // namespace mlpack

// Include implementation.
#include "mean_pooling_impl.hpp"

#endif
