/**
 * @file methods/ann/layer/max_pooling.hpp
 * @author Marcus Edel
 * @author Nilay Jain
 *
 * Definition of the MaxPooling class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MAX_POOLING_HPP
#define MLPACK_METHODS_ANN_LAYER_MAX_POOLING_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * The max pooling rule for convolution neural networks. Take the maximum value
 * within the receptive block.
 */
class MaxPoolingRule
{
 public:
  /**
   * Return the maximum value within the receptive block.
   *
   * @param input Input used to perform the pooling operation.  Could be an
   *     Armadillo subview.
   */
  template<typename MatType>
  typename MatType::elem_type Pooling(const MatType& input)
  {
    return max(vectorise(input));
  }

  template<typename MatType>
  std::tuple<size_t, typename MatType::elem_type> PoolingWithIndex(
      const MatType& input)
  {
    const size_t index = input.index_max();
    const typename MatType::elem_type maxVal = input[index];

    return std::tuple<size_t, typename MatType::elem_type>(index, maxVal);
  }
};

/**
 * Implementation of the MaxPooling layer.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class MaxPoolingType : public Layer<MatType>
{
 public:
  //! Create the MaxPooling object.
  MaxPoolingType();

  /**
   * Create the MaxPooling object using the specified number of units.
   *
   * @param kernelWidth Width of the pooling window.
   * @param kernelHeight Height of the pooling window.
   * @param strideWidth Width of the stride operation.
   * @param strideHeight Width of the stride operation.
   * @param floor If true, then a pooling operation that would oly part of the
   *              input will be skipped.
   */
  MaxPoolingType(const size_t kernelWidth,
                 const size_t kernelHeight,
                 const size_t strideWidth = 1,
                 const size_t strideHeight = 1,
                 const bool floor = true);

  // Virtual destructor.
  virtual ~MaxPoolingType() { }

  //! Copy the given MaxPoolingType.
  MaxPoolingType(const MaxPoolingType& other);
  //! Take ownership of the given MaxPoolingType.
  MaxPoolingType(MaxPoolingType&& other);
  //! Copy the given MaxPoolingType.
  MaxPoolingType& operator=(const MaxPoolingType& other);
  //! Take ownership of the given MaxPoolingType.
  MaxPoolingType& operator=(MaxPoolingType&& other);

  //! Clone the MaxPoolingType object. This handles polymorphism correctly.
  MaxPoolingType* Clone() const { return new MaxPoolingType(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g);

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

  //! Get the value of the rounding operation.
  bool const& Floor() const { return floor; }
  //! Modify the value of the rounding operation.
  bool& Floor() { return floor; }

  //! Compute the size of the output given `InputDimensions()`.
  void ComputeOutputDimensions();

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
   * @param poolingIndices The pooled indices.
   */
  void PoolingOperation(
      const arma::Cube<typename MatType::elem_type>& input,
      arma::Cube<typename MatType::elem_type>& output,
      arma::Cube<size_t>& poolingIndices)
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
          const std::tuple<size_t, typename MatType::elem_type> poolResult =
              pooling.PoolingWithIndex(input.slice(s).submat(
                  rowidx,
                  colidx,
                  rowEnd,
                  colEnd));

          // Now map the returned pooling index, which corresponds to the
          // submatrix we gave, back to its position in the (linearized) input.
          const size_t poolIndex = std::get<0>(poolResult);
          const size_t poolingCol = poolIndex / (kernelWidth);
          const size_t poolingRow = poolIndex % (kernelWidth);
          const size_t unmappedPoolingIndex = (rowidx + poolingRow) +
              input.n_rows * (colidx + poolingCol);

          poolingIndices(i, j, s) = unmappedPoolingIndex;
          output(i, j, s) = std::get<1>(poolResult);
        }
      }
    }
  }

  /**
   * Apply pooling to all slices of the input and store the results, but not the
   * indices used.
   *
   * @param input The input to apply the pooling rule to.
   * @param output The pooled result.
   */
  void PoolingOperation(
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

          output(i, j, s) = pooling.Pooling(input.slice(s).submat(
              rowidx,
              colidx,
              rowEnd,
              colEnd));
        }
      }
    }
  }

  /**
   * Apply unpooling to all slices of the input and store the results.
   *
   * @param error The backward error.
   * @param output The pooled result.
   * @param poolingIndices The pooled indices (from `PoolingOperation()`).
   */
  void UnpoolingOperation(
      const MatType& error,
      MatType& output,
      const arma::Mat<size_t>& poolingIndices)
  {
    for (size_t i = 0; i < poolingIndices.n_elem; ++i)
    {
      output(poolingIndices(i)) += error(i);
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

  //! Locally-stored number of channels.
  size_t channels;

  //! Locally-stored pooling strategy.
  MaxPoolingRule pooling;

  //! Locally-stored pooling indices.
  arma::Cube<size_t> poolingIndices;
}; // class MaxPoolingType

// Standard MaxPooling layer.
using MaxPooling = MaxPoolingType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "max_pooling_impl.hpp"

#endif
