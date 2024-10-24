// Temporarily drop.
/**
 * @file methods/ann/layer/glimpse.hpp
 * @author Marcus Edel
 *
 * Definition of the GlimpseLayer class, which takes an input image and a
 * location to extract a retina-like representation of the input image at
 * different increasing scales.
 *
 * For more information, see the following.
 *
 * @code
 * @article{CoRR2014,
 *   author  = {Volodymyr Mnih, Nicolas Heess, Alex Graves, Koray Kavukcuoglu},
 *   title   = {Recurrent Models of Visual Attention},
 *   journal = {CoRR},
 *   volume  = {abs/1406.6247},
 *   year    = {2014},
 *   url     = {https://arxiv.org/abs/1406.6247}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_GLIMPSE_HPP
#define MLPACK_METHODS_ANN_LAYER_GLIMPSE_HPP

#include <mlpack/prereqs.hpp>

#include "layer_types.hpp"
#include <algorithm>

namespace mlpack {


/*
 * The mean pooling rule for convolution neural networks. Average all values
 * within the receptive block.
 */
class MeanPoolingRule
{
 public:
  /*
   * Return the average value within the receptive block.
   *
   * @param input Input used to perform the pooling operation.
   */
  template<typename MatType>
  double Pooling(const MatType& input)
  {
    return arma::mean(arma::mean(input));
  }

  /*
   * Set the average value within the receptive block.
   *
   * @param input Input used to perform the pooling operation.
   * @param value The unpooled value.
   * @param output The unpooled output data.
   */
  template<typename MatType>
  void Unpooling(const MatType& input, const double value, MatType& output)
  {
    output = zeros<MatType>(input.n_rows, input.n_cols);
    const double mean = arma::mean(arma::mean(input));

    output.elem(arma::find(mean == input, 1)).fill(value);
  }
};

/**
 * The glimpse layer returns a retina-like representation
 * (down-scaled cropped images) of increasing scale around a given location in a
 * given image.
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
class GlimpseType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create the GlimpseLayer object using the specified ratio and rescale
   * parameter.
   *
   * @param inSize The size of the input units.
   * @param size The used glimpse size (height = width).
   * @param depth The number of patches to crop per glimpse.
   * @param scale The scaling factor used to create the increasing retina-like
   *        representation.
   * @param inputWidth The input width of the given input data.
   * @param inputHeight The input height of the given input data.
   */
  GlimpseType(const size_t inSize = 0,
              const size_t size = 0,
              const size_t depth = 3,
              const size_t scale = 2,
              const size_t inputWidth = 0,
              const size_t inputHeight = 0);

  /**
   * Ordinary feed forward pass of the glimpse layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of the glimpse layer.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  //! Set the locationthe x and y coordinate of the center of the output
  //! glimpse.
  void Location(const arma::mat& location) { this->location = location; }

  //! Get the input width.
  size_t const& InputWidth() const { return inputWidth; }
  //! Modify input the width.
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

  //! Get the number of patches to crop per glimpse.
  size_t const& Depth() const { return depth; }

  //! Get the scale fraction.
  size_t const& Scale() const { return scale; }

  //! Get the size of the input units.
  size_t InSize() const { return inSize; }

  //! Get the used glimpse size (height = width).
  size_t GlimpseSize() const { return size;}

  const std::vector<size_t> OutputDimensions() const
  {
    std::vector<size_t> result(inputDimensions.size(), 0);
    result[0] = outputWidth;
    result[1] = outputHeight;
    for (size_t i = 2; i < inputDimensions.size(); ++i)
      result[i] = inputDimensions[i];
    return result;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * Transform the given input by changing rows to columns.
   *
   * @param w The input matrix used to perform the transformation.
   */
  void Transform(arma::mat& w)
  {
    arma::mat t = w;

    for (size_t i = 0, k = 0; i < w.n_elem; ++k)
    {
      for (size_t j = 0; j < w.n_cols; ++j, ++i)
      {
        w(k, j) = t(i);
      }
    }
  }

  /*
   * Transform the given input by changing rows to columns.
   *
   * @param w The input matrix used to perform the transformation.
   */
  void Transform(arma::cube& w)
  {
    for (size_t i = 0; i < w.n_slices; ++i)
    {
      arma::mat t = w.slice(i);
      Transform(t);
      w.slice(i) = t;
    }
  }

  /**
   * Apply pooling to the input and store the results to the output parameter.
   *
   * @param kSize the kernel size used to perform the pooling operation.
   * @param input The input to be apply the pooling rule.
   * @param output The pooled result.
   */
  void Pooling(const size_t kSize,
               const InputType& input,
               OutputType& output)
  {
    const size_t rStep = kSize;
    const size_t cStep = kSize;

    for (size_t j = 0; j < input.n_cols; j += cStep)
    {
      for (size_t i = 0; i < input.n_rows; i += rStep)
      {
        output(i / rStep, j / cStep) += pooling.Pooling(
            input(arma::span(i, i + rStep - 1), arma::span(j, j + cStep - 1)));
      }
    }
  }

  /**
   * Apply unpooling to the input and store the results.
   *
   * @param input The input to be apply the unpooling rule.
   * @param error The error used to perform the unpooling operation.
   * @param output The pooled result.
   */
  void Unpooling(const InputType& input,
                 const OutputType& error,
                 OutputType& output)
  {
    const size_t rStep = input.n_rows / error.n_rows;
    const size_t cStep = input.n_cols / error.n_cols;

    OutputType unpooledError;
    for (size_t j = 0; j < input.n_cols; j += cStep)
    {
      for (size_t i = 0; i < input.n_rows; i += rStep)
      {
        const InputType& inputArea = input(arma::span(i, i + rStep - 1),
                                           arma::span(j, j + cStep - 1));

        pooling.Unpooling(inputArea, error(i / rStep, j / cStep),
            unpooledError);

        output(arma::span(i, i + rStep - 1),
            arma::span(j, j + cStep - 1)) += unpooledError;
      }
    }
  }

  /**
   * Apply ReSampling to the input and store the results in the output
   * parameter.
   *
   * @param input The input to be apply the ReSampling rule.
   * @param output The pooled result.
   */
  void ReSampling(const InputType& input, OutputType& output)
  {
    double wRatio = (double) (input.n_rows - 1) / (size - 1);
    double hRatio = (double) (input.n_cols - 1) / (size - 1);

    double iWidth = input.n_rows - 1;
    double iHeight = input.n_cols - 1;

    for (size_t y = 0; y < size; y++)
    {
      for (size_t x = 0; x < size; x++)
      {
        double ix = wRatio * x;
        double iy = hRatio * y;

        // Get the 4 nearest neighbors.
        double ixNw = std::floor(ix);
        double iyNw = std::floor(iy);
        double ixNe = ixNw + 1;
        double iySw = iyNw + 1;

        // Get surfaces to each neighbor.
        double se = (ix - ixNw) * (iy - iyNw);
        double sw = (ixNe - ix) * (iy - iyNw);
        double ne = (ix - ixNw) * (iySw - iy);
        double nw = (ixNe - ix) * (iySw - iy);

        // Calculate the weighted sum.
        output(y, x) = input(iyNw, ixNw) * nw +
            input(iyNw, std::min(ixNe,  iWidth)) * ne +
            input(std::min(iySw, iHeight), ixNw) * sw +
            input(std::min(iySw, iHeight), std::min(ixNe, iWidth)) * se;
      }
    }
  }

  /**
   * Apply DownwardReSampling to the input and store the results into the output
   * parameter.
   *
   * @param input The input to be apply the DownwardReSampling rule.
   * @param error The error used to perform the DownwardReSampling operation.
   * @param output The DownwardReSampled result.
   */
  void DownwardReSampling(const InputType& input,
                          const OutputType& error,
                          OutputType& output)
  {
    double iWidth = input.n_rows - 1;
    double iHeight = input.n_cols - 1;

    double wRatio = iWidth / (size - 1);
    double hRatio = iHeight / (size - 1);

    for (size_t y = 0; y < size; y++)
    {
      for (size_t x = 0; x < size; x++)
      {
        double ix = wRatio * x;
        double iy = hRatio * y;

        // Get the 4 nearest neighbors.
        double ixNw = std::floor(ix);
        double iyNw = std::floor(iy);
        double ixNe = ixNw + 1;
        double iySw = iyNw + 1;

        // Get surfaces to each neighbor.
        double se = (ix - ixNw) * (iy - iyNw);
        double sw = (ixNe - ix) * (iy - iyNw);
        double ne = (ix - ixNw) * (iySw - iy);
        double nw = (ixNe - ix) * (iySw - iy);

        double ograd = error(y, x);

        output(iyNw, ixNw) = output(iyNw, ixNw) + nw * ograd;
        output(iyNw, std::min(ixNe, iWidth)) = output(iyNw,
            std::min(ixNe, iWidth)) + ne * ograd;
        output(std::min(iySw, iHeight), ixNw) = output(std::min(iySw, iHeight),
            ixNw) + sw * ograd;
        output(std::min(iySw, iHeight), std::min(ixNe, iWidth)) = output(
            std::min(iySw, iHeight), std::min(ixNe, iWidth)) + se * ograd;
      }
    }
  }

  //! The size of the input units.
  size_t inSize;

  //! The used glimpse size (height = width).
  size_t size;

  //! The number of patches to crop per glimpse.
  size_t depth;

  //! The scale fraction.
  size_t scale;

  //! Locally-stored input width.
  size_t inputWidth;

  //! Locally-stored input height.
  size_t inputHeight;

  //! Locally-stored output width.
  size_t outputWidth;

  //! Locally-stored output height.
  size_t outputHeight;

  //! Locally-stored depth of the input.
  size_t inputDepth;

  //! Locally-stored transformed input parameter.
  arma::Cube<typename InputType::elem_type> inputTemp;

  //! Locally-stored transformed output parameter.
  arma::Cube<typename OutputType::elem_type> outputTemp;

  //! The x and y coordinate of the center of the output glimpse.
  OutputType location;

  //! Locally-stored object to perform the mean pooling operation.
  MeanPoolingRule pooling;

  //! Location-stored module location parameter.
  std::vector<OutputType> locationParameter;

  //! Location-stored transformed gradient paramter.
  arma::Cube<typename OutputType::elem_type> gTemp;
}; // class GlimpseType

// Standard Glimpse layer.
using Glimpse = GlimpseType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "glimpse_impl.hpp"

#endif
