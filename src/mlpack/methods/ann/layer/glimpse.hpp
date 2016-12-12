/**
 * @file glimpse.hpp
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
 * }
 * @endcode
 */
#ifndef MLPACK_METHODS_ANN_LAYER_GLIMPSE_HPP
#define MLPACK_METHODS_ANN_LAYER_GLIMPSE_HPP

#include <mlpack/core.hpp>

#include "layer_types.hpp"
#include <algorithm>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


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
    output = arma::zeros<MatType>(input.n_rows, input.n_cols);
    const double mean = arma::mean(arma::mean(input));

    output.elem(arma::find(mean == input, 1)).fill(value);
  }
};

/**
 * The glimpse layer returns a retina-like representation
 * (down-scaled cropped images) of increasing scale around a given location in a
 * given image.
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
class Glimpse
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
  Glimpse(const size_t inSize,
          const size_t size,
          const size_t depth = 3,
          const size_t scale = 2,
          const size_t inputWidth = 0,
          const size_t inputHeight = 0) :
      inSize(inSize),
      size(size),
      depth(depth),
      scale(scale),
      inputWidth(inputWidth),
      inputHeight(inputHeight)
  {
    // Nothing to do here.
  }

  /**
   * Ordinary feed forward pass of the glimpse layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
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

  /**
   * Ordinary feed backward pass of the glimpse layer.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>&& /* input */,
                arma::Mat<eT>&& gy,
                arma::Mat<eT>&& g)
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

  //! Get the input parameter.
  InputDataType& InputParameter() const {return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const {return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the detla.
  OutputDataType& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Set the locationthe x and y coordinate of the center of the output
  //! glimpse.
  void Location(const arma::mat& location)
  {
    this->location = location;
  }

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

  //! Get the value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

 private:
  /*
   * Transform the given input by changing rows to columns.
   *
   * @param w The input matrix used to perform the transformation.
   */
  void Transform(arma::mat& w)
  {
    arma::mat t = w;

    for (size_t i = 0, k = 0; i < w.n_elem; k++)
    {
      for (size_t j = 0; j < w.n_cols; j++, i++)
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
    for (size_t i = 0; i < w.n_slices; i++)
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
  template<typename eT>
  void Pooling(const size_t kSize,
               const arma::Mat<eT>& input,
               arma::Mat<eT>& output)
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
  template<typename eT>
  void Unpooling(const arma::Mat<eT>& input,
                 const arma::Mat<eT>& error,
                 arma::Mat<eT>& output)
  {
    const size_t rStep = input.n_rows / error.n_rows;
    const size_t cStep = input.n_cols / error.n_cols;

    arma::Mat<eT> unpooledError;
    for (size_t j = 0; j < input.n_cols; j += cStep)
    {
      for (size_t i = 0; i < input.n_rows; i += rStep)
      {
        const arma::Mat<eT>& inputArea = input(arma::span(i, i + rStep - 1),
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
  template<typename eT>
  void ReSampling(const arma::Mat<eT>& input, arma::Mat<eT>& output)
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
  template<typename eT>
  void DownwardReSampling(const arma::Mat<eT>& input,
                          const arma::Mat<eT>& error,
                          arma::Mat<eT>& output)
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

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored depth of the input.
  size_t inputDepth;

  //! Locally-stored transformed input parameter.
  arma::cube inputTemp;

  //! Locally-stored transformed output parameter.
  arma::cube outputTemp;

  //! The x and y coordinate of the center of the output glimpse.
  arma::mat location;

  //! Locally-stored object to perform the mean pooling operation.
  MeanPoolingRule pooling;

  //! Location-stored module location parameter.
  std::vector<arma::mat> locationParameter;

  //! Location-stored transformed gradient paramter.
  arma::cube gTemp;

  //! If true use maximum a posteriori during the forward pass.
  bool deterministic;
}; // class GlimpseLayer

}; // namespace ann
}; // namespace mlpack

#endif