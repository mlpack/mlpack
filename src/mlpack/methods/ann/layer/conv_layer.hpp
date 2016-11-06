/**
 * @file conv_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the ConvLayer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONV_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_CONV_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/convolution_rules/border_modes.hpp>
#include <mlpack/methods/ann/convolution_rules/naive_convolution.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the ConvLayer class. The ConvLayer class represents a
 * single layer of a neural network.
 *
 * @tparam ForwardConvolutionRule Convolution to perform forward process.
 * @tparam BackwardConvolutionRule Convolution to perform backward process.
 * @tparam GradientConvolutionRule Convolution to calculate gradient.
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename ForwardConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename BackwardConvolutionRule = NaiveConvolution<FullConvolution>,
    typename GradientConvolutionRule = NaiveConvolution<ValidConvolution>,
    typename InputDataType = arma::cube,
    typename OutputDataType = arma::cube
>
class ConvLayer
{
 public:
  /**
   * Create the ConvLayer object using the specified number of input maps,
   * output maps, filter size, stride and padding parameter.
   *
   * @param inMaps The number of input maps.
   * @param outMaps The number of output maps.
   * @param wfilter Width of the filter/kernel.
   * @param wfilter Height of the filter/kernel.
   * @param xStride Stride of filter application in the x direction.
   * @param yStride Stride of filter application in the y direction.
   * @param wPad Spatial padding width of the input.
   * @param hPad Spatial padding height of the input.
   */
  ConvLayer(const size_t inMaps,
            const size_t outMaps,
            const size_t wfilter,
            const size_t hfilter,
            const size_t xStride = 1,
            const size_t yStride = 1,
            const size_t wPad = 0,
            const size_t hPad = 0) :
      wfilter(wfilter),
      hfilter(hfilter),
      inMaps(inMaps),
      outMaps(outMaps),
      xStride(xStride),
      yStride(yStride),
      wPad(wPad),
      hPad(hPad)
  {
    weights.set_size(wfilter, hfilter, inMaps * outMaps);
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Cube<eT>& input, arma::Cube<eT>& output)
  {
    const size_t wConv = ConvOutSize(input.n_rows, wfilter, xStride, wPad);
    const size_t hConv = ConvOutSize(input.n_cols, hfilter, yStride, hPad);

    output = arma::zeros<arma::Cube<eT> >(wConv, hConv, outMaps);
    for (size_t outMap = 0, outMapIdx = 0; outMap < outMaps; outMap++)
    {
      for (size_t inMap = 0; inMap < inMaps; inMap++, outMapIdx++)
      {
        arma::Mat<eT> convOutput;
        ForwardConvolutionRule::Convolution(input.slice(inMap),
            weights.slice(outMap), convOutput);

        output.slice(outMap) += convOutput;
      }
    }
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Cube<eT>& /* unused */,
                const arma::Cube<eT>& gy,
                arma::Cube<eT>& g)
  {
    g = arma::zeros<arma::Cube<eT> >(inputParameter.n_rows,
                                     inputParameter.n_cols,
                                     inputParameter.n_slices);

    for (size_t outMap = 0, outMapIdx = 0; outMap < inMaps; outMap++)
    {
      for (size_t inMap = 0; inMap < outMaps; inMap++, outMapIdx++)
      {
        arma::Mat<eT> rotatedFilter;
        Rotate180(weights.slice(outMap * outMaps + inMap), rotatedFilter);

        arma::Mat<eT> output;
        BackwardConvolutionRule::Convolution(gy.slice(inMap), rotatedFilter,
            output);

        g.slice(outMap) += output;
      }
    }
  }

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param d The calculated error.
   * @param g The calculated gradient.
   */
  template<typename InputType, typename eT>
  void Gradient(const InputType& input,
                const arma::Cube<eT>& d,
                arma::Cube<eT>& g)
  {
    g = arma::zeros<arma::Cube<eT> >(weights.n_rows, weights.n_cols,
        weights.n_slices);

    for (size_t outMap = 0; outMap < outMaps; outMap++)
    {
      for (size_t inMap = 0, s = outMap; inMap < inMaps; inMap++, s += outMaps)
      {
        arma::Cube<eT> inputSlices = input.slices(inMap, inMap);
        arma::Cube<eT> deltaSlices = d.slices(outMap, outMap);

        arma::Cube<eT> output;
        GradientConvolutionRule::Convolution(inputSlices, deltaSlices, output);

        for (size_t i = 0; i < output.n_slices; i++)
          g.slice(s) += output.slice(i);
      }
    }
  }

  //! Get the weights.
  OutputDataType const& Weights() const { return weights; }
  //! Modify the weights.
  OutputDataType& Weights() { return weights; }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(weights, "weights");
    ar & data::CreateNVP(wfilter, "wfilter");
    ar & data::CreateNVP(hfilter, "hfilter");
    ar & data::CreateNVP(inMaps, "inMaps");
    ar & data::CreateNVP(outMaps, "outMaps");
    ar & data::CreateNVP(xStride, "xStride");
    ar & data::CreateNVP(yStride, "yStride");
    ar & data::CreateNVP(wPad, "wPad");
    ar & data::CreateNVP(hPad, "hPad");
  }

 private:
  /*
   * Rotates a 3rd-order tesor counterclockwise by 180 degrees.
   *
   * @param input The input data to be rotated.
   * @param output The rotated output.
   */
  template<typename eT>
  void Rotate180(const arma::Cube<eT>& input, arma::Cube<eT>& output)
  {
    output = arma::Cube<eT>(input.n_rows, input.n_cols, input.n_slices);

    // * left-right flip, up-down flip */
    for (size_t s = 0; s < output.n_slices; s++)
      output.slice(s) = arma::fliplr(arma::flipud(input.slice(s)));
  }

  /*
   * Rotates a dense matrix counterclockwise by 180 degrees.
   *
   * @param input The input data to be rotated.
   * @param output The rotated output.
   */
  template<typename eT>
  void Rotate180(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    // * left-right flip, up-down flip */
    output = arma::fliplr(arma::flipud(input));
  }

  /*
   * Return the convolution output size.
   *
   * @param size The size of the input (row or column).
   * @param k The size of the filter (width or height).
   * @param s The stride size (x or y direction).
   * @param p The size of the padding (width or height).
   * @return The convolution output size.
   */
  size_t ConvOutSize(const size_t size,
                     const size_t k,
                     const size_t s,
                     const size_t p)
  {
    return std::floor(size + p * 2 - k) / s + 1;
  }

  //! Locally-stored filter/kernel width.
  size_t wfilter;

  //! Locally-stored filter/kernel height.
  size_t hfilter;

  //! Locally-stored number of input maps.
  size_t inMaps;

  //! Locally-stored number of output maps.
  size_t outMaps;

  //! Locally-stored stride of the filter in x-direction.
  size_t xStride;

  //! Locally-stored stride of the filter in y-direction.
  size_t yStride;

  //! Locally-stored padding width.
  size_t wPad;

  //! Locally-stored padding height.
  size_t hPad;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class ConvLayer

//! Layer traits for the convolution layer.
template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
class LayerTraits<ConvLayer<ForwardConvolutionRule,
                            BackwardConvolutionRule,
                            GradientConvolutionRule,
                            InputDataType,
                            OutputDataType> >
{
 public:
  static const bool IsBinary = false;
  static const bool IsOutputLayer = false;
  static const bool IsBiasLayer = false;
  static const bool IsLSTMLayer = false;
  static const bool IsConnection = true;
};

} // namespace ann
} // namespace mlpack

#endif
