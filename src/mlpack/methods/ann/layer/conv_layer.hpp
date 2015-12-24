/**
 * @file conv_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the ConvLayer class.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_CONV_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_CONV_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>
#include <mlpack/methods/ann/convolution_rules/border_modes.hpp>
#include <mlpack/methods/ann/convolution_rules/naive_convolution.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the ConvLayer class. The ConvLayer class represents a
 * single layer of a neural network.
 *
 * @tparam OptimizerType Type of the optimizer used to update the weights.
 * @tparam WeightInitRule Rule used to initialize the weight matrix.
 * @tparam ForwardConvolutionRule Convolution to perform forward process.
 * @tparam BackwardConvolutionRule Convolution to perform backward process.
 * @tparam GradientConvolutionRule Convolution to calculate gradient.
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    template<typename, typename> class OptimizerType = mlpack::ann::RMSPROP,
    class WeightInitRule = NguyenWidrowInitialization,
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
   * @param WeightInitRule The weight initialization rule used to initialize the
   *        weight matrix.
   */
  ConvLayer(const size_t inMaps,
            const size_t outMaps,
            const size_t wfilter,
            const size_t hfilter,
            const size_t xStride = 1,
            const size_t yStride = 1,
            const size_t wPad = 0,
            const size_t hPad = 0,
            WeightInitRule weightInitRule = WeightInitRule()) :
      wfilter(wfilter),
      hfilter(hfilter),
      inMaps(inMaps),
      outMaps(outMaps),
      xStride(xStride),
      yStride(yStride),
      wPad(wPad),
      hPad(hPad),
      optimizer(new OptimizerType<ConvLayer<OptimizerType,
                                            WeightInitRule,
                                            ForwardConvolutionRule,
                                            BackwardConvolutionRule,
                                            GradientConvolutionRule,
                                            InputDataType,
                                            OutputDataType>,
                                            OutputDataType>(*this)),
      ownsOptimizer(true)
  {
    weightInitRule.Initialize(weights, wfilter, hfilter, inMaps * outMaps);
  }

  /**
   * Delete the convolution layer object and its optimizer.
   */
  ~ConvLayer()
  {
    if (ownsOptimizer)
      delete optimizer;
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
   * @param d The calculated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Cube<eT>& d, arma::Cube<eT>& g)
  {
    g = arma::zeros<arma::Cube<eT> >(weights.n_rows, weights.n_cols,
        weights.n_slices);

    for (size_t outMap = 0; outMap < outMaps; outMap++)
    {
      for (size_t inMap = 0, s = outMap; inMap < inMaps; inMap++, s += outMaps)
      {
        arma::Cube<eT> inputSlices = inputParameter.slices(inMap, inMap);
        arma::Cube<eT> deltaSlices = d.slices(outMap, outMap);

        arma::Cube<eT> output;
        GradientConvolutionRule::Convolution(inputSlices, deltaSlices, output);

        for (size_t i = 0; i < output.n_slices; i++)
          g.slice(s) += output.slice(i);
      }
    }
  }

  //! Get the optimizer.
  OptimizerType<ConvLayer<OptimizerType,
                          WeightInitRule,
                          ForwardConvolutionRule,
                          BackwardConvolutionRule,
                          GradientConvolutionRule,
                          InputDataType,
                          OutputDataType>, OutputDataType>& Optimizer() const
  {
    return *optimizer;
  }
  //! Modify the optimizer.
  OptimizerType<ConvLayer<OptimizerType,
                          WeightInitRule,
                          ForwardConvolutionRule,
                          BackwardConvolutionRule,
                          GradientConvolutionRule,
                          InputDataType,
                          OutputDataType>, OutputDataType>& Optimizer()
  {
    return *optimizer;
  }

  //! Get the weights.
  OutputDataType& Weights() const { return weights; }
  //! Modify the weights.
  OutputDataType& Weights() { return weights; }

  //! Get the input parameter.
  InputDataType& InputParameter() const {return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const {return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const {return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType& Gradient() const {return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

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
  const size_t wfilter;

  //! Locally-stored filter/kernel height.
  const size_t hfilter;

  //! Locally-stored number of input maps.
  const size_t inMaps;

  //! Locally-stored number of output maps.
  const size_t outMaps;

  //! Locally-stored stride of the filter in x-direction.
  const size_t xStride;

  //! Locally-stored stride of the filter in y-direction.
  const size_t yStride;

  //! Locally-stored padding width.
  const size_t wPad;

  //! Locally-stored padding height.
  const size_t hPad;

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

  //! Locally-stored pointer to the optimzer object.
  OptimizerType<ConvLayer<OptimizerType,
                          WeightInitRule,
                          ForwardConvolutionRule,
                          BackwardConvolutionRule,
                          GradientConvolutionRule,
                          InputDataType,
                          OutputDataType>, OutputDataType>* optimizer;

  //! Parameter that indicates if the class owns a optimizer object.
  bool ownsOptimizer;
}; // class ConvLayer

//! Layer traits for the convolution layer.
template<
    template<typename, typename> class OptimizerType,
    typename WeightInitRule,
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
class LayerTraits<ConvLayer<OptimizerType,
                            WeightInitRule,
                            ForwardConvolutionRule,
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
