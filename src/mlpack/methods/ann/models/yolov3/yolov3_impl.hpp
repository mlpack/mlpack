/**
 * @file methods/ann/models/yolov3/yolov3_tiny_impl.hpp
 * @author Andrew Furey
 *
 * Definition of the YOLOv3 model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_MODELS_MODELS_YOLOV3_IMPL_HPP
#define MLPACK_METHODS_ANN_MODELS_MODELS_YOLOV3_IMPL_HPP

#include <mlpack/methods/ann/models/yolov3/yolov3.hpp>

namespace mlpack {

template <typename OutputLayerType,
          typename InitializationRuleType,
          typename MatType>
YOLOv3<
           OutputLayerType,
           InitializationRuleType,
           MatType
>::YOLOv3(const size_t imgSize,
          const size_t numClasses,
          const size_t predictionsPerCell,
          const std::vector<ElemType>& anchors) :
  imgSize(imgSize),
  predictionsPerCell(predictionsPerCell),
  numAttributes(numClasses + 5),
{
  if (anchors.size() != predictionsPerCell * 6)
  {
    std::ostringstream errMessage;
    errMessage << "YOLOv3::YOLOv3(): Expected " << predictionsPerCell *
      6 << " anchor points, but received " << anchors.size();
    throw std::logic_error(errMessage.str());
  }

  const size_t anchors1 = predictionsPerCell * 2;
  const size_t anchors2 = predictionsPerCell * 4;

  const std::vector<ElemType>
    smallAnchors(anchors.begin(), anchors.begin() + anchors1),
    middleAnchors(anchors.begin() + anchors1, anchors.begin() + anchors2),
    largeAnchors(anchors.begin() + anchors2, anchors.end());

  const std::vector<double> scaleFactor = { 2.0, 2.0 };

  model = Model();
  model.InputDimensions() = { imgSize, imgSize, 3 };

  size_t convolution0 = ConvolutionBlock(32, 3);

  size_t layer4 = DownsampleBlock(convolution0, 64, 1);
  size_t layer11 = DownsampleBlock(layer4, 128, 2);
  size_t layer36 = DownsampleBlock(layer11, 256, 8);
  size_t layer61 = DownsampleBlock(layer36, 512, 8);
  size_t layer74 = DownsampleBlock(layer61, 1024, 4);

  size_t layer75 = ConvolutionBlock(512, 1);
  size_t layer76 = ConvolutionBlock(1024, 3);
  size_t layer77 = ConvolutionBlock(512, 1);
  size_t layer78 = ConvolutionBlock(1024, 3);
  size_t layer79 = ConvolutionBlock(512, 1);
  size_t layer80 = ConvolutionBlock(1024, 3);
  size_t layer81 =
    ConvolutionBlock(numAttributes * predictionsPerCell, 1, 1, false);

  model.Connect(layer74, layer75);
  model.Connect(layer75, layer76);
  model.Connect(layer76, layer77);
  model.Connect(layer77, layer78);
  model.Connect(layer78, layer79);
  model.Connect(layer79, layer80);
  model.Connect(layer80, layer81);

  size_t layer82 = ConvolutionBlock(256, 1);
  size_t upsample82 =
    model.template Add<mlpack::NearestInterpolation<MatType>>(scaleFactor);
  model.Connect(layer79, layer82);
  model.Connect(layer82, upsample82);

  size_t layer84 = ConvolutionBlock(256, 1);
  size_t layer85 = ConvolutionBlock(512, 3);
  size_t layer86 = ConvolutionBlock(256, 1);
  size_t layer87 = ConvolutionBlock(512, 3);
  size_t layer88 = ConvolutionBlock(256, 1);
  size_t layer89 = ConvolutionBlock(512, 3);
  size_t layer90 =
    ConvolutionBlock(numAttributes * predictionsPerCell, 1, 1, false);

  // Concat
  model.Connect(upsample82, layer84);
  model.Connect(layer61, layer84); // default is concat along last axis.

  model.Connect(layer84, layer85);
  model.Connect(layer85, layer86);
  model.Connect(layer86, layer87);
  model.Connect(layer87, layer88);
  model.Connect(layer88, layer89);
  model.Connect(layer89, layer90);

  size_t layer91 = ConvolutionBlock(128, 1);
  size_t upsample91
    = model.template Add<mlpack::NearestInterpolation<MatType>>(scaleFactor);
  model.Connect(layer88, layer91);
  model.Connect(layer91, upsample91);

  size_t layer93 = ConvolutionBlock(128, 1);
  size_t layer94 = ConvolutionBlock(256, 3);
  size_t layer95 = ConvolutionBlock(128, 1);
  size_t layer96 = ConvolutionBlock(256, 3);
  size_t layer97 = ConvolutionBlock(128, 1);
  size_t layer98 = ConvolutionBlock(256, 3);
  size_t layer99 = ConvolutionBlock(255, 1, 1, false); // coco

  // Concat
  model.Connect(upsample91, layer93);
  model.Connect(layer36, layer93); // default is concat along last axis.

  model.Connect(layer93, layer94);
  model.Connect(layer94, layer95);
  model.Connect(layer95, layer96);
  model.Connect(layer96, layer97);
  model.Connect(layer97, layer98);
  model.Connect(layer98, layer99);

  size_t detection0 =
    model.template Add<YOLOv3Layer<MatType>>(imgSize, numAttributes,
      imgSize / 32, predictionsPerCell, largeAnchors);

  size_t detection1 =
    model.template Add<YOLOv3Layer<MatType>>(imgSize, numAttributes,
      imgSize / 16, predictionsPerCell, middleAnchors);

  size_t detection2 =
    model.template Add<YOLOv3Layer<MatType>>(imgSize, numAttributes,
      imgSize / 8, predictionsPerCell, smallAnchors);

  model.Connect(layer81, detection0);
  model.Connect(layer90, detection1);
  model.Connect(layer99, detection2);

  // Concat outputs.
  size_t lastLayer = model.template Add<mlpack::Identity>();
  model.Connect(detection0, lastLayer);
  model.Connect(detection1, lastLayer);
  model.Connect(detection2, lastLayer);

  model.SetNetworkMode(false);
  model.Reset();
}

template <typename OutputLayerType,
          typename InitializationRuleType,
          typename MatType>
size_t YOLOv3<OutputLayerType, InitializationRuleType, MatType>
::ConvolutionBlock(const size_t maps,
                   const size_t kernel,
                   const size_t stride,
                   const bool batchNorm,
                   const ElemType reluSlope)
{
  if (kernel != 3 && kernel != 1)
  {
    std::ostringstream errMessage;
    errMessage << "Kernel size for convolutions in yolov3-tiny must be 3"
        "or 1, but you gave " << kernel;
    throw std::logic_error(errMessage.str());
  }

  size_t pad = kernel == 3 ? 1 : 0;
  MultiLayer<MatType> block;
  block.template Add<Convolution<MatType>>(
    maps, kernel, kernel, stride, stride, pad, pad, "none", !batchNorm);

  // set epsilon to zero, not used in darknet.
  if (batchNorm)
    block.template Add<BatchNorm<MatType>>(2, 2, 0, false);

  block.template Add<LeakyReLU<MatType>>(reluSlope);
  return model.Add(block);
}

template <typename OutputLayerType,
          typename InitializationRuleType,
          typename MatType>
size_t YOLOv3<OutputLayerType, InitializationRuleType, MatType>
::DownsampleBlock(const size_t previousLayer,
                  const size_t maps,
                  const size_t shortcuts)
{
  if (shortcuts == 0)
  {
    throw std::logic_error("YOLOv3::DownsampleBlock(): Number of shortcuts"
        " must be greater than zero.");
  }

  size_t convolution0 = ConvolutionBlock(maps, 3, 2);
  model.Connect(previousLayer, convolution0);

  size_t previous = convolution0;
  for (size_t i = 0; i < shortcuts; i++)
  {
    size_t convolution1 = ConvolutionBlock(maps / 2, 1);
    size_t convolution2 = ConvolutionBlock(maps, 3);

    size_t residual = model.template Add<mlpack::Identity>();
    model.SetConnection(residual, ADDITION);

    model.Connect(previous, convolution1);
    model.Connect(convolution1, convolution2);
    model.Connect(convolution2, residual);
    model.Connect(previous, residual);

    previous = residual;
  }

  return previous;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
template<typename Archive>
void YOLOv3<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(model));
  ar(CEREAL_NVP(imgSize));
  ar(CEREAL_NVP(predictionsPerCell));
  ar(CEREAL_NVP(numAttributes));
}

} // namespace mlpack

#endif
