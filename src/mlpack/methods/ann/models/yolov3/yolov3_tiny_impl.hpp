/**
 * @file methods/ann/models/yolov3/yolov3_tiny_impl.hpp
 * @author Andrew Furey
 *
 * Definition of the YOLOv3-tiny model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_MODELS_MODELS_YOLOV3_TINY_IMPL_HPP
#define MLPACK_METHODS_ANN_MODELS_MODELS_YOLOV3_TINY_IMPL_HPP

#include <mlpack/methods/ann/models/yolov3/yolov3_tiny.hpp>

namespace mlpack {

template <typename OutputLayerType,
          typename InitializationRuleType,
          typename MatType>
YOLOv3Tiny<
           OutputLayerType,
           InitializationRuleType,
           MatType
>::YOLOv3Tiny(const size_t imgSize,
              const size_t numClasses,
              const size_t predictionsPerCell,
              const size_t maxDetections,
              const std::vector<ElemType>& anchors) :
  imgSize(imgSize),
  predictionsPerCell(predictionsPerCell),
  numAttributes(numClasses + 5),
  maxDetections(maxDetections)
{
  if (anchors.size() != predictionsPerCell * 4)
  {
    std::ostringstream errMessage;
    errMessage << "YOLOv3Tiny::YOLOv3Tiny(): Expected " << predictionsPerCell *
      4 << " anchor points, but received " << anchors.size();
    throw std::logic_error(errMessage.str());
  }

  const size_t mid = predictionsPerCell * 2;
  numBoxes = (imgSize / 16) * (imgSize / 16) * predictionsPerCell +
             (imgSize / 32) * (imgSize / 32) * predictionsPerCell;

  const std::vector<ElemType>
    smallAnchors(anchors.begin(), anchors.begin() + mid),
    largeAnchors(anchors.begin() + mid, anchors.end());

  const std::vector<double> scaleFactor = { 2.0, 2.0 };

  model = Model();
  model.InputDimensions() = { imgSize, imgSize, 3 };

  size_t convolution0 = ConvolutionBlock(16, 3);
  size_t maxPool1 = MaxPool2x2(2);
  size_t convolution2 = ConvolutionBlock(32, 3);
  size_t maxPool3 = MaxPool2x2(2);
  size_t convolution4 = ConvolutionBlock(64, 3);
  size_t maxPool5 = MaxPool2x2(2);
  size_t convolution6 = ConvolutionBlock(128, 3);
  size_t maxPool7 = MaxPool2x2(2);
  size_t convolution8 = ConvolutionBlock(256, 3);
  size_t maxPool9 = MaxPool2x2(2);
  size_t convolution10 = ConvolutionBlock(512, 3);
  size_t maxPool11 = MaxPool2x2(1);
  size_t convolution12 = ConvolutionBlock(1024, 3);
  size_t convolution13 = ConvolutionBlock(256, 1);

  // Detection head for larger objects.
  size_t convolution14 = ConvolutionBlock(512, 3);
  size_t convolution15 =
    ConvolutionBlock(predictionsPerCell * numAttributes, 1, false);
  size_t detections16 =
    model.template Add<YOLOv3Layer<MatType>>(imgSize, numAttributes,
      imgSize / 32, predictionsPerCell, largeAnchors);

  size_t convolution17 = ConvolutionBlock(128, 1);
  // Upsample for more fine-grained detections.
  size_t upsample18 =
    model.template Add<NearestInterpolation<MatType>>(scaleFactor);

  // Detection head for smaller objects.
  size_t convolution19 = ConvolutionBlock(256, 3);
  size_t convolution20 =
    ConvolutionBlock(predictionsPerCell * numAttributes, 1, false);
  size_t detections21 =
    model.template Add<YOLOv3Layer<MatType>>(imgSize, numAttributes,
      imgSize / 16, predictionsPerCell, smallAnchors);


  // the DAGNetwork class requires one explicit output layer for concatenations,
  // so we use the Identity layer for pure concatentation, and no other compute.
  size_t concatLayer22 = model.template Add<Identity<MatType>>();

  model.Connect(convolution0, maxPool1);
  model.Connect(maxPool1, convolution2);
  model.Connect(convolution2, maxPool3);
  model.Connect(maxPool3, convolution4);
  model.Connect(convolution4, maxPool5);
  model.Connect(maxPool5, convolution6);
  model.Connect(convolution6, maxPool7);
  model.Connect(maxPool7, convolution8);

  model.Connect(convolution8, maxPool9);
  model.Connect(maxPool9, convolution10);
  model.Connect(convolution10, maxPool11);
  model.Connect(maxPool11, convolution12);
  model.Connect(convolution12, convolution13);

  model.Connect(convolution13, convolution14);
  model.Connect(convolution14, convolution15);
  model.Connect(convolution15, detections16);

  model.Connect(convolution13, convolution17);
  model.Connect(convolution17, upsample18);

  // Concat convolution8 + upsample18 => convolution19
  model.Connect(upsample18, convolution19);
  model.Connect(convolution8, convolution19);
  // Set axis not necessary, since default is concat along channels.

  model.Connect(convolution19, convolution20);
  model.Connect(convolution20, detections21);
  // Again, set axis not necessary, since default is concat along channels.

  // Concatenation order shouldn't matter.
  model.Connect(detections16, concatLayer22);
  model.Connect(detections21, concatLayer22);

  model.SetNetworkMode(false);
  model.Reset();
}

template <typename OutputLayerType,
          typename InitializationRuleType,
          typename MatType>
size_t YOLOv3Tiny<OutputLayerType, InitializationRuleType, MatType>
::ConvolutionBlock(const size_t maps,
                   const size_t kernel,
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
    maps, kernel, kernel, 1, 1, pad, pad, "none", !batchNorm);

  // set epsilon to zero, not used in darknet.
  if (batchNorm)
    block.template Add<BatchNorm<MatType>>(2, 2, 0, false);

  block.template Add<LeakyReLU<MatType>>(reluSlope);
  return model.Add(block);
}

template <typename OutputLayerType,
          typename InitializationRuleType,
          typename MatType>
size_t YOLOv3Tiny<OutputLayerType, InitializationRuleType, MatType>
::MaxPool2x2(const size_t stride)
{
  // All max pool layers have kernel size 2
  MultiLayer<MatType> block;
  if (stride == 1)
  {
    // One layer with odd input size, with kernel size 2, stride 1.
    // Padding on the right and bottom are needed.
    ElemType min = -arma::Datum<ElemType>::inf;
    block.template Add<Padding<MatType>>(0, 1, 0, 1, min);
  }
  block.template Add<MaxPooling<MatType>>(2, 2, stride, stride);
  return model.Add(block);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
template<typename Archive>
void YOLOv3Tiny<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(model));
  ar(CEREAL_NVP(imgSize));
  ar(CEREAL_NVP(predictionsPerCell));
  ar(CEREAL_NVP(numAttributes));
  ar(CEREAL_NVP(maxDetections));
  ar(CEREAL_NVP(numBoxes));
}

} // namespace mlpack

#endif
