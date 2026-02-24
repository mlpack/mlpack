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

template <typename MatType,
          typename OutputLayerType,
          typename InitializationRuleType>
YOLOv3<MatType, OutputLayerType, InitializationRuleType>
::YOLOv3(const size_t imgSize,
         const std::vector<ElemType>& anchors,
         const std::vector<std::string>& classNames) :
  model(),
  imgSize(imgSize),
  numAttributes(classNames.size() + 5),
  classNames(classNames),
  anchors(anchors)
{
  const size_t predictionsPerCell = 3;
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
  size_t layer99 =
    ConvolutionBlock(predictionsPerCell * numAttributes, 1, 1, false);

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

  if (model.OutputDimensions()[0] != numAttributes)
  {
    std::ostringstream errMessage;
    errMessage << "YOLOv3::YOLOv3(): Expected number of attributes (" << numAttributes
      << ") does not match the output number of attributes ("
      << model.OutputDimensions()[0] << ")";
    throw std::logic_error(errMessage.str());
  }
}

template <typename MatType,
          typename OutputLayerType,
          typename InitializationRuleType>
size_t YOLOv3<MatType, OutputLayerType, InitializationRuleType>
::ConvolutionBlock(const size_t maps,
                   const size_t kernel,
                   const size_t stride,
                   const bool batchNorm,
                   const ElemType reluSlope)
{
  if (kernel != 3 && kernel != 1)
  {
    std::ostringstream errMessage;
    errMessage << "YOLOv3::ConvolutionBlock(): Kernel size for convolutions in yolov3-tiny must be 3"
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

template <typename MatType,
          typename OutputLayerType,
          typename InitializationRuleType>
size_t YOLOv3<MatType, OutputLayerType, InitializationRuleType>
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

template <typename MatType,
          typename OutputLayerType,
          typename InitializationRuleType>
template <typename Archive>
void YOLOv3<MatType, OutputLayerType, InitializationRuleType>
::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(model));
  ar(CEREAL_NVP(imgSize));
  ar(CEREAL_NVP(numAttributes));
  ar(CEREAL_NVP(classNames));
}

template <typename MatType,
          typename OutputLayerType,
          typename InitializationRuleType>
void YOLOv3<MatType, OutputLayerType, InitializationRuleType>
::DrawBoundingBoxes(const MatType& rawOutput,
                    MatType& image,
                    const ImageOptions& opts,
                    const double ignoreThresh)
{
  // Helpers
  const size_t numBoxes = model.OutputDimensions()[1];
  const size_t size = opts.Width() * opts.Height() * opts.Channels();

  const size_t numClasses = classNames.size();
  const arma::Col<ElemType> red = {255.0f, 0, 0};

  const size_t width = opts.Width();
  const size_t height = opts.Height();

  arma::ucolvec nmsIndices;

  ElemType ratio;
  ElemType xOffset = 0;
  ElemType yOffset = 0;

  if (width > height)
  {
    // landscape
    ratio =  (ElemType)width / imgSize;
    yOffset = (imgSize - (height * imgSize / (ElemType)width)) / 2;
  }
  else
  {
    // portrait
    ratio =  (ElemType)height / imgSize;
    xOffset = (imgSize - (width * imgSize / (ElemType)height)) / 2;
  }

  // Draw bounding boxes using NMS for each image in the batch.
  for (size_t batch = 0; batch < rawOutput.n_cols; batch++)
  {
    MatType rawOutputAlias, imageAlias;

    MakeAlias(rawOutputAlias, rawOutput, numAttributes, numBoxes,
        rawOutput.n_rows * batch);
    MakeAlias(imageAlias, image, size, 1, size * batch);

    const MatType& bboxes = rawOutputAlias.submat(0, 0, 3, numBoxes - 1);
    const MatType& objectness = rawOutputAlias.submat(4, 0, 4, numBoxes - 1);
    const MatType& confs =
      rawOutputAlias.submat(5, 0, numAttributes - 1, numBoxes - 1);

    // NMS on objectness, not class confidences. Will produce false negatives.
    NMS<>::Evaluate(bboxes, objectness, nmsIndices, 0.4);

    MatType chosenBoxes = bboxes.cols(nmsIndices);
    MatType classConfs = MatType(confs.cols(nmsIndices)).each_row() %
      objectness.cols(nmsIndices);
    arma::umat chosenConfs = arma::index_max(classConfs, 0);

    for (size_t b = 0; b < nmsIndices.n_rows; b++)
    {
      const size_t chosenClass = chosenConfs.at(0, b);
      if (classConfs.at(chosenClass, b) < ignoreThresh)
        continue;

      const std::string& label = classNames[chosenClass];
      const ElemType x1 =
        (chosenBoxes.at(0, b) - chosenBoxes.at(2, b) / 2 - xOffset) * ratio;
      const ElemType y1 =
        (chosenBoxes.at(1, b) - chosenBoxes.at(3, b) / 2 - yOffset) * ratio;
      const ElemType x2 =
        (chosenBoxes.at(0, b) + chosenBoxes.at(2, b) / 2 - xOffset) * ratio;
      const ElemType y2 =
        (chosenBoxes.at(1, b) + chosenBoxes.at(3, b) / 2 - yOffset) * ratio;

      const arma::Col<ElemType> bbox = arma::Col<ElemType>({x1, y1, x2, y2});
      BoundingBoxImage(imageAlias, opts, bbox, red, 1, label, 2);
    }
  }
}

} // namespace mlpack

#endif
