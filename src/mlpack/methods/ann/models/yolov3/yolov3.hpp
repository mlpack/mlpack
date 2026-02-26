/**
 * @file methods/ann/models/yolov3.hpp
 * @author Andrew Furey
 *
 * Definition of the YOLOv3 model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_MODELS_YOLOV3_HPP
#define MLPACK_METHODS_ANN_MODELS_YOLOV3_HPP

#include <mlpack/prereqs.hpp>

#include <mlpack/methods/ann/dag_network.hpp>
#include <mlpack/methods/ann/loss_functions/loss_functions.hpp>
#include <mlpack/methods/ann/init_rules/init_rules.hpp>

#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/models/yolov3/yolov3_layer.hpp>

#include <mlpack/core/data/image_letterbox.hpp>
#include <mlpack/core/data/image_bounding_box.hpp>

namespace mlpack {

/**
 * YOLOv3 is a one-stage object detection model.
 *
 * The input to the model must be a square image. Look at image_letterbox.hpp
 * to preprocess images before inference.
 *
 * The output of the model is a matrix. The rows are data points per
 * bounding box (including x, y, w, h, objectness, and classifications).
 * The columns represent each bounding box.
 *
 * For more information, refer to the following paper:
 *
 * @code
 * @article{yolov3,
 *   title     = {YOLOv3: An Incremental Improvement},
 *   author    = {Redmon, Joseph and Farhadi, Ali},
 *   journal   = {arXiv},
 *   year      = {2018}
 * }
 * @endcode
 *
 */
template <typename MatType = arma::mat,
          typename OutputLayerType = EmptyLossType<MatType>,
          typename InitializationRuleType = RandomInitialization>
class YOLOv3
{
 public:
  // Helper types.
  using ModelType =
    DAGNetwork<OutputLayerType, InitializationRuleType, MatType>;
  using ElemType = typename MatType::elem_type;
  using CubeType = typename GetCubeType<MatType>::type;

  YOLOv3() { /* Nothing to do. */ }

  /**
   * Create the YOLOv3 model.
   *
   * imgSize The width and height of preprocessed images.
   * anchors Vector of anchor width and heights. Formatted as
      [w0, h0, w1, h1, ... ]. Each anchors is a [w, h] pair. There must be 3 * 3
      anchors, since YOLOv3 has three output layers and makes 3 predictions
      for each cell per layer. Therefore, anchors.size() must be 3 * 6 = 18.
   * classNames Vector of strings where each string is a name corresponding
      to a class the model can predict.
   */
  YOLOv3(const size_t imgSize,
         const std::vector<ElemType>& anchors,
         const std::vector<std::string>& classNames);

  ~YOLOv3() { /* Nothing to do. */ }

  /**
   * Returns the graph representation of the model.
   */
  ModelType& Model() const { return model; }

  /**
   * Returns the width and height of the preprocessed image that
   * gets passed into the network.
   */
  size_t ImageSize() const { return imgSize; }

  /**
   * Returns the number of classes a bounding box may be. The pretrained
   * weights were trained on the COCO dataset, which conatins 80 different
   * classes.
   */
  size_t NumClasses() const { return numAttributes - 5; }

  /**
   * Returns the classes the model can identify.
   */
  const std::vector<std::string>& ClassNames() { return classNames; }

  /**
   * Returns the anchors used in the YOLOv3 layers.
   */
  const std::vector<ElemType>& Anchors() { return anchors; }

  /**
   * Ordinary feed forward pass of the network. Preprocesses image
   * and writes to `output`, which can optionally be the raw outputs
   * of the model, or a copy of `image` with the bounding boxes
   * drawn onto it.
   *
   * The image is expected to contain pixel values between 0-255.
   */
  void Predict(const MatType& image,
               const ImageOptions& opts,
               MatType& output,
               const bool drawBoxes = false,
               const double ignoreThreshold = 0.7)
  {
    MatType preprocessed;
    PreprocessImage(image, opts, preprocessed);

    if (drawBoxes)
    {
      MatType rawOutput;
      output = image;
      model.Predict(preprocessed, rawOutput);
      DrawBoundingBoxes(rawOutput, output, opts, ignoreThreshold);
    }
    else
    {
      model.Predict(preprocessed, output);

      // Update coordinates to be in the original image space.
      const size_t numBoxes = model.OutputDimensions()[1];
      ElemType ratio, xOffset, yOffset;
      FixBoundingBoxes(opts.Width(), opts.Height(), ratio, xOffset, yOffset);
      CubeType outputAlias;
      MakeAlias(outputAlias, output, numAttributes, numBoxes, output.n_cols);

      outputAlias.row(0) = (outputAlias.row(0) - xOffset) * ratio;
      outputAlias.row(1) = (outputAlias.row(1) - yOffset) * ratio;
      outputAlias.row(2) = outputAlias.row(2) * ratio;
      outputAlias.row(3) = outputAlias.row(3) * ratio;
    }
  }

  /**
   * Ordinary feed forward pass of the network.
   *
   * Expects preprocessing to be done by the user. Returns the raw
   * outputs of the model.
   */
  void Predict(const MatType& preprocessedInput,
               MatType& rawOutput)
  {
    model.Predict(preprocessedInput, rawOutput);
  }

  // Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * Adds a MultiLayer to the internal DAGNetwork. The MultiLayer includes
   * a Convolution, BatchNorm (if batchNorm is true) and LeakyReLU.
   * If batchNorm is true, the convolution layer will not have a bias,
   * otherwise it will.
   * 
   * The convolution kernel size must be 3 or 1. If the kernel size is 3,
   * padding will be added.
   *
   * @param maps Number of output maps of the convolution layer.
   * @param kernel Size of the convolution kernel.
   * @param stride Size of the convolution stride.
   * @param batchNorm Boolean for including a batchnorm layer.
   * @param reluSlope Slope used in LeakyReLU. Default is 0.1 because
      pretrained weights used 0.1.
   */
  size_t ConvolutionBlock(const size_t maps,
                          const size_t kernel,
                          const size_t stride = 1,
                          const bool batchNorm = true,
                          const ElemType reluSlope = 0.1);

  /**
   * Adds a MultiLayer to the internal DAGNetwork. The MultiLayer includes
   * a Convolutions, BatchNorm (if batchNorm is true) and LeakyReLU.
   * If batchNorm is true, the convolution layer will not have a bias,
   * otherwise it will.
   * 
   * The convolution kernel size must be 3 or 1. If the kernel size is 3,
   * padding will be added.
   *
   * @param previousLayer
   * @param maps Number of output maps of the convolution layer.
   * @param shortcuts
   */
  size_t DownsampleBlock(const size_t previousLayer,
                         const size_t maps,
                         const size_t shortcuts);

  /**
   * Preprocesses the `input` image and writes to `output`. The steps for
   * `YOLOv3` are normalize pixel values to be 0-1, letterbox the image
   * then group the channels for convolutions.
   */
  void PreprocessImage(const MatType& input,
                       const ImageOptions& opts,
                       MatType& output) const
  {
    MatType preprocessed = input / 255.0;
    ImageOptions preprocessedOpt = opts;
    LetterboxImages(preprocessed, preprocessedOpt, imgSize, imgSize, 0.5);
    output = GroupChannels(preprocessed, preprocessedOpt);
  }

  /**
   * Used to fix bounding boxes when the input images use the `LetterboxImages`
   * transform. Writes to `ratio`, `xOffset` and `yOffset`.
   */
  void FixBoundingBoxes(const size_t width,
                        const size_t height,
                        ElemType& ratio,
                        ElemType& xOffset,
                        ElemType& yOffset)
  {
    xOffset = 0;
    yOffset = 0;

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
  }

  /**
   * Read the raw output of the model and draw the bounding boxes onto the
   * given `image`. If the bounding boxes confidence is greater than
   * `ignoreThresh` it will be drawn onto the image.
   */
  void DrawBoundingBoxes(const MatType& rawOutput,
                         MatType& image,
                         const ImageOptions& opts,
                         const double ignoreThresh);

  // DAGNetwork containing the graph of the YOLOv3 model
  ModelType model;
  // Width and height of input image
  size_t imgSize;
  // Number of output classes + 5 for (x, y, w, h, objectness)
  size_t numAttributes;
  // Class names for each possible object the model can identify.
  std::vector<std::string> classNames;
  // Anchors for YOLOv3Layer for easy retrieval
  std::vector<ElemType> anchors;
};

} // namespace mlpack

#include "yolov3_impl.hpp"

#endif
