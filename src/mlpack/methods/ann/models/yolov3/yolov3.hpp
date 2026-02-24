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
template <typename MatType = arma::fmat,
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
   * @param imgSize The width and height of preprocessed images.
   * @param numClasses The number of output classes. Pretrained weights were
       trained on COCO which has 80 classes.
   * @param anchors Vector of anchor width and heights. Formatted as
      [w0, h0, w1, h1, ... ]. Each anchors is a [w, h] pair. There must be 3 * 3
      anchors, since YOLOv3 has three output layers and makes 3 predictions
      for each cell per layer. Therefore, anchors.size() must be 3 * 6 = 18.
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
   * Ordinary feed forward pass of the network. Get raw outputs from the
   * model, with optional preprocessing done via the `preprocess` argument.
   *
   * @param input Input data used for evaluating the specified function.
      The input matrix dimensions should be (imgSize * imgSize * 3, batchSize).
   * @param output Resulting bounding boxes and classification probabilities.
      The bounding boxes are represented as (cx, cy, w, h) where (cx, cy) points
      to the center of the box. The bounding boxes are normalized based on the
      `imgSize`.
   */

  void Predict(MatType& image,
               const ImageOptions& opts,
               const double ignoreThreshold = 0.7)
  {
    MatType preprocessed, rawOutput;
    PreprocessImage(image, opts, preprocessed);
    model.Predict(preprocessed, rawOutput);
    DrawBoundingBoxes(rawOutput, image, opts, ignoreThreshold);
  }

  void Predict(const MatType& image,
               const ImageOptions& opts,
               MatType& rawOutput,
               const bool preprocess = false)
  {
    MatType preprocessed;
    if (preprocess)
    {
      PreprocessImage(image, opts, preprocessed);
      model.Predict(preprocessed, rawOutput);
    }
    else
    {
      model.Predict(image, rawOutput);
    }
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
   * Read the raw output of the model and draw the bounding boxes onto the
   * given `image`. If the bounding boxes confidence is greater than
   * `ignoreThresh` it will be drawn onto the image.
   */
  void DrawBoundingBoxes(const MatType& rawOutput,
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

  // DAGNetwork containing the graph of the YOLOv3 model
  ModelType model;
  // Width and height of input image
  size_t imgSize;
  // Number of output classes + 5 for (x, y, w, h, objectness)
  size_t numAttributes;
  // Class names for each possible object the model can identify.
  std::vector<std::string> classNames;
};

} // namespace mlpack

#include "yolov3_impl.hpp"

#endif
