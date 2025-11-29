/**
 * @file methods/ann/models/yolov3_tiny.hpp
 * @author Andrew Furey
 *
 * Definition of the YOLOv3-tiny model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_MODELS_YOLOV3_TINY_HPP
#define MLPACK_METHODS_ANN_MODELS_YOLOV3_TINY_HPP

#include <mlpack/prereqs.hpp>

#include <mlpack/methods/ann/dag_network.hpp>
#include <mlpack/methods/ann/loss_functions/loss_functions.hpp>
#include <mlpack/methods/ann/init_rules/init_rules.hpp>

#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/models/yolov3/yolov3_layer.hpp>

namespace mlpack {

/**
 * YOLOv3-tiny is a small one-stage object detection model.
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
template <typename OutputLayerType = EmptyLoss,
          typename InitializationRuleType = RandomInitialization,
          typename MatType = arma::mat>
class YOLOv3Tiny
{
 public:
  // Helper types.
  using ModelType =
    DAGNetwork<OutputLayerType, InitializationRuleType, MatType>;
  using ElemType = typename MatType::elem_type;
  using CubeType = typename GetCubeType<MatType>::type;

  YOLOv3Tiny() { /* Nothing to do. */ }

  /**
   * Create the YOLOv3Tiny model.
   *
   * @param imgSize The width and height of input images. Pretrained weights
       used 416.
   * @param numClasses The number of output classes. Pretrained weights were
       trained on COCO which has 80 classes.
   * @param predictionsPerCell Each YOLO layer predicts `predictionsPerCell`
       boxes per grid cell. Pretrained weights use 3.
   * @param anchors Vector of anchor width and heights. Formatted as
      [w0, h0, w1, h1, ... ]. Each anchors is a [w, h] pair. There must be
      predictionsPerCell * 2 anchors, since YOLOv3Tiny has two output layers.
      Therefore, anchors.size() must be predictionsPerCell * 4.
   */
  YOLOv3Tiny(const size_t imgSize,
             const size_t numClasses,
             const size_t predictionsPerCell,
             const size_t maxDetections,
             const std::vector<ElemType>& anchors);

  ~YOLOv3Tiny() { /* Nothing to do. */ }

  /**
   * Returns the graph representation of the model.
   */
  ModelType& Model() { return model; }

  /**
   * Ordinary feed forward pass of the network.
   *
   * @param input Input data used for evaluating the specified function.
      The input matrix dimensions should be (imgSize * imgSize, batchSize).
   * @param output Resulting bounding boxes.
   * @param numDetections Number of bounding boxes in output per batch.
   */
  void Predict(const MatType& input,
               MatType& output,
               arma::ucolvec& numDetections)
  {
    const size_t batchSize = input.n_cols;
    MatType batchedOutput;
    model.Predict(input, batchedOutput);

    CubeType batchedOutputAlias;
    MakeAlias(batchedOutputAlias, batchedOutput,
              numAttributes, numBoxes, batchSize);

    output = MatType(numAttributes * maxDetections, batchSize,
                     arma::fill::zeros);
    CubeType outputAlias;
    MakeAlias(outputAlias, output,
              numAttributes, maxDetections, batchSize);

    numDetections = arma::ucolvec(batchSize, arma::fill::zeros);
    for (size_t i = 0; i < batchSize; i++)
    {
      arma::ucolvec indices;

      const MatType& bboxes = batchedOutputAlias.slice(i);
      const MatType& confs = batchedOutputAlias.slice(i).row(4).t();
      NMS<true>::Evaluate<MatType, MatType, arma::ucolvec>
        (bboxes, confs, indices);

      numDetections(i) = std::min<size_t>(maxDetections, indices.n_rows);
      for (size_t j = 0; j < numDetections(i); j++)
      {
        outputAlias.slice(i).col(j) =
          batchedOutputAlias.slice(i).col(indices(j));
      }
    }
  }

  // Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * Adds a MultiLayer to the internal DAGNetwork. The MultiLayer includes
   * a Convolutions, BatchNorm (if batchNorm is true) and LeakyReLU.
   * If batchNorm is true, the convolution layer will not have a bias,
   * otherwise it will.
   * 
   * The convolution kernel size must be 3 or 1. If the kernel size is 3,
   * padding will be added.
   *
   * @param maps Number of output maps of the convolution layer.
   * @param kernel Size of the convolution kernel
   * @param batchNorm Boolean for including a batchnorm layer.
   * @param reluSlope Slope used in LeakyReLU. Default is 0.1 because
      pretrained weights used 0.1.
   */
  size_t ConvolutionBlock(const size_t maps,
                          const size_t kernel,
                          const bool batchNorm = true,
                          const ElemType reluSlope = 0.1);

  /**
   * Adds a MultiLayer to the internal DAGNetwork. The MultiLayer includes
   * a MaxPooling layer and an optional Padding layer depending on the stride
   * size.
   *
   * @param stride Stride of the MaxPooling kernel.
   */
  size_t MaxPool2x2(const size_t stride);

  // DAGNetwork containing the graph of the YOLOv3Tiny model
  ModelType model;
  // Width and height of input image
  size_t imgSize;
  // Predictions per cell for each YOLO layer
  size_t predictionsPerCell;
  // Number of output classes + 5 for (x, y, w, h, objectness)
  size_t numAttributes;
  // Max detections used during batched inference, because NMS returns variable
  // number of outputs.
  size_t maxDetections;
  // Total number of boxes
  size_t numBoxes;
};

} // namespace mlpack

#include "yolov3_tiny_impl.hpp"

#endif
