/**
 * @file methods/ann/models/yolov3/yolov3_layer.hpp
 * @author Andrew Furey
 *
 * Definition of the YOLOv3 layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_MODELS_YOLOV3_YOLOV3_LAYER_HPP
#define MLPACK_METHODS_ANN_MODELS_YOLOV3_YOLOV3_LAYER_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>

/**
 * Helper layer for YOLOv3. Used as the last layer to normalize bounding
 * boxes and classification probilities.
 *
 * Returns bounding boxes in YOLO format. Bounding boxes consist of
 * a center x and center y coordinate, width and height, objectness score
 * and class probabilities.
 *
 * This layer outputs `gridSize` x `gridSize` x `predictionsPerCell`
 * bounding boxes.
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
namespace mlpack {

template <typename MatType = arma::mat>
class YOLOv3Layer : public Layer<MatType>
{
 public:
  using ElemType = typename MatType::elem_type;

  using CubeType = typename GetCubeType<MatType>::type;

  YOLOv3Layer() { /* Nothing to do. */ }

  /**
   * YOLOv3Layer constructor.
   *
   * Input dimensions are expected to be 3d, normally from the outputs of a
   * convolution layer.
   *
   * @param imgSize The width and height of input images. Pretrained weights
       used 416.
   * @param numAttributes Total number of attributes representing a bounding
       box. The same as 5 plus the number of classes in the dataset.
   * @param predictionsPerCell Each YOLO layer predicts `predictionsPerCell`
       boxes per grid cell. Pretrained weights use 3.
   * @param anchors Vector of anchor width and heights. Formatted as
      [w0, h0, w1, h1, ... ]. Each anchors is a [w, h] pair. There must be
      `predictionsPerCell` anchor pairs.
   */
  YOLOv3Layer(const size_t imgSize,
              const size_t numAttributes,
              const size_t gridSize,
              const size_t predictionsPerCell,
              const std::vector<ElemType>& anchors);

  YOLOv3Layer* Clone() const override { return new YOLOv3Layer(*this); }

  // Copy the given YOLOv3Layer.
  YOLOv3Layer(const YOLOv3Layer& other);
  // Take ownership of the given YOLOv3Layer.
  YOLOv3Layer(YOLOv3Layer&& other);
  // Copy the given YOLOv3Layer.
  YOLOv3Layer& operator=(const YOLOv3Layer& other);
  // Take ownership of the given YOLOv3Layer.
  YOLOv3Layer& operator=(YOLOv3Layer&& other);

  void ComputeOutputDimensions() override;

  /**
   * NOTE: This will be changed when training is implemented.
   *
   * Takes in 3d input and outputs bounding boxes, based on anchors,
   * input image size and cell position, for each batch item.
   *
   * Bounding boxes are outputted in the format: x1, y1, x2, y2.
   *
   * @param input Input data representing outputs of model.
   * @param output Resulting bounding boxes after being normalized to image.
   */
  void Forward(const MatType& input, MatType& output) override;

  /**
   * NOTE: This will be changed when training is implemented.
   *
   * Currently not implemented.
   */
  void Backward(const MatType& input,
                const MatType& output,
                const MatType& gy,
                MatType& g) override;

  // Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  void GenerateAnchors()
  {
    anchorsW = MatType(grid, predictionsPerCell, arma::fill::none);
    anchorsH = MatType(grid, predictionsPerCell, arma::fill::none);

    for (size_t i = 0; i < predictionsPerCell; i++)
    {
      anchorsW.col(i).fill(anchors[i * 2]);
      anchorsH.col(i).fill(anchors[i * 2 + 1]);
    }
  }

  // Original input image size.
  size_t imgSize;
  // Number of attributes representing a bounding box.
  size_t numAttributes;
  // Width and height of grid, since grids must be square.
  // Should be equivalent to inputDimensions[0] and inputdDimensions[1].
  size_t gridSize;
  // Cached gridSize * gridSize
  size_t grid;
  // Vector of anchor pairs.
  std::vector<ElemType> anchors;
  // Number of bounding boxes per cell.
  size_t predictionsPerCell;
  // Matrix of anchor widths.
  MatType anchorsW;
  // Matrix of anchor height.
  MatType anchorsH;
};

} // namespace mlpack

#include "yolov3_layer_impl.hpp"

#endif
