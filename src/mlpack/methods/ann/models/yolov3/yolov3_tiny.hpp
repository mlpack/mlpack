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
 * For more information, refer to the following paper:
 *
 * @code
 * @article{yolov3,
 *   title     ={YOLOv3: An Incremental Improvement},
 *   author    ={Redmon, Joseph and Farhadi, Ali},
 *   journal   = {arXiv},
 *   year      ={2018}
 * }
 * @endcode
 *
 */
template <typename OutputLayerType = EmptyLoss,
          typename InitializationRuleType = RandomInitialization,
          typename MatType = arma::mat>
class YOLOv3tiny
{
 public:

  // Helper types.
  using ModelType =
    DAGNetwork<OutputLayerType, InitializationRuleType, MatType>;
  using Type = typename MatType::elem_type;
  using CubeType = typename GetCubeType<MatType>::type;

  YOLOv3tiny() { /* Nothing to do. */ }

  YOLOv3tiny(const size_t imgSize,
             const size_t numClasses,
             const size_t predictionsPerCell,
             const std::vector<Type>& anchors);

  ~YOLOv3tiny() { /* Nothing to do. */ }

  ModelType& Model() { return model; }

  void Predict(const MatType& input, MatType& output)
  {
    model.Predict(input, output);
  }

 private:

  size_t ConvolutionBlock(const size_t maps,
                          const size_t kernel,
                          const bool batchNorm = true,
                          const Type reluSlope = 0.1);

  size_t MaxPool2x2(const size_t stride);

  size_t YOLO(const size_t imgSize,
              const size_t gridSize,
              const std::vector<Type>& anchors);

  ModelType model;
  size_t imgSize;
  size_t predictionsPerCell;
  size_t numAttributes;
  MatType parameters;
};

} // namespace mlpack

#include "yolov3_tiny_impl.hpp"

#endif
