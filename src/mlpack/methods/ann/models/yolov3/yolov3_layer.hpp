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

namespace mlpack {

template <typename MatType = arma::mat>
class YOLOv3Layer : public Layer<MatType>
{
 public:
  YOLOv3Layer() { /* Nothing to do. */ }

  YOLOv3Layer(const size_t imgSize,
              const size_t numAttributes,
              const size_t gridSize,
              const size_t predictionsPerCell,
              const std::vector<typename MatType::elem_type> anchors);

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

  // Output format: cx, cy, w, h
  void Forward(const MatType& input, MatType& output) override;

  void Backward(const MatType& input,
                const MatType& output,
                const MatType& gy,
                MatType& g) override;

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  using Type = typename MatType::elem_type;

  using CubeType = typename GetCubeType<MatType>::type;

  size_t imgSize;

  size_t numAttributes;

  size_t gridSize;
  // Cached gridSize * gridSize
  size_t grid;

  MatType anchorsW;

  MatType anchorsH;

  size_t predictionsPerCell;
};

} // namespace mlpack

#include "yolov3_layer_impl.hpp"

#endif
