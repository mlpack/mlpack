/**
 * @file methods/ann/models/yolov3/yolov3_layer_impl.hpp
 * @author Andrew Furey
 *
 * Implementation of the YOLOv3 layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_MODELS_YOLOV3_YOLOV3_LAYER_IMPL_HPP
#define MLPACK_METHODS_ANN_MODELS_YOLOV3_YOLOV3_LAYER_IMPL_HPP

#include "yolov3_layer.hpp"

namespace mlpack {

template <typename MatType>
YOLOv3Layer<MatType>::YOLOv3Layer(
    const size_t imgSize,
    const size_t numAttributes,
    const size_t gridSize,
    const size_t predictionsPerCell,
    const std::vector<typename MatType::elem_type> anchors) :
    Layer<MatType>(),
    imgSize(imgSize),
    numAttributes(numAttributes),
    gridSize(gridSize),
    grid(gridSize * gridSize),
    predictionsPerCell(predictionsPerCell)
{
  if (anchors.size() != 2 * predictionsPerCell)
  {
    std::ostringstream errMessage;
    errMessage << "YOLOv3 must have " << predictionsPerCell
                << " (w, h) anchors but you gave "
                << anchors.size() / 2 << ".";
    throw std::logic_error(errMessage.str());
  }

  anchorsW = MatType(grid, predictionsPerCell, arma::fill::none);
  anchorsH = MatType(grid, predictionsPerCell, arma::fill::none);

  // Could maybe use .each_row()?
  for (size_t i = 0; i < predictionsPerCell; i++)
  {
    anchorsW.col(i).fill(anchors[i * 2]);
    anchorsH.col(i).fill(anchors[i * 2 + 1]);
  }
}

template<typename MatType>
YOLOv3Layer<MatType>::
YOLOv3Layer(const YOLOv3Layer& other) :
    Layer<MatType>(),
    imgSize(imgSize),
    numAttributes(numAttributes),
    gridSize(gridSize),
    anchorsW(anchorsW),
    anchorsH(anchorsH),
    predictionsPerCell(predictionsPerCell)
{
  // Nothing to do here.
}

template<typename MatType>
YOLOv3Layer<MatType>::
YOLOv3Layer(YOLOv3Layer&& other) :
    Layer<MatType>(std::move(other)),
    imgSize(std::move(imgSize)),
    numAttributes(std::move(numAttributes)),
    gridSize(std::move(gridSize)),
    anchorsW(std::move(anchorsW)),
    anchorsH(std::move(anchorsH)),
    predictionsPerCell(std::move(predictionsPerCell))
{
  // Nothing to do here.
}

template<typename MatType>
YOLOv3Layer<MatType>&
YOLOv3Layer<MatType>::
operator=(const YOLOv3Layer& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    imgSize = other.imgSize;
    numAttributes = other.numAttributes;
    gridSize = other.gridSize;
    anchorsW = other.anchorsW;
    anchorsH = other.anchorsH;
    predictionsPerCell = other.predictionsPerCell;
  }
  return *this;
}

template<typename MatType>
YOLOv3Layer<MatType>&
YOLOv3Layer<MatType>::
operator=(YOLOv3Layer&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    imgSize = std::move(other.imgSize);
    numAttributes = std::move(other.numAttributes);
    gridSize = std::move(other.gridSize);
    anchorsW = std::move(other.anchorsW);
    anchorsH = std::move(other.anchorsH);
    predictionsPerCell = std::move(other.predictionsPerCell);
  }
  return *this;
}

template <typename MatType>
void YOLOv3Layer<MatType>::ComputeOutputDimensions()
{
  if (this->inputDimensions.size() != 3)
  {
    std::ostringstream errMessage;
    errMessage << "YOLOv3Layer::ComputeOutputDimensions(): "
               << "Input dimensions must be 3D, but there are "
               << this->inputDimensions.size() << " input "
               << "dimensions";
    throw std::logic_error(errMessage.str());
  }

  if (this->inputDimensions[0] != this->inputDimensions[1])
    throw std::logic_error("YOLOv3Layer::ComputeOutputDimensions(): "
      "Input dimensions must be square.");

  if (grid != this->inputDimensions[0] * this->inputDimensions[1] ||
      gridSize != this->inputDimensions[0])
  {
    throw std::logic_error("YOLOv3Layer::ComputeOutputDimensions(): "
      "Grid is the wrong size.");
  }

  this->outputDimensions = { numAttributes, grid * predictionsPerCell };
}

template <typename MatType>
void YOLOv3Layer<MatType>::Forward(const MatType& input, MatType& output)
{
  Type stride = imgSize / (Type)(gridSize);
  size_t batchSize = input.n_cols;
  output.set_size(input.n_rows, batchSize);

  CubeType inputCube;
  MakeAlias(inputCube, input, grid * numAttributes, predictionsPerCell,
    batchSize);

  CubeType outputCube(grid * numAttributes, predictionsPerCell, batchSize,
    arma::fill::zeros);

  CubeType reshapedCube;
  MakeAlias(reshapedCube, output, numAttributes,
    predictionsPerCell * grid, batchSize);

  // Input dimensions: gridSize 
  MatType offset = arma::regspace<MatType>(0, gridSize - 1);

#if ARMA_VERSION_MAJOR < 15
  // If arma::repcube is not available
  CubeType anchorsWBS(anchorsW.n_rows, anchorsW.n_cols, batchSize);
  CubeType anchorsHBS(anchorsW.n_rows, anchorsW.n_cols, batchSize);
  CubeType xOffset(grid, predictionsPerCell, batchSize);

  arma::Col<Type> offsetT =
    arma::vectorise(arma::repmat(offset.t(), gridSize, 1));
  CubeType yOffset(grid, predictionsPerCell, batchSize);
  for (size_t i = 0; i < batchSize; i++)
  {
    anchorsWBS.slice(i) = anchorsW;
    anchorsWBS.slice(i) = anchorsH;
    xOffset.slice(i) = arma::repmat(offset, gridSize, predictionsPerCell);
    yOffset.slice(i) = arma::repmat(offsetT, 1, predictionsPerCell);
  }
#else
  CubeType anchorsWBS = arma::repcube(anchorsW, 1, 1, batchSize);
  CubeType anchorsHBS = arma::repcube(anchorsH, 1, 1, batchSize);
  CubeType xOffset = arma::repcube(offset, gridSize,
    predictionsPerCell, batchSize);

  CubeType yOffset = arma::repcube(
    arma::vectorise(arma::repmat(offset.t(), gridSize, 1)),
    1, predictionsPerCell, batchSize);
#endif

  // TODO: add if (this->training). Add check for different batchSize.
  const size_t cols = predictionsPerCell - 1;
  // x
  outputCube.tube(0, 0, grid - 1, cols) =
    (xOffset + 1 / (1 + arma::exp(-inputCube.tube(0, 0, grid - 1, cols))))
    * stride;

  // y
  outputCube.tube(grid, 0, grid * 2 - 1, cols) =
    (yOffset + 1 / (1 + arma::exp(-inputCube.tube(grid, 0, grid * 2 - 1, cols))
    )) * stride;

  // w
  outputCube.tube(grid * 2, 0, grid * 3 - 1, cols) =
    anchorsWBS % arma::exp(inputCube.tube(grid * 2, 0, grid * 3 - 1, cols));

  // h
  outputCube.tube(grid * 3, 0, grid * 4 - 1, cols) =
    anchorsHBS % arma::exp(inputCube.tube(grid * 3, 0, grid * 4 - 1, cols));

  // apply logistic sigmoid to objectness and classification logits.
  outputCube.tube(grid * 4, 0, outputCube.n_rows - 1, cols) = 1. /
    (1 + arma::exp(-inputCube.tube(grid * 4, 0, inputCube.n_rows - 1, cols)));

  // Reshape, for each batch item.
  for (size_t i = 0; i < reshapedCube.n_slices; i++)
  {
    reshapedCube.slice(i) =
      arma::reshape(
        arma::reshape(
          outputCube.slice(i), grid, numAttributes * predictionsPerCell
        ).t(),
        numAttributes, predictionsPerCell * grid
      );
  }
}

template <typename MatType>
void YOLOv3Layer<MatType>::Backward(
    const MatType& input,
    const MatType& output,
    const MatType& gy,
    MatType& g)
{
  throw std::runtime_error("YOLOv3::Backward() not implemented.");
}

template <typename MatType>
template <typename Archive>
void YOLOv3Layer<MatType>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));
  ar(CEREAL_NVP(imgSize));
  ar(CEREAL_NVP(numAttributes));
  ar(CEREAL_NVP(gridSize));
  ar(CEREAL_NVP(grid));
  ar(CEREAL_NVP(anchorsW));
  ar(CEREAL_NVP(anchorsH));
  ar(CEREAL_NVP(predictionsPerCell));
}

} // namespace mlpack

#endif
