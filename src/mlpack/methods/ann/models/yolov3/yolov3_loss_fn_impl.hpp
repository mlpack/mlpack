/**
 * @file methods/ann/loss_functions/yolov3_loss_fn_impl.hpp
 * @author Andrew Furey
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_MODELS_YOLOV3_LOSS_FN_IMPL_HPP
#define MLPACK_METHODS_ANN_MODELS_YOLOV3_LOSS_FN_IMPL_HPP

#include "yolov3_loss_fn.hpp"

namespace mlpack {

template<typename MatType>
typename YOLOv3Loss<MatType>::ElemType
YOLOv3Loss<MatType>::Forward(const MatType& predictions,
                             const MatType& targets,
                             const MatType& bestPredictionIndices,
                             const MatType& ignorePredictions,
                             const MatType& scales,
                             const MatType& numTargets)
{

  const size_t batchSize = predictions.n_cols;
  const IndexType bestPredictionIndicesInt = conv_to<IndexType>::from(bestPredictionIndices);

  CheckShapes(predictions,
              targets,
              bestPredictionIndicesInt,
              ignorePredictions,
              scales,
              numTargets);

  ElemType loss = 0;
  MatType probabilities, correctBoxes, keepObject;

  CubeType predictionsCube, targetsCube;
  MakeAlias(predictionsCube, predictions, numAttributes, numBoxes, batchSize);
  MakeAlias(targetsCube, targets, numAttributes, numTruths, batchSize);

  MatType error = MatType(numAttributes, numBoxes);
  MatType errorTruths = MatType(numAttributes, numTruths);

  MatType repeatedNumTargets = repmat(numTargets, numAttributes, 1);
  for (size_t i = 0; i < batchSize; i++)
  {
    error.fill(0);

    // boxes that don't match the targets only affect the objectness loss (objectness should be 0).
    error.row(4) = -log(1. - (1. / (1. + exp(-predictionsCube.slice(i).row(4)))));

    // gather boxes that match the ground truths
    correctBoxes = predictionsCube.slice(i).cols(bestPredictionIndicesInt.col(i));

    // squared error for box coords x, y, w, y.
    errorTruths.rows(0, 3) = repmat(scales.col(i).t(), 4, 1) %
      pow(targetsCube.slice(i).rows(0, 3) - correctBoxes.rows(0, 3), 2);

    // binary cross entropy for objectness and classification for boxes
    // whose anchors match the ground truths the best.
    probabilities = 1. / (1. + exp(-correctBoxes.rows(4, numAttributes - 1)));

    // Objectness
    errorTruths.row(4) = -log(probabilities.row(0));

    // Classes
    errorTruths.rows(5, numAttributes - 1) =
      targetsCube.slice(i).rows(5, numAttributes - 1) *
      -log(probabilities.rows(1, numAttributes - 5)) +
      (1. - targetsCube.slice(i).rows(5, numAttributes - 1)) *
      -log(1. - probabilities.rows(1, numAttributes - 5));

    // handle variable number of targets in each image in the batch.
    keepObject = conv_to<MatType>::from(
      keepObjectRange < repmat(repeatedNumTargets.col(i), 1, numTruths));

    error.cols(bestPredictionIndicesInt.col(i)) =
      error.cols(bestPredictionIndicesInt.col(i)) %
      (1. - keepObject) + errorTruths % keepObject;

    // at end, ignore boxes as needed, then accumulate into loss scalar.
    loss +=
      accu(error % (1. - repmat(ignorePredictions.col(i).t(), numAttributes, 1)));
  }

  return loss;
}

template<typename MatType>
void YOLOv3Loss<MatType>::Backward(const MatType& predictions,
                                   const MatType& targets,
                                   const MatType& bestPredictionIndices,
                                   const MatType& ignorePredictions,
                                   const MatType& scales,
                                   const MatType& numTargets,
                                   MatType& loss)
{
  const size_t batchSize = predictions.n_cols;

  const IndexType bestPredictionIndicesInt = conv_to<IndexType>::from(bestPredictionIndices);

  CheckShapes(predictions,
              targets,
              bestPredictionIndicesInt,
              ignorePredictions,
              scales,
              numTargets);

  loss.fill(0);

  CubeType predictionsCube, targetsCube, lossCube;
  MakeAlias(predictionsCube, predictions, numAttributes, numBoxes, batchSize);
  MakeAlias(targetsCube, targets, numAttributes, numTruths, batchSize);
  MakeAlias(lossCube, loss, numAttributes, numBoxes, batchSize);

  MatType dLossdBoxes = MatType(numAttributes, numTruths);

  MatType probabilities, correctBoxes, keepObject;

  MatType repeatedNumTargets = repmat(numTargets, numAttributes, 1);

  lossCube.row(4) = 1. / (1. + exp(-predictionsCube.row(4)));
  for (size_t i = 0; i < batchSize; i++)
  {
    correctBoxes = predictionsCube.slice(i).cols(bestPredictionIndicesInt.col(i));
    dLossdBoxes.rows(0, 3) = repmat(scales.col(i).t(), 4, 1) % (2. * (correctBoxes.rows(0, 3) - targetsCube.slice(i).rows(0, 3)));

    probabilities = 1. / (1. + exp(-correctBoxes.rows(4, numAttributes - 1)));
    dLossdBoxes.row(4) = probabilities.row(0) - 1.;

    dLossdBoxes.rows(5, numAttributes - 1) =
      targetsCube.slice(i).rows(5, numAttributes - 1) * (probabilities.rows(1, numAttributes - 5) - 1.) +
      (1 - targetsCube.slice(i).rows(5, numAttributes - 1)) * (probabilities.rows(1, numAttributes - 5));

    keepObject = conv_to<MatType>::from(
      // keepObjectRange < repmat(repmat(numTargets, numAttributes, 1).col(i), 1, numTruths));
      keepObjectRange < repmat(repeatedNumTargets.col(i), 1, numTruths));
    lossCube.slice(i).cols(bestPredictionIndicesInt.col(i)) =
      lossCube.slice(i).cols(bestPredictionIndicesInt.col(i)) % (1 - keepObject) +
      (dLossdBoxes % keepObject);

    lossCube.slice(i) %=
      (1. - repmat(ignorePredictions.col(i).t(), numAttributes, 1));
  }

}

template <typename MatType>
void YOLOv3Loss<MatType>::CheckShapes(const MatType& predictions,
                                      const MatType& targets,
                                      const IndexType& bestPredictionIndices,
                                      const MatType& ignorePredictions,
                                      const MatType& scales,
                                      const MatType& numTargets)
{
  const size_t batchSize = predictions.n_cols;

  std::ostringstream errMessage;
  if (predictions.n_rows != numAttributes * numBoxes)
  {
    errMessage << "YOLOv3Loss::Forward(): Expected predictions to be of shape ("
      << numAttributes * numBoxes << ", " << batchSize << "), but got (" <<
      predictions.n_rows << ", " << predictions.n_cols << ").";
    throw std::logic_error(errMessage.str());
  }

  if (targets.n_rows != numAttributes * numTruths ||
      targets.n_cols != batchSize)
  {
    errMessage << "YOLOv3Loss::Forward(): Expected targets to be of shape ("
      << numAttributes * numTruths << ", " << batchSize << "), but got (" <<
      targets.n_rows << ", " << targets.n_cols << ").";
    throw std::logic_error(errMessage.str());
  }

  if (bestPredictionIndices.n_rows != numTruths ||
      bestPredictionIndices.n_cols != batchSize)
  {
    errMessage << "YOLOv3Loss::Forward(): Expected bestPredictionIndices to be"
      " of shape (" << numTruths << ", " << batchSize << "), but got (" <<
      bestPredictionIndices.n_rows << ", " << bestPredictionIndices.n_cols <<
      ").";
    throw std::logic_error(errMessage.str());
  }

  if (ignorePredictions.n_rows != numBoxes ||
      ignorePredictions.n_cols != batchSize)
  {
    errMessage << "YOLOv3Loss::Forward(): Expected ignorePredictions to be"
      " of shape (" << numBoxes << ", " << batchSize << "), but got (" <<
      ignorePredictions.n_rows << ", " << ignorePredictions.n_cols << ").";
    throw std::logic_error(errMessage.str());
  }

  if (scales.n_rows != numTruths || scales.n_cols != batchSize)
  {
    errMessage << "YOLOv3Loss::Forward(): Expected scales to be of shape ("
      << numTruths << ", " << batchSize << "), but got (" <<
      scales.n_rows << ", " << scales.n_cols << ").";
    throw std::logic_error(errMessage.str());
  }

  if (numTargets.n_rows != 1 || numTargets.n_cols != batchSize)
  {
    errMessage << "YOLOv3Loss::Forward(): Expected numTargets to be of shape ("
      << 1 << ", " << batchSize << "), but got (" <<
      numTargets.n_rows << ", " << numTargets.n_cols << ").";
    throw std::logic_error(errMessage.str());
  }
}

template<typename MatType>
template<typename Archive>
void YOLOv3Loss<MatType>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(numBoxes));
  ar(CEREAL_NVP(numTruths));
  ar(CEREAL_NVP(numAttributes));

  if (cereal::is_loading<Archive>())
  {
    keepObjectRange = repmat(regspace(0, numTruths - 1).t(), numAttributes, 1);
  }
}

} // namespace mlpack

#endif
