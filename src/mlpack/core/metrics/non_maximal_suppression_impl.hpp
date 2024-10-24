/**
 * @file core/metrics/nms_metric_impl.hpp
 * @author Kartik Dutt
 *
 * Implementation of Non Maximal Suppression metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_METRICS_NMS_IMPL_HPP
#define MLPACK_CORE_METRICS_NMS_IMPL_HPP

// In case it hasn't been included.
#include "non_maximal_suppression.hpp"

#include <mlpack/core/util/log.hpp>

namespace mlpack {

template<bool UseCoordinates>
template<
    typename BoundingBoxesType,
    typename ConfidenceScoreType,
    typename OutputType
>
void NMS<UseCoordinates>::Evaluate(
    const BoundingBoxesType& boundingBoxes,
    const ConfidenceScoreType& confidenceScores,
    OutputType& selectedIndices,
    const double threshold)
{
  Log::Assert(boundingBoxes.n_rows == 4, "Bounding boxes must "
      "contain only 4 rows determining coordinates of bounding "
      "box either in {x1, y1, x2, y2} or {x1, y1, h, w} format."
      "Refer to the documentation for more information.");

  Log::Assert(confidenceScores.n_cols != boundingBoxes.n_cols, "Each "
      "bounding box must correspond to atleast and only 1 bounding box. "
      "Found " + std::to_string(confidenceScores.n_cols) + " confidence "
      "scores for " + std::to_string(boundingBoxes.n_cols) +
      " bounding boxes.");

  // Clear selected bounding boxes.
  selectedIndices.clear();

  // Obtain Sorted indices for bounding boxes according to
  // their confidence scores.
  arma::ucolvec sortedIndices = arma::sort_index(confidenceScores);

  // Pre-Compute area of each bounding box.
  arma::mat area;
  if (UseCoordinates)
  {
    area = (boundingBoxes.row(2) - boundingBoxes.row(0)) %
        (boundingBoxes.row(3) - boundingBoxes.row(1));
  }
  else
  {
    area = boundingBoxes.row(2) % boundingBoxes.row(3);
  }

  while (sortedIndices.n_elem > 0)
  {
    size_t selectedIndex = sortedIndices(sortedIndices.n_elem - 1);

    // Choose the box with the largest probability.
    selectedIndices.insert_rows(0, arma::uvec(1).fill(selectedIndex));

    // Check if there are other bounding boxes to compare with.
    if (sortedIndices.n_elem == 1)
    {
      break;
    }

    // Remove the last index.
    sortedIndices = sortedIndices(arma::span(0, sortedIndices.n_rows - 2),
        arma::span());

    // Get x and y coordinates for remaining bounding boxes.
    BoundingBoxesType x2 = boundingBoxes.submat(arma::uvec(1).fill(2),
        sortedIndices);

    BoundingBoxesType x1 = boundingBoxes.submat(arma::uvec(1).fill(0),
        sortedIndices);

    BoundingBoxesType y2 = boundingBoxes.submat(arma::uvec(1).fill(3),
        sortedIndices);

    BoundingBoxesType y1 = boundingBoxes.submat(arma::uvec(1).fill(1),
        sortedIndices);

    size_t selectedX2 = boundingBoxes(2, selectedIndex);
    size_t selectedY2 = boundingBoxes(3, selectedIndex);
    size_t selectedX1 = boundingBoxes(0, selectedIndex);
    size_t selectedY1 = boundingBoxes(1, selectedIndex);

    if (!UseCoordinates)
    {
      // Change height - width representation to coordinate represention.
      x2 = x2 + x1;
      y2 = y2 + y1;
      selectedX2 = selectedX2 + selectedX1;
      selectedY2 = selectedY2 + selectedY1;
    }

    // Calculate points of intersection between the bounding box with
    // highest confidence score and remaining bounding boxes.
    x2 = arma::clamp(x2, DBL_MIN, selectedX2);
    y2 = arma::clamp(y2, DBL_MIN, selectedY2);
    x1 = arma::clamp(x1, selectedX1, DBL_MAX);
    y1 = arma::clamp(y1, selectedY1, DBL_MAX);

    BoundingBoxesType intersectionArea = arma::clamp(x2 - x1, 0.0, DBL_MAX) %
          arma::clamp(y2 - y1, 0.0, DBL_MAX);

    // Calculate IoU of remaining boxes with the last bounding box with
    // the highest confidence score.
    BoundingBoxesType calculateIoU = intersectionArea /
        (area(sortedIndices).t() - intersectionArea + area(selectedIndex));

    sortedIndices = sortedIndices(arma::find(calculateIoU <= threshold));
  }

  selectedIndices = arma::flipud(selectedIndices);
}

template<bool UseCoordinates>
template<typename Archive>
void NMS<UseCoordinates>::serialize(
    Archive& /* ar */,
    const uint32_t /* version */)
{
  // Nothing to do here.
}

} // namespace mlpack

#endif
