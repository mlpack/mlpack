/**
 * @file nms_metric_impl.hpp
 * @author Kartik Dutt
 *
 * Implementation of Non Maximal Supression metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_METRICS_NMS_IMPL_HPP
#define MLPACK_CORE_METRICS_NMS_IMPL_HPP

// In case it hasn't been included.
#include "non_maximal_supression.hpp"

namespace mlpack {
namespace metric {

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
  Log::Assert(boundingBoxes.n_rows == 4, "Bounding boxes must \
      contain only 4 rows determining coordinates of bounding \
      box either in {x1, y1, x2, y2} or {x1, y1, h, w} format.\
      Refer to the documentation for more information.");

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
    area = (boundingBoxes.row(2)) % (boundingBoxes.row(3));
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

    // Calculate IoU of remaining boxes with the last bounding box with
    // the highest confidence score.
    BoundingBoxesType intersectionArea;
    if (UseCoordinates)
    {
      intersectionArea = arma::clamp(arma::clamp(
          boundingBoxes.submat(arma::uvec(1).fill(2), sortedIndices), DBL_MIN,
          boundingBoxes(2, selectedIndex)) - arma::clamp(
          boundingBoxes.submat(arma::uvec(1).fill(0), sortedIndices),
          boundingBoxes(0, selectedIndex), DBL_MAX), 0.0, DBL_MAX) %
          arma::clamp(arma::clamp(boundingBoxes.submat(arma::uvec(1).fill(3),
          sortedIndices), DBL_MIN, boundingBoxes(3, selectedIndex)) -
          arma::clamp(boundingBoxes.submat(arma::uvec(1).fill(1),
          sortedIndices), boundingBoxes(1, selectedIndex), DBL_MAX),
          0.0, DBL_MAX);
    }
    else
    {
      intersectionArea = arma::clamp(arma::clamp(
          boundingBoxes.submat(arma::uvec(1).fill(2), sortedIndices) +
          boundingBoxes.submat(arma::uvec(1).fill(0), sortedIndices), DBL_MIN,
          boundingBoxes(2, selectedIndex) + boundingBoxes(0, selectedIndex)) -
          arma::clamp(boundingBoxes.submat(arma::uvec(1).fill(0),
          sortedIndices), boundingBoxes(0, selectedIndex), DBL_MAX), 0.0,
          DBL_MAX) % arma::clamp(arma::clamp(
          boundingBoxes.submat(arma::uvec(1).fill(3), sortedIndices) +
          boundingBoxes.submat(arma::uvec(1).fill(1),
          sortedIndices), DBL_MIN, boundingBoxes(3, selectedIndex) +
          boundingBoxes(1, selectedIndex)) -
          arma::clamp(boundingBoxes.submat(arma::uvec(1).fill(1),
          sortedIndices), boundingBoxes(1, selectedIndex), DBL_MAX),
          0.0, DBL_MAX);
    }

    BoundingBoxesType calculateIoU = intersectionArea /
        (area(sortedIndices).t() - intersectionArea + area(selectedIndex));

    sortedIndices = sortedIndices(arma::find(calculateIoU <= threshold));
  }

  selectedIndices = arma::flipud(selectedIndices);
}

template<bool UseCoordinates>
template<typename Archive>
void NMS<UseCoordinates>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(useCoordinates);
}

} // namespace metric
} // namespace mlpack
#endif
