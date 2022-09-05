/**
 * @file core/metrics/non_maximal_suppression.hpp
 * @author Kartik Dutt
 *
 * Definition of Non Maximal Suppression metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_METRICS_NMS_HPP
#define MLPACK_CORE_METRICS_NMS_HPP

namespace mlpack {

/**
 * Definition of Non Maximal Suppression.
 *
 * Performs non-maximal suppression (NMS) on the boxes according to their
 * Intersection-over-Union (IoU). NMS iteratively removes lower scoring boxes
 * which have an IoU greater than threshold with another high scoring box.
 *
 * For bounding box representation there are two common representation
 * either as coordinates i.e. each value in vector represents a
 * coordinate in the format x0, y0, x1, y1 where x0, y0 represent the
 * lower left coordinate and x1, y1, represent upper right coordinate.
 * 
 * Second representation follows the following representation : x0, y0, h, w.
 * Where x0 and y0 are bottom left bounding box coordinates and h, w are
 * height and width of the bounding box.
 *
 * @tparam UseCoordinates Toggles between the two representation of bounding box.
 *                        If true, each value in vector represents a coordinate 
 *                        in the formate x0, y0, x1, y1. Else the bounding box is
 *                        represented as x0, y0, h, w.
 */
template<bool UseCoordinates = false>
class NMS
{
 public:
  //! Default constructor required to satisfy the Metric policy.
  NMS() { /* Nothing to do here. */ }

  /**
   * Performs non-maximal suppression.
   *
   * @param boundingBoxes Column major representation of bounding boxes
   *                      i.e. Each column corresponds to a different bounding
   *                      box. Each bounding box should contain 4 points only
   *                      either {x1, y1, x2, y2} or {x1, y1, h, w} depending
   *                      on UseCoordinates parameter.
   * @param confidenceScores Vector containing confidence score corresponding
   *                         to each bounding box.
   * @param selectedIndices Output of Non Maximal Suppression (NMS) is stored
   *                        here. It contains a list of indices corresponding
   *                        to bounding boxes in input parameter, sorted
   *                        in descending order of the confidence scores.
   * @param threshold Threshold used to discard all overlapping bounding boxes
   *                  that have IoU greater than the threshold.
   */
  template<
      typename BoundingBoxesType,
      typename ConfidenceScoreType,
      typename OutputType
  >
  static void Evaluate(const BoundingBoxesType& boundingBoxes,
                       const ConfidenceScoreType& confidenceScores,
                       OutputType& selectedIndices,
                       const double threshold = 0.5);

  static const bool useCoordinates = UseCoordinates;

  //! Serialize the metric.
  template <typename Archive>
  void serialize(Archive &ar, const uint32_t /* version */);
}; // Class NMS.

} // namespace mlpack

// Include implementation.
#include "non_maximal_suppression_impl.hpp"

#endif
