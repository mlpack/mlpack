/**
 * @file core/metrics/iou_metric.hpp
 * @author Kartik Dutt
 *
 * Definition of Intersection Over Union metric. It is defined as intersection
 * of given bounding-boxes / masks divided by their union. Useful metric for
 * object detection.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_METRICS_IOU_HPP
#define MLPACK_CORE_METRICS_IOU_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Definition of Intersection over Union metric.
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
 * @tparam useCoordinates Toggles between the two representation of bounding
 *     box.  If true, each value in vector represents a coordinate in the
 *     format x0, y0, x1, y1. Else the bounding box is represented as x0, y0,
 *     h, w.
 */
template<bool UseCoordinates = false>
class IoU
{
 public:
  /**
   * Computes the Intersection over Union metric between of two
   * bounding boxes having pattern bx, by, h, w.
   *
   * @tparam VecTypeA Type of first vector.
   * @tparam VecTypeB Type of second vector.
   * @param a First vector.
   * @param b Second vector.
   * @return IoU of vectors a and b.
   */
  template<typename VecTypeA, typename VecTypeB>
  static typename VecTypeA::elem_type Evaluate(const VecTypeA& a,
                                               const VecTypeB& b);

  static const bool useCoordinates = UseCoordinates;

  //! Serialize the metric.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);
}; // class IoU

} // namespace mlpack

// Include implementation.
#include "iou_metric_impl.hpp"

#endif
