/**
 * @file core/metrics/iou_metric_impl.hpp
 * @author Kartik Dutt
 *
 * Implementation of Intersection Over Union metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_METRICS_IOU_IMPL_HPP
#define MLPACK_CORE_METRICS_IOU_IMPL_HPP

// In case it hasn't been included.
#include "iou_metric.hpp"

#include <mlpack/core/util/log.hpp>

namespace mlpack {

template<bool UseCoordinates>
template <typename VecTypeA, typename VecTypeB>
typename VecTypeA::elem_type IoU<UseCoordinates>::Evaluate(
    const VecTypeA& a,
    const VecTypeB& b)
{
  Log::Assert(a.n_elem == b.n_elem && a.n_elem == 4, "Incorrect "
      "shape for bounding boxes. They must contain 4 elements either be "
      "{x0, y0, x1, y1} or {x0, y0, h, w}. Refer to the documentation "
      "for more information.");

  // Bounding boxes represented as {x0, y0, x1, y1}.
  if (UseCoordinates)
  {
    // Check the correctness of bounding box.
    if (a(0) >= a(2) || a(1) >= a(3) || b(0) >= b(2) || b(1) >= b(3))
    {
        Log::Fatal << "Check the correctness of bounding boxes i.e. " <<
            "{x0, y0} must represent lower left coordinates and " <<
            "{x1, y1} must represent upper right coordinates of bounding" <<
            "box." << std::endl;
    }

    typename VecTypeA::elem_type interSectionArea = std::max(0.0,
        std::min(a(2), b(2)) - std::max(a(0), b(0)) + 1) * std::max(0.0,
        std::min(a(3), b(3)) - std::max(a(1), b(1)) + 1);

    // Union of Area can be calculated using the following equation
    // A union B = A + B - A intersection B.
    return interSectionArea / (1.0 * ((a(2) - a(0) + 1) * (a(3) - a(1) + 1) +
        (b(2) - b(0) + 1) * (b(3) - b(1) + 1) - interSectionArea));
  }

  // Bounding boxes represented as {x0, y0, h, w}.
  // Check correctness of bounding box.
  Log::Assert(a(2) > 0 && b(2) > 0 && a(3) > 0 && b(3) > 0, "Height and width "
      "of bounding boxes must be greater than zero.");

  typename VecTypeA::elem_type interSectionArea = std::max(0.0,
      std::min(a(0) + a(2), b(0) + b(2)) - std::max(a(0), b(0)) + 1)
      * std::max(0.0, std::min(a(1) + a(3), b(1) + b(3)) - std::max(a(1),
      b(1)) + 1);

  return interSectionArea / (1.0 * ((a(2) + 1) * (a(3) + 1) + (b(2) + 1) *
      (b(3) + 1) - interSectionArea));
}
template<bool UseCoordinates>
template<typename Archive>
void IoU<UseCoordinates>::serialize(
    Archive& /* ar */,
    const uint32_t /* version */)
{
  // Nothing to do here.
}

} // namespace mlpack
#endif
