/**
 * @file core/distances/iou_distance.hpp
 * @author Ryan Curtin
 *
 * Intersection-over-Union (IoU) distance.  This is closely related to the
 * Intersection-over-Union metric (in metrics/), but satisfies the properties of
 * the triangle inequality.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTANCES_IOU_DISTANCE_HPP
#define MLPACK_CORE_DISTANCES_IOU_DISTANCE_HPP

#include <mlpack/core/metrics/iou_metric.hpp>

namespace mlpack {

/**
 * The Intersection-over-Union (IoU) distance is a distance metric that
 * satisfies the triangle inequality and other properties of metrics.  It is
 * defined as 1 minus the IoU metric:
 *
 * ```
 * d(x, y) = 1 - iou_metric(x, y)
 * ```
 *
 * See the documentation for the `IoU` class for more information on the
 * intersection-over-union documentation (in metrics/iou_metric.hpp).
 *
 * For more information on why this satisfies the triangle inequality and other
 * properties of metrics, see
 *
 * ```
 * @article{kosub2019note,
 *   title={A note on the triangle inequality for the Jaccard distance},
 *   author={Kosub, Sven},
 *   journal={Pattern Recognition Letters},
 *   volume={120},
 *   pages={36--38},
 *   year={2019},
 *   publisher={Elsevier}
 * }
 * ```
 *
 * The inputs to Evaluate() are expected to be a four-dimensional vector
 * representing the bounding boxes.  There are two representations that can be
 * used, depending on the value of the `UseCoordinates` template parameter.
 *
 * If `UseCoordinates` is `false` (the default), then each four-dimensional
 * vector specifies (in order) the values `[x0, y0, h, w]`, where `(x0, y0)` is
 * the lower left coordinates, `h` is the height, and `w` is the width.
 *
 * If `UseCoordinates` is `true`, then each four-dimensional vector specifies
 * (in order) the values `[x0, y0, x1, y1]`, where `(x0, y0)` is the lower left
 * coordinates, and `(x1, y1)` is the upper right coordinate.
 */
template<bool UseCoordinates = false>
class IoUDistance
{
 public:
  // Evaluate the metric.
  template<typename VecTypeA, typename VecTypeB>
  static typename VecTypeA::elem_type Evaluate(const VecTypeA& a,
                                               const VecTypeB& b)
  {
    using ElemType = typename VecTypeA::elem_type;

    return (ElemType) (1.0 - IoU<UseCoordinates>::Evaluate(a, b));
  }
};

} // namespace mlpack

#endif
