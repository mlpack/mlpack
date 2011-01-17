/** @file bounds_aux.h
 *
 *  A collection of methods that provide additional methods for
 *  computing bound-related quantities.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *  @bug No known bugs.
 */

#ifndef MLPACK_SERIES_EXPANSION_BOUNDS_AUX_H
#define MLPACK_SERIES_EXPANSION_BOUNDS_AUX_H

#include <armadillo>
#include "core/math/range.h"
#include "core/metric_kernels/lmetric.h"
#include "core/table/dense_matrix.h"
#include "core/table/dense_point.h"
#include "core/tree/ball_bound.h"
#include "core/tree/hrect_bound.h"

namespace mlpack {
namespace series_expansion {

class BoundsAux {

  public:

    /** @brief We assume always L2 metric. I have not seen series
     *         expansion being applied under a metric other than L_2
     *         metric in the Cartesian setting.
     */
    static const int t_pow = 2;

  public:
    static void MaxDistanceSq(
      const core::tree::HrectBound &bound1,
      const core::tree::HrectBound &bound2,
      arma::vec *furthest_point_in_bound1,
      double *furthest_dsqd) {

      furthest_point_in_bound1->set_zeros(bound1.dim());
      int dim = furthest_point_in_bound1.length();
      *furthest_dsqd = 0;

      for(int d = 0; d < dim; d++) {

        const core::math::Range &bound1_range = bound1.get(d);
        const core::math::Range &bound2_range = bound2.get(d);

        double v1 = bound2_range.hi - bound1_range.lo;
        double v2 = bound1_range.hi - bound2_range.lo;
        double v;

        if(v1 > v2) {
          (*furthest_point_in_bound1)[d] = bound1_range.lo;
          v = v1;
        }
        else {
          (*furthest_point_in_bound1)[d] = bound1_range.hi;
          v = v2;
        }
        (*furthest_dsqd) += math::PowAbs<t_pow, 1>(v); // v is non-negative
      }
      (*furthest_dsqd) = math::Pow<2, t_pow>(*furthest_dsqd);
    }

    static void MaxDistanceSq(
      const core::tree::HrectBound &bound1,
      const core::table::DensePoint &bound2_centroid,
      arma::vec *furthest_point_in_bound1,
      double *furthest_dsqd) {

      furthest_point_in_bound1->set_zeros(bound1.center().length());
      int dim = furthest_point_in_bound1->n_elem;
      *furthest_dsqd = 0;

      for(int d = 0; d < dim; d++) {

        const core::math::Range &bound1_range = bound1.get(d);
        double v1 = bound2_centroid[d] - bound1_range.lo;
        double v2 = bound1_range.hi - bound2_centroid[d];
        double v;

        if(v1 > v2) {
          (*furthest_point_in_bound1)[d] = bound1_range.lo;
          v = v1;
        }
        else {
          (*furthest_point_in_bound1)[d] = bound1_range.hi;
          v = v2;
        }
        (*furthest_dsqd) += math::PowAbs<t_pow, 1>(v); // v is non-negative
      }
      (*furthest_dsqd) = math::Pow<2, t_pow>(*furthest_dsqd);
    }

    static void MaxDistanceSq(
      const core::tree::BallBound &bound1,
      const core::table::DensePoint &bound2_centroid,
      arma::vec *furthest_point_in_bound1,
      double *furthest_dsqd) {

      *furthest_dsqd = 0;

      // First compute the distance between the centroid of the bounding
      // ball and the given point.
      double distance =
        LMetric<t_pow>::Distance(bound1.center(), bound2_centroid);

      // Compute the unit vector that has the same direction as the
      // vector pointing from the given point to the bounding ball
      // center.
      arma::vec bound1_center_alias;
      arma::vec bound2_centroid_alias;
      core::table::DensePointToArmaVec(bound1.center(), &bound1_center_alias);
      core::table::DensePointToArmaVec(bound2_centroid, &bound2_centroid_alias);
      arma::vec unit_vector = bound1_center_alias - bound2_centroid_alias;
      unit_vector *= 1.0 / distance;
      (*furthest_point_in_bound1) = bound1_center_alias;
      (*furthest_point_in_bound1) += bound1.radius() * unit_vector;

      // Temporary LMetric object. Eliminate when there is an issue
      // later.
      core::metric_kernels::LMetric<2> l2_metric;
      (*furthest_dsqd) = l2_metric.DistanceSq(
                           bound2_centroid, *furthest_point_in_bound1);
    }

    /** @brief Returns the maximum side length of the bounding box that
     *         encloses the given ball bound. That is, twice the radius
     *         of the given ball bound.
     */
    static double MaxSideLengthOfBoundingBox(
      const core::tree::BallBound &ball_bound) {
      return ball_bound.radius() * 2;
    }

    /** @brief Returns the maximum side length of the bounding box that
     *         encloses the given bounding box. That is, its maximum
     *         side length.
     */
    static double MaxSideLengthOfBoundingBox(
      const core::tree::HrectBound &bound) {

      double max_length = 0;
      for(int d = 0; d < bound.dim(); d++) {
        const core::math::Range &range = bound.get(d);
        max_length = std::max(max_length, range.width());
      }
      return max_length;
    }

    /** @brief Returns the maximum distance between two bound types in
     *         L1 sense.
     */
    static double MaxL1Distance(
      const core::tree::BallBound &ball_bound1,
      const core::tree::BallBound &ball_bound2,
      int *dimension) {

      const core::table::DensePoint &center1 = ball_bound1.center();
      const core::table::DensePoint &center2 = ball_bound2.center();
      int dim = ball_bound1.center().length();
      double l1_distance = 0;
      for(int d = 0; d < dim; d++) {
        l1_distance += fabs(center1[d] - center2[d]);
      }
      l1_distance += ball_bound1.radius() + ball_bound2.radius();
      *dimension = center1.length();
      return l1_distance;
    }

    /** @brief Returns the maximum distance between two bound types in
     *         L1 sense.
     */
    static double MaxL1Distance(
      const core::tree::HrectBound &bound1,
      const core::tree::HrectBound &bound2,
      int *dimension) {

      double farthest_distance_manhattan = 0;
      for(int d = 0; d < bound1.dim(); d++) {
        const core::math::Range &range1 = bound1.get(d);
        const core::math::Range &range2 = bound2.get(d);
        double bound1_centroid_coord = range1.lo + range1.width() / 2;
        farthest_distance_manhattan =
          max(farthest_distance_manhattan,
              max(fabs(bound1_centroid_coord - range2.lo),
                  fabs(bound1_centroid_coord - range2.hi)));
      }
      *dimension = bound1.dim();
      return farthest_distance_manhattan;
    }
};
}
}

#endif
