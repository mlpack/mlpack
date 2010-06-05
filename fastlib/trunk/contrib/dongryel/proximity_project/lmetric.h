/** @author Dongryeol Lee
 *
 *  @file lmetric.h
 *
 *  @brief Implements the general L_p metric object.
 */

#ifndef CONTRIB_DONGRYEL_PROXIMITY_PROJECT_LMETRIC_H
#define CONTRIB_DONGRYEL_PROXIMITY_PROJECT_LMETRIC_H

#include "fastlib/math/math_lib.h"
#include "contrib/dongryel/proximity_project/metric.h"

namespace proximity_project {
template<int t_pow>
class LMetric: public virtual Metric {

  public:
    double Distance(
      const Vector &first_point, const Vector &second_point) const {
      return math::Pow<1, t_pow>(
               this->DistanceIneq(first_point, second_point));
    }

    double DistanceSq(
      const Vector &first_point, const Vector &second_point) const {
      return math::Pow<2, 1>(Distance(first_point, second_point));
    }

    double DistanceIneq(
      const Vector &first_point, const Vector &second_point) const {
      return la::RawLMetric<t_pow>(first_point, second_point);
    }
};

class LMetric<2>: public virtual Metric {

  public:
    double Distance(
      const Vector &first_point, const Vector &second_point) const {

      return math::Pow<1, 2>(DistanceIneq(a, b));
    }

    double DistanceSq(
      const Vector &first_point, const Vector &second_point) const {

      return la::RawLMetric<2>(first_point, second_point);
    }

    double DistanceIneq(
      const Vector &first_point, const Vector &second_point) const {
      return this->Distance(first_point, second_point);
    }
};
};

#endif
