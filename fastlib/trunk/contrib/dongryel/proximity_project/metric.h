/** @author Dongryeol Lee
 *
 *  @file metric.h
 *
 *  @brief The general metric that can be inherited from.
 */

#ifndef CONTRIB_DONGRYEL_PROXIMITY_PROJECT_METRIC_H
#define CONTRIB_DONGRYEL_PROXIMITY_PROJECT_METRIC_H

#include "fastlib/la/matrix.h"

namespace fl {
namespace ml {
class Metric {
  public:
    virtual double Distance(
      const Vector &first_point, const Vector &second_point) const = 0;

    virtual double DistanceSq(
      const Vector &first_point, const Vector &second_point) const = 0;

    virtual double DistanceIneq(
      const Vector &first_point, const Vector &second_point) const = 0;
};
};
};

#endif
