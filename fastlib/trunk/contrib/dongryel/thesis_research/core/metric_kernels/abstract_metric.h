/** @file abstract_metric.h
 *
 *  A prototype for an abstract metric.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_METRIC_KERNELS_ABSTRACT_METRIC_H
#define CORE_METRIC_KERNELS_ABSTRACT_METRIC_H

#include "core/table/abstract_point.h"

namespace core {
namespace metric_kernels {
class AbstractMetric {

  public:

    virtual ~AbstractMetric() {
    }

    /**
    * Computes the distance metric between two points.
    */
    virtual double Distance(
      const core::table::AbstractPoint& a,
      const core::table::AbstractPoint& b) const = 0;

    /**
     * Computes the distance metric between two points, raised to a
     * particular power.
     *
     * This might be faster so that you could get, for instance, squared
     * L2 distance.
     */
    virtual double DistanceSq(
      const core::table::AbstractPoint &a,
      const core::table::AbstractPoint &b) const = 0;
};
};
};

#endif
