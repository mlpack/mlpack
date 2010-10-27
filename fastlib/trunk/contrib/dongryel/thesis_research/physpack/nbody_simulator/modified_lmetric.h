/** @file modified_lmetric.h
 *
 *  An implementation of general modified L_p metric that returns an
 *  epsilon for very small distance values.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef PHYSPACK_NBODY_SIMULATOR_MODIFIED_LMETRIC_H
#define PHYSPACK_NBODY_SIMULATOR_MODIFIED_LMETRIC_H

#include <armadillo>
#include "core/math/math_lib.h"
#include "core/metric_kernels/lmetric.h"

namespace physpack {
namespace nbody_simulator {

template<int t_pow>
class ModifiedLMetric: public core::metric_kernels::AbstractMetric {
  private:
    core::metric_kernels::LMetric<t_pow> metric_;

  public:

    double Distance(
      const core::table::AbstractPoint& a,
      const core::table::AbstractPoint& b) const {
      return std::max(
               metric_.Distance(a, b),
               sqrt(std::numeric_limits<double>::epsilon()));
    }

    double DistanceIneq(
      const core::table::AbstractPoint &a,
      const core::table::AbstractPoint &b) const {
      return std::max(
               metric_.DistanceIneq(a, b),
               std::numeric_limits<double>::epsilon());
    }

    double DistanceSq(
      const core::table::AbstractPoint &a,
      const core::table::AbstractPoint &b) const {
      return std::max(
               metric_.DistanceSq(a, b),
               std::numeric_limits<double>::epsilon());
    }
};
};
};

#endif
