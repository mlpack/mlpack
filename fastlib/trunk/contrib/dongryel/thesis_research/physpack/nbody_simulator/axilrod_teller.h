/** @file axilrod_teller.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef PHYSPACK_NBODY_SIMULATOR_AXILROD_TELLER_H
#define PHYSPACK_NBODY_SIMULATOR_AXILROD_TELLER_H

#include "core/gnp/triple_range_distance_sq.h"
#include "core/gnp/triple_distance_sq.h"

namespace physpack {
namespace nbody_simulator {
class AxilrodTeller {
  public:

    core::math::Range RangeUnnormOnSq(
      const core::gnp::TripleRangeDistanceSq &range_in) const {

      // Put the code for computing the minimum and the maximum
      // using the global optimization based nonconvex optimization.
      return core::math::Range(0.0, 0.0);
    }

    double EvalUnnormOnSq(
      const core::gnp::TripleDistanceSq &squared_distances) const {
      double first_cosine =
        (squared_distances.distance_sq(0, 2) +
         squared_distances.distance_sq(0, 1) -
         squared_distances.distance_sq(1, 2)) / (
          2.0 * sqrt(
            squared_distances.distance_sq(0, 2) *
            squared_distances.distance_sq(0, 1)));
      double second_cosine =
        (squared_distances.distance_sq(1, 2) +
         squared_distances.distance_sq(0, 1) -
         squared_distances.distance_sq(0, 2)) /
        (2.0 * sqrt(
           squared_distances.distance_sq(1, 2) *
           squared_distances.distance_sq(0, 1)));
      double third_cosine =
        (squared_distances.distance_sq(1, 2) +
         squared_distances.distance_sq(0, 2) -
         squared_distances.distance_sq(0, 1)) /
        (2.0 * sqrt(
           squared_distances.distance_sq(1, 2) *
           squared_distances.distance_sq(0, 2)));
      double numerator = 1.0 + 3.0 * first_cosine *
                         second_cosine * third_cosine;
      double denominator = pow(
                             squared_distances.distance_sq(0, 1) *
                             squared_distances.distance_sq(0, 2) *
                             squared_distances.distance_sq(1, 2), 1.5);
      return numerator / denominator;
    }
};
};
};

#endif
