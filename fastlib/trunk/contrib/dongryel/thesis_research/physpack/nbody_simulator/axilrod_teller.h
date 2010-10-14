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

    double minimum_negative_contribution(
      const core::gnp::TripleRangeDistanceSq &range_in) const {

      const arma::mat &min_distance_sq = range_in.min_distance_sq();
      const arma::mat &max_distance_sq = range_in.max_distance_sq();
      double numerator_first_part =
        -3.0 * core::math::Pow<3, 1>(max_distance_sq.at(0, 1));
      double max_diff =
        std::max(
          fabs(
            max_distance_sq.at(0, 2) - min_distance_sq.at(1, 2)),
          fabs(
            min_distance_sq.at(0, 2) - max_distance_sq.at(1, 2)));

      double numerator_second_part =
        -3.0 * core::math::Sqr(max_diff) *
        (max_distance_sq.at(0, 2) + max_distance_sq.at(1, 2))

        double denominator = 8.0 * pow(
                               min_distance_sq.at(0, 1) *
                               min_distance_sq.at(0, 2) *
                               min_distance_sq.at(1, 2), 2.5);

      return (numerator_first_part + numerator_second_part) / denominator;
    }

    double maximum_negative_contribution(
      const core::gnp::TripleRangeDistanceSq &range_in) const {

      const arma::mat &min_distance_sq = range_in.min_distance_sq();
      const arma::mat &max_distance_sq = range_in.max_distance_sq();
      double numerator_first_part =
        -3.0 * core::math::Pow<3, 1>(min_distance_sq.at(0, 1));
      double numerator_second_part = 0;
      if(max_distance_sq.at(0, 2) <= min_distance_sq.at(1, 2) ||
          max_distance_sq.at(1, 2) <= min_distance_sq.at(0, 2)) {
        double difference =
          std::min(
            fabs(min_distance_sq.at(1, 2) - max_distance_sq.at(0, 2)),
            fabs(min_distance_sq.at(0, 2) - max_distance_sq.at(1, 2)));
        numerator_second_part =
          -3.0 * core::math::Sqr(diff) *
          (min_distance_sq.at(0, 2) + min_distance_sq.at(1, 2));
      }

      double denominator = 8.0 * pow(
                             max_distance_sq.at(0, 1) *
                             max_distance_sq.at(0, 2) *
                             max_distance_sq.at(1, 2), 2.5);

      return (numerator_first_part + numerator_second_part) / denominator;
    }

    double minimum_positive_contribution(
      const core::gnp::TripleRangeDistanceSq &range_in) const {

      const arma::mat &min_distance_sq = range_in.min_distance_sq();
      const arma::mat &max_distance_sq = range_in.max_distance_sq();
      double numerator_first_part =
        3.0 * core::math::Sqr(min_distance_sq.at(0, 1)) *
        (min_distance_sq.at(0, 2) + min_distance_sq.at(1, 2));
      double numerator_second_part =
        min_distance_sq.at(0, 1) * (
          3.0 * core::math::Sqr(min_distance_sq.at(0, 2)) +
          2.0 * min_distance_sq.at(0, 2) * min_distance_sq.at(1, 2) +
          3.0 * core::math::Sqr(min_distance_sq.at(1, 2)));

      double denominator = 8.0 * pow(
                             max_distance_sq.at(0, 1) *
                             max_distance_sq.at(0, 2) *
                             max_distance_sq.at(1, 2), 2.5);

      return (numerator_first_part + numerator_second_part) / denominator;
    }

    double maximum_positive_contribution(
      const core::gnp::TripleRangeDistanceSq &range_in) const {

      const arma::mat &min_distance_sq = range_in.min_distance_sq();
      const arma::mat &max_distance_sq = range_in.max_distance_sq();
      double numerator_first_part =
        3.0 * core::math::Sqr(max_distance_sq.at(0, 1)) *
        (max_distance_sq.at(0, 2) + max_distance_sq.at(1, 2));
      double numerator_second_part =
        max_distance_sq.at(0, 1) * (
          3.0 * core::math::Sqr(max_distance_sq.at(0, 2)) +
          2.0 * max_distance_sq.at(0, 2) * max_distance_sq.at(1, 2) +
          3.0 * core::math::Sqr(max_distance_sq.at(1, 2)));

      double denominator = 8.0 * pow(
                             min_distance_sq.at(0, 1) *
                             min_distance_sq.at(0, 2) *
                             min_distance_sq.at(1, 2), 2.5);

      return (numerator_first_part + numerator_second_part) / denominator;
    }

    core::math::Range RangeUnnormOnSq(
      const core::gnp::TripleRangeDistanceSq &range_in) const {

      // Compute the negative contribution bound.
      core::math::Range negative_range(
        this->minimum_negative_contribution(range_in),
        this->maximum_negative_contribution(range_in));

      // Compute the positive contribution bound.
      core::math::Range positive_range(
        this->minimum_positive_contribution(range_in),
        this->maximum_positive_contribution(range_in));

      // Take the sum of two ranges.
      core::math::Range overall_range = negative_range +
                                        positive_range;

      return overall_range;
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
