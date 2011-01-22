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
  private:

    double normalizing_constant_;

  public:

    void Init(double normalizing_constant_in) {
      normalizing_constant_ = normalizing_constant_in;
    }

    template<typename TableType>
    double minimum_negative_contribution(
      const core::gnp::TripleRangeDistanceSq<TableType> &range_in) const {

      const arma::mat &min_distance_sq = range_in.min_distance_sq();
      const arma::mat &max_distance_sq = range_in.max_distance_sq();
      double numerator = - 0.375 * (
                           core::math::Pow<3, 1>(max_distance_sq.at(0, 1)) +
                           core::math::Pow<3, 1>(max_distance_sq.at(0, 2)) +
                           core::math::Pow<3, 1>(max_distance_sq.at(1, 2)));
      double denominator = core::math::Pow<5, 2>(
                             min_distance_sq.at(0, 1) *
                             min_distance_sq.at(0, 2) *
                             min_distance_sq.at(1, 2));

      return numerator / denominator / normalizing_constant_;
    }

    template<typename TableType>
    double maximum_negative_contribution(
      const core::gnp::TripleRangeDistanceSq<TableType> &range_in) const {

      const arma::mat &min_distance_sq = range_in.min_distance_sq();
      const arma::mat &max_distance_sq = range_in.max_distance_sq();
      double numerator = - 0.375 * (
                           core::math::Pow<3, 1>(min_distance_sq.at(0, 1)) +
                           core::math::Pow<3, 1>(min_distance_sq.at(0, 2)) +
                           core::math::Pow<3, 1>(min_distance_sq.at(1, 2)));
      double denominator = core::math::Pow<5, 2>(
                             max_distance_sq.at(0, 1) *
                             max_distance_sq.at(0, 2) *
                             max_distance_sq.at(1, 2));

      return numerator / denominator / normalizing_constant_;
    }

    template<typename TableType>
    double minimum_positive_contribution(
      const core::gnp::TripleRangeDistanceSq<TableType> &range_in) const {

      const arma::mat &min_distance_sq = range_in.min_distance_sq();
      const arma::mat &max_distance_sq = range_in.max_distance_sq();
      double first_numerator =
        3.0 *
        core::math::Sqr(min_distance_sq.at(0, 1)) *
        (min_distance_sq.at(0, 2) + min_distance_sq.at(1, 2));
      double second_numerator =
        3.0 * min_distance_sq.at(0, 2) * min_distance_sq.at(1, 2) *
        (min_distance_sq.at(0, 2) + min_distance_sq.at(1, 2));
      double third_numerator =
        min_distance_sq.at(0, 1) *
        (3.0 * core::math::Sqr(min_distance_sq.at(0, 2)) +
         2.0 * min_distance_sq.at(0, 2) * min_distance_sq.at(1, 2) +
         3.0 * core::math::Sqr(min_distance_sq.at(1, 2))) ;
      double denominator = 8.0 * core::math::Pow<5, 2>(
                             max_distance_sq.at(0, 1) *
                             max_distance_sq.at(0, 2) *
                             max_distance_sq.at(1, 2));

      return (first_numerator + second_numerator + third_numerator) /
             denominator / normalizing_constant_;
    }

    template<typename TableType>
    double maximum_positive_contribution(
      const core::gnp::TripleRangeDistanceSq<TableType> &range_in) const {

      const arma::mat &min_distance_sq = range_in.min_distance_sq();
      const arma::mat &max_distance_sq = range_in.max_distance_sq();
      double first_numerator =
        3.0 *
        core::math::Sqr(max_distance_sq.at(0, 1)) *
        (max_distance_sq.at(0, 2) + max_distance_sq.at(1, 2));
      double second_numerator =
        3.0 * max_distance_sq.at(0, 2) * max_distance_sq.at(1, 2) *
        (max_distance_sq.at(0, 2) + max_distance_sq.at(1, 2));
      double third_numerator =
        max_distance_sq.at(0, 1) *
        (3.0 * core::math::Sqr(max_distance_sq.at(0, 2)) +
         2.0 * max_distance_sq.at(0, 2) * max_distance_sq.at(1, 2) +
         3.0 * core::math::Sqr(max_distance_sq.at(1, 2))) ;
      double denominator = 8.0 * core::math::Pow<5, 2>(
                             min_distance_sq.at(0, 1) *
                             min_distance_sq.at(0, 2) *
                             min_distance_sq.at(1, 2));

      return (first_numerator + second_numerator + third_numerator) /
             denominator / normalizing_constant_;
    }

    template<typename TableType>
    void RangeUnnormOnSq(
      const core::gnp::TripleRangeDistanceSq<TableType> &range_in,
      core::math::Range *negative_range,
      core::math::Range *positive_range) const {

      // Compute the negative contribution bound.
      negative_range->Init(
        this->minimum_negative_contribution(range_in),
        this->maximum_negative_contribution(range_in));

      // Compute the positive contribution bound.
      positive_range->Init(
        this->minimum_positive_contribution(range_in),
        this->maximum_positive_contribution(range_in));
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
      return numerator / denominator / normalizing_constant_;
    }
};
}
}

#endif
