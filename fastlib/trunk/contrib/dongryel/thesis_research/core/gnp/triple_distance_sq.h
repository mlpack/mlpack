/** @file triple_distance_sq.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_TRIPLE_DISTANCE_SQ_H
#define CORE_GNP_TRIPLE_DISTANCE_SQ_H

#include <armadillo>
#include "core/metric_kernels/abstract_metric.h"
#include "core/table/table.h"
#include "core/table/dense_point.h"

namespace core {
namespace gnp {
class TripleDistanceSq {
  private:
    arma::mat distance_sq_;

    std::vector< core::table::DenseConstPoint > points_;

  public:

    void set_distance_sq(
      int first_point_pos, int second_point_pos,
      double squared_distance) {

      distance_sq_.at(first_point_pos, second_point_pos) = squared_distance;
      distance_sq_.at(second_point_pos, first_point_pos) = squared_distance;
    }

    TripleDistanceSq() {
      distance_sq_.set_size(3, 3);
      distance_sq_.fill(0.0);
      points_.resize(3);
    }

    double distance_sq(int first_pos, int second_pos) const {
      return distance_sq_.at(first_pos, second_pos);
    }

    void ReplaceOnePoint(
      const core::metric_kernels::AbstractMetric &metric_in,
      const core::table::DenseConstPoint &new_point_in,
      int point_pos_in) {

      points_[point_pos_in].Alias(new_point_in);

      for (int existing_point_pos = point_pos_in - 1;
           existing_point_pos >= 0; existing_point_pos--) {

        // Change for the first existing point.
        metric_in.DistanceSq(new_point_in, points_[existing_point_pos]);
        set_distance_sq(
          point_pos_in, existing_point_pos,
          metric_in.DistanceSq(new_point_in, points_[existing_point_pos]));
      }
    }
};
};
};

#endif
