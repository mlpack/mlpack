/** @file triple_distance_sq.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_TRIPLE_DISTANCE_SQ_H
#define CORE_GNP_TRIPLE_DISTANCE_SQ_H

#include <armadillo>
#include "core/table/table.h"
#include "core/table/dense_point.h"

namespace core {
namespace gnp {
class TripleDistanceSq {
  private:
    arma::mat distance_sq_;

    std::vector< core::table::DensePoint > points_;

    std::vector<int> point_indices_;

  public:

    void PrintPoints() const {
      printf("TripleDistanceSq object has the following points:\n");
      for(unsigned int i = 0; i < points_.size(); i++) {
        points_[i].Print();
      }
    }

    int point_index(int index_in) const {
      return point_indices_[index_in];
    }

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
      point_indices_.resize(3);
    }

    double distance_sq(int first_pos, int second_pos) const {
      return distance_sq_.at(first_pos, second_pos);
    }

    template<typename MetricType>
    void ReplaceOnePoint(
      const MetricType &metric_in,
      const core::table::DensePoint &new_point_in,
      int new_point_index_in,
      int point_pos_in) {

      points_[point_pos_in].Alias(new_point_in);

      // Replace the index with the new point.
      point_indices_[point_pos_in] = new_point_index_in;

      // Recompute the distances versus the set of existing points.
      if(point_pos_in > 0) {
        set_distance_sq(
          point_pos_in, 0,
          metric_in.DistanceSq(new_point_in, points_[0]));
      }
      if(point_pos_in > 1) {
        set_distance_sq(
          point_pos_in, 1,
          metric_in.DistanceSq(new_point_in, points_[1]));
      }
    }
};
}
}

#endif
