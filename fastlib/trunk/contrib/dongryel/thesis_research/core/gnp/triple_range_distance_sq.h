/** @file triple_range_distance_sq.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_TRIPLE_RANGE_DISTANCE_SQ_H
#define CORE_GNP_TRIPLE_RANGE_DISTANCE_SQ_H

#include <armadillo>
#include "core/math/math_lib.h"
#include "core/metric_kernels/abstract_metric.h"
#include "core/table/table.h"

namespace core {
namespace gnp {
template<typename TableType>
class TripleRangeDistanceSq {
  public:
    typedef typename TableType::TreeType TreeType;

  private:
    arma::mat min_distance_sq_;

    arma::mat max_distance_sq_;

    std::vector< TreeType * > nodes_;

    std::vector< double > num_tuples_;

  private:

    void ComputeNumTuples_(const TableType &table_in) {
      if(nodes_[0] == nodes_[1]) {

        // node_0 = node_1 = node_2
        if(nodes_[1] == nodes_[2]) {
          num_tuples_[0] =
            num_tuples_[1] =
              num_tuples_[2] =
                core::math::BinomialCoefficient<double>(
                  table_in.get_node_count(nodes_[0]) - 1, 2);
        }

        // node_0 = node_1, node_1 \not = node_2
        else {
          num_tuples_[0] = num_tuples_[1] =
                             (table_in.get_node_count(nodes_[0]) - 1) *
                             table_in.get_node_count(nodes_[2]);
          num_tuples_[2] = core::math::BinomialCoefficient<double>(
                             table_in.get_node_count(nodes_[0]), 2);
        }
      }
      else {

        // node_0 \not = node_1, node_1 = node_2
        if(nodes_[1] == nodes_[2]) {
          num_tuples_[1] = num_tuples_[2] =
                             (table_in.get_node_count(nodes_[1]) - 1) *
                             table_in.get_node_count(nodes_[0]);
          num_tuples_[0] = core::math::BinomialCoefficient<double>(
                             table_in.get_node_count(nodes_[1]), 2);
        }

        // node_0 \not = node_1, node_1 \not = node_2
        else {
          num_tuples_[0] = table_in.get_node_count(nodes_[1]) *
                           table_in.get_node_count(nodes_[2]);
          num_tuples_[1] = table_in.get_node_count(nodes_[0]) *
                           table_in.get_node_count(nodes_[2]);
          num_tuples_[2] = table_in.get_node_count(nodes_[0]) *
                           table_in.get_node_count(nodes_[1]);
        }
      }
    }

  public:

    double num_tuples(int node_index) const {
      return num_tuples_[node_index];
    }

    TreeType *node(int node_index) const {
      return nodes_[node_index];
    }

    TreeType *node(int node_index) {
      return nodes_[node_index];
    }

    void set_range_distance_sq(
      int first_node_index, int second_node_index,
      const core::math::Range &range_in) {

      min_distance_sq_.at(first_node_index, second_node_index) = range_in.lo;
      min_distance_sq_.at(second_node_index, first_node_index) = range_in.lo;
      max_distance_sq_.at(first_node_index, second_node_index) = range_in.hi;
      max_distance_sq_.at(second_node_index, first_node_index) = range_in.hi;
    }

    const arma::mat &min_distance_sq() const {
      return min_distance_sq_;
    }

    const arma::mat &max_distance_sq() const {
      return max_distance_sq_;
    }

    const std::vector< TreeType *> &nodes() const {
      return nodes_;
    }

    TripleRangeDistanceSq() {
      min_distance_sq_.set_size(3, 3);
      max_distance_sq_.set_size(3, 3);
      min_distance_sq_.fill(0.0);
      max_distance_sq_.fill(0.0);
      nodes_.resize(3);
      num_tuples_.resize(3);
    }

    TripleRangeDistanceSq(const TripleRangeDistanceSq &ranges_in) {
      min_distance_sq_ = ranges_in.min_distance_sq();
      max_distance_sq_ = ranges_in.max_distance_sq();
      nodes_ = ranges_in.nodes();
    }

    void ReplaceOneNodeBackward(
      const core::metric_kernels::AbstractMetric &metric_in,
      const TableType &table_in,
      TreeType *new_node_in,
      int node_index_in) {

      nodes_[node_index_in] = new_node_in;
      const typename TreeType::BoundType &new_node_bound =
        table_in.get_node_bound(new_node_in);

      for(int existing_node_index = node_index_in + 1;
          existing_node_index < 3; existing_node_index++) {

        // Change for the first existing node.
        core::math::Range existing_range_distance_sq =
          new_node_bound.RangeDistanceSq(
            metric_in,
            table_in.get_node_bound(nodes_[existing_node_index]));
        set_range_distance_sq(
          node_index_in, existing_node_index, existing_range_distance_sq);
      }

      // Recompute the number of tuples for each node.
      if(node_index_in == 0) {
        ComputeNumTuples_(table_in);
      }
    }

    void ReplaceOneNodeForward(
      const core::metric_kernels::AbstractMetric &metric_in,
      const TableType &table_in,
      TreeType *new_node_in,
      int node_index_in) {

      nodes_[node_index_in] = new_node_in;
      const typename TreeType::BoundType &new_node_bound =
        table_in.get_node_bound(new_node_in);

      for(int existing_node_index = 0; existing_node_index < node_index_in;
          existing_node_index++) {

        // Change for the first existing node.
        core::math::Range existing_range_distance_sq =
          new_node_bound.RangeDistanceSq(
            metric_in,
            table_in.get_node_bound(nodes_[existing_node_index]));
        set_range_distance_sq(
          node_index_in, existing_node_index, existing_range_distance_sq);
      }

      // Recompute the number of tuples for each node.
      if(node_index_in == 2) {
        ComputeNumTuples_(table_in);
      }
    }

    void Init(
      const core::metric_kernels::AbstractMetric &metric_in,
      const TableType &table,
      const std::vector< TreeType * > &nodes_in) {
      for(unsigned int j = 0; j < nodes_.size(); j++) {
        nodes_[j] = nodes_in[j];
      }
      for(unsigned int j = 0; j < nodes_.size(); j++) {
        const typename TreeType::BoundType &outer_bound =
          table.get_node_bound(nodes_[j]);
        for(unsigned int i = j + 1; i < nodes_.size(); i++) {
          const typename TreeType::BoundType &inner_bound =
            table.get_node_bound(nodes_[i]);
          core::math::Range range_distance_sq =
            outer_bound.RangeDistanceSq(metric_in, inner_bound);
          set_range_distance_sq(i, j, range_distance_sq);
        }
      }

      // Compute the number of tuples for each node.
      ComputeNumTuples_(table);
    }

    core::math::Range RangeDistanceSq(
      int first_index, int second_index) const {
      return core::math::Range(
               min_distance_sq_.at(first_index, second_index),
               max_distance_sq_.at(first_index, second_index));
    }
};
};
};

#endif
