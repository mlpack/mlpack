/** @file gen_metric.h
 *
 *  The generic metric tree builder.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_GEN_METRIC_TREE_H
#define CORE_TREE_GEN_METRIC_TREE_H

#include <vector>
#include "core/tree/ball_bound.h"
#include "core/tree/general_spacetree.h"
#include "core/table/dense_matrix.h"
#include "core/table/memory_mapped_file.h"

namespace core {
namespace tree {

/** @brief The general metric tree specification.
 */
template<typename IncomingStatisticType>
class GenMetricTree {
  public:

    typedef core::tree::BallBound BoundType;

    typedef IncomingStatisticType StatisticType;

  private:

    /** @brief Computes the furthest point from the given pivot and
     *         finds out the index.
     */
    template<typename MetricType>
    static int FurthestColumnIndex_(
      const MetricType &metric_in,
      const core::table::DensePoint &pivot,
      const core::table::DenseMatrix &matrix,
      int begin, int count,
      double *furthest_distance) {

      int furthest_index = -1;
      int end = begin + count;
      *furthest_distance = -1.0;

      for(int i = begin; i < end; i++) {
        core::table::DensePoint point;
        matrix.MakeColumnVector(i, &point);
        double distance_between_center_and_point =
          metric_in.Distance(pivot, point);

        if((*furthest_distance) < distance_between_center_and_point) {
          *furthest_distance = distance_between_center_and_point;
          furthest_index = i;
        }
      }
      return furthest_index;
    }

  public:

    template<typename MetricType>
    static void FindBoundFromMatrix(
      const MetricType &metric_in,
      const core::table::DenseMatrix &matrix,
      int first, int count, BoundType *bounds) {

      MakeLeafNode(metric_in, matrix, first, count, bounds);
    }

    /** @brief Makes a leaf node in the metric tree.
     */
    template<typename MetricType>
    static void MakeLeafNode(
      const MetricType &metric_in,
      const core::table::DenseMatrix& matrix,
      int begin, int count, BoundType *bounds) {

      bounds->center().SetZero();

      int end = begin + count;
      arma::vec bound_ref;
      core::table::DensePointToArmaVec(bounds->center(), &bound_ref);
      for(int i = begin; i < end; i++) {
        arma::vec col_point;
        matrix.MakeColumnVector(i, &col_point);
        bound_ref += col_point;
      }
      bound_ref = (1.0 / static_cast<double>(count)) * bound_ref;

      double furthest_distance;
      FurthestColumnIndex_(
        metric_in, bounds->center(), matrix, begin, count, &furthest_distance);
      bounds->set_radius(furthest_distance);
    }

    template<typename MetricType, typename TreeType>
    static void CombineBounds(
      const MetricType &metric_in,
      core::table::DenseMatrix &matrix,
      TreeType *node, TreeType *left, TreeType *right) {

      // Compute the weighted sum of the two pivots
      arma::vec bound_ref;
      core::table::DensePointToArmaVec(node->bound().center(), &bound_ref);
      arma::vec left_bound_ref;
      core::table::DensePointToArmaVec(left->bound().center(), &left_bound_ref);
      arma::vec right_bound_ref;
      core::table::DensePointToArmaVec(
        right->bound().center(), &right_bound_ref);
      bound_ref = left->count() * left_bound_ref +
                  right->count() * right_bound_ref;
      bound_ref =
        (1.0 / static_cast<double>(node->count())) * bound_ref;

      double left_max_dist, right_max_dist;
      FurthestColumnIndex_(
        metric_in, node->bound().center(), matrix, left->begin(),
        left->count(), &left_max_dist);
      FurthestColumnIndex_(
        metric_in, node->bound().center(), matrix, right->begin(),
        right->count(), &right_max_dist);
      node->bound().set_radius(std::max(left_max_dist, right_max_dist));
    }

    template<typename MetricType>
    static void ComputeMemberships(
      const MetricType &metric_in,
      const core::table::DenseMatrix &matrix,
      int first, int end,
      BoundType &left_bound, BoundType &right_bound,
      int *left_count, std::deque<bool> *left_membership) {

      for(int left = first; left < end; left++) {

        // Make alias of the current point.
        core::table::DensePoint point;
        matrix.MakeColumnVector(left, &point);

        // Compute the distances from the two pivots.
        double distance_from_left_pivot =
          metric_in.Distance(point, left_bound.center());
        double distance_from_right_pivot =
          metric_in.Distance(point, right_bound.center());

        // We swap if the point is further away from the left pivot.
        if(distance_from_left_pivot > distance_from_right_pivot) {
          (*left_membership)[left - first] = false;
        }
        else {
          (*left_membership)[left - first] = true;
          (*left_count)++;
        }
      }
    }

    template<typename MetricType, typename TreeType, typename IndexType>
    static bool AttemptSplitting(
      const MetricType &metric_in,
      core::table::DenseMatrix& matrix, TreeType *node, TreeType **left,
      TreeType **right, int leaf_size, IndexType *old_from_new,
      core::table::MemoryMappedFile *m_file_in) {

      // Pick a random row.
      int random_row = core::math::RandInt(
                         node->begin(), node->begin() + node->count());
      core::table::DensePoint random_row_vec;
      matrix.MakeColumnVector(random_row, & random_row_vec);

      // Now figure out the furthest point from the random row picked
      // above.
      double furthest_distance;
      int furthest_from_random_row =
        FurthestColumnIndex_(
          metric_in, random_row_vec, matrix, node->begin(), node->count(),
          &furthest_distance);
      core::table::DensePoint furthest_from_random_row_vec;
      matrix.MakeColumnVector(
        furthest_from_random_row, &furthest_from_random_row_vec);

      // Then figure out the furthest point from the furthest point.
      double furthest_from_furthest_distance;
      int furthest_from_furthest_random_row =
        FurthestColumnIndex_(
          metric_in, furthest_from_random_row_vec, matrix, node->begin(),
          node->count(), &furthest_from_furthest_distance);
      core::table::DensePoint furthest_from_furthest_random_row_vec;
      matrix.MakeColumnVector(
        furthest_from_furthest_random_row,
        &furthest_from_furthest_random_row_vec);

      if(furthest_from_furthest_distance <
          std::numeric_limits<double>::epsilon()) {
        return false;
      }
      else {
        *left = (m_file_in) ?
                m_file_in->Construct<TreeType>() : new TreeType();
        *right = (m_file_in) ?
                 m_file_in->Construct<TreeType>() : new TreeType();

        ((*left)->bound().center()).Copy(furthest_from_random_row_vec);
        ((*right)->bound().center()).Copy(
          furthest_from_furthest_random_row_vec);

        int left_count = TreeType::MatrixPartition(
                           metric_in, matrix, node->begin(), node->count(),
                           (*left)->bound(), (*right)->bound(), old_from_new);

        (*left)->Init(node->begin(), left_count);
        (*right)->Init(node->begin() + left_count, node->count() - left_count);
      }

      return true;
    }
};
}
}

#endif
